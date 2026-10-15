#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""微信 4.0 加密聊天库自动解密导出（基于 wechatauto-replica，密钥从微信进程提取）

用法:
  python decrypt_wechat.py [--db-dir DIR] [--out OUT.jsonl] [--keys-dir DIR]

流程:
  1. 实例化 WeChatDB（自动提取密钥，27 库全量）
  2. 读 message_resource.db 的 SenderName2Id（real_sender_id -> 用户名）
  3. 读 contact.db 昵称映射
  4. 遍历 message_0..N.db 全部 Msg_<md5> 会话表，导出 JSONL
  5. stdout 输出 [PROGRESS] / [DONE] / [ERROR] 供宿主程序解析

JSONL 字段与现有索引一致: name/user/is_self/ts/type/content
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'ai-twin', 'venv', 'Lib', 'site-packages'))

from wechatauto import db as dbm


def progress(msg: str) -> None:
    print("[PROGRESS] " + msg, flush=True)


def done(total: int) -> None:
    print("[DONE] total=%d" % total, flush=True)


def fail(msg: str) -> None:
    print("[ERROR] " + msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keys-dir", default=None)
    ap.add_argument("--max-dbs", type=int, default=0, help="测试用：最多处理几个消息库")
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'wechat_decrypt.jsonl')
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    keys_dir = args.keys_dir or os.path.join(tempfile.gettempdir(), 'wct_export_keys')
    os.makedirs(keys_dir, exist_ok=True)

    try:
        w = dbm.WeChatDB(
            db_dir=args.db_dir,
            workdir=keys_dir,
            keys_file=os.path.join(keys_dir, 'keys.json'),
        )
    except Exception as e:
        fail("初始化失败: %r" % e)
        return 2
    progress("stage=keys status=ok account=%s keys=%d" % (w.account, len(w._keys)))
    if not w._keys:
        fail("未提取到任何数据库密钥，请确认微信已登录且窗口保持打开")
        return 3

    # 发送者 id -> username（message_resource.db SenderName2Id）
    try:
        sender_index = w._sender_id_index()
    except Exception as e:
        sender_index = {}
        progress("stage=sender_index warn=%r" % e)
    progress("stage=sender_index rows=%d" % len(sender_index))

    # 昵称映射 username -> 显示名
    try:
        nick_map = w.nickname_map(refresh=True)
    except Exception as e:
        nick_map = {}
        progress("stage=nickname warn=%r" % e)
    progress("stage=nickname rows=%d" % len(nick_map))

    self_wxid = w.wxid
    progress("stage=self wxid=%s" % self_wxid)

    # 会话表名 -> user_name（每个分片库 Name2Id）
    def build_table_map(conn):
        m = {}
        try:
            for (uname,) in conn.execute("SELECT user_name FROM Name2Id"):
                if uname:
                    m["Msg_" + hashlib.md5(uname.encode()).hexdigest()] = uname
        except Exception:
            pass
        return m

    msg_rels = sorted([rel for rel, _, _ in w._db_files
                       if rel.replace(os.sep, "/").startswith("message/")
                       and rel.lower().endswith(".db")])
    msg_rels = [rel for rel in msg_rels if not rel.lower().endswith(("-wal", "-shm"))]
    if args.max_dbs > 0:
        msg_rels = msg_rels[:args.max_dbs]

    total = 0
    with open(out, "w", encoding="utf-8") as fout:
        for rel in msg_rels:
            try:
                conn = w._open(rel)
            except Exception as e:
                progress("db=%s error=%r" % (rel, e))
                continue
            try:
                tmap = build_table_map(conn)
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Msg_%'")]
                nrows = 0
                for t in tables:
                    uname = tmap.get(t)
                    if not uname:
                        continue  # 无会话映射的表跳过
                    try:
                        cur = conn.execute(
                            "SELECT real_sender_id, create_time, local_type, message_content, "
                            "compress_content FROM %s" % t)
                    except Exception:
                        continue
                    while True:
                        rows = cur.fetchmany(2000)
                        if not rows:
                            break
                        for sid, ct, lt, mc, cc in rows:
                            # 发送者
                            is_self = False
                            sender_name = ""
                            if sid in (2, "2", 2.0, None) or str(sid) == "2":
                                is_self = True
                                sender_name = nick_map.get(self_wxid, "我")
                            else:
                                su = sender_index.get(int(sid), "") if sid else ""
                                if su == self_wxid:
                                    is_self = True
                                    sender_name = nick_map.get(self_wxid, "我")
                                else:
                                    sender_name = nick_map.get(su, su) if su else str(sid)
                            # 类型
                            mtype = dbm.WeChatDB._msg_type_name(lt) if lt is not None else "其他"
                            # 内容
                            content = mc
                            if isinstance(content, bytes):
                                content = dbm.WeChatDB._friendly_content(content, mtype)
                            if isinstance(content, bytes):
                                content = content.decode("utf-8", "replace")
                            if not content or not str(content).strip():
                                if isinstance(cc, bytes) and cc:
                                    cc_text = dbm.WeChatDB._friendly_content(cc, mtype)
                                    if cc_text and str(cc_text).strip():
                                        content = cc_text
                            content = str(content).strip() if content else ""
                            if not content:
                                content = "[%s]" % mtype
                            # 时间：毫秒 -> 秒
                            ts = int(ct) if ct else 0
                            if ts > 1e12:
                                ts //= 1000
                            rec = {
                                "name": sender_name,
                                "user": uname,
                                "is_self": is_self,
                                "ts": ts,
                                "type": mtype,
                                "content": content,
                            }
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            nrows += 1
                    progress("db=%s table=%s rows=%d" % (rel, t, nrows))
                total += nrows
                conn.close()
                progress("db=%s done rows=%d" % (rel, nrows))
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                progress("db=%s error=%r" % (rel, e))
    done(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
