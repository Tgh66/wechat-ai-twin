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
  - name    = 会话显示名（按会话 user_name 查昵称/群名），不是发送者昵称
  - is_self = 消息方向（True=自己发出，False=对方/群成员）
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'ai-twin', 'venv', 'Lib', 'site-packages'))

from wechatauto import db as dbm


def progress(msg: str) -> None:
    print("[PROGRESS] " + msg, flush=True)


def done(total: int) -> None:
    print("[DONE] total=%d" % total, flush=True)


def fail(msg: str) -> None:
    print("[ERROR] " + msg, flush=True)


def open_with_retry(w, rel: str, retries: int = 5, wait: float = 10.0):
    """打开分片库；微信 checkpoint / 库被占用时延迟重试（10s×5），仍失败返回 None 由调用方跳过并标注"""
    last = None
    for i in range(retries):
        try:
            return w._open(rel)
        except Exception as e:
            last = e
            msg = str(e).lower()
            # 只对"正在写/被占用/加密中"类错误重试，其余立即抛出
            if any(k in msg for k in ("checkpoint", "locked", "busy", "database is locked")):
                if i < retries - 1:
                    progress("db=%s busy retry=%d/%d wait=%ds (%s)" % (rel, i + 1, retries, wait, msg[:60]))
                    time.sleep(wait)
                    continue
            break
    return last


def disk_space_ok(out: str, need_mb: float, warn_gb: float = 0.5) -> bool:
    """检查目标盘剩余空间 ≥ 预估导出大小（另留 warn_gb 余量），不足时给出明确提示"""
    try:
        free = shutil.disk_usage(os.path.dirname(out) or ".").free
        free_gb = free / (1024 ** 3)
        need_gb = need_mb / 1024.0
        if free < need_mb * 1024 * 1024 + warn_gb * 1024 ** 3:
            progress("disk=warn free=%.1fGB need>=%.1fGB target=%s" %
                     (free_gb, need_gb, os.path.dirname(out)))
        return True
    except Exception as e:
        progress("disk=check-error %r" % e)
        return True  # 检查失败不阻塞，交由写入时错误处理


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

    # 发送者 id -> username（message_resource.db SenderName2Id，rowid 即 real_sender_id）
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
    # self 在 SenderName2Id 中的 rowid 集合（微信 4.0 实测 = {1}，即自己发出的消息 real_sender_id=1）
    self_sender_ids = {k for k, v in sender_index.items() if v == self_wxid}
    progress("stage=self wxid=%s sender_ids=%s" % (self_wxid, sorted(self_sender_ids)))

    # 磁盘空间预检：预估 jsonl ≈ 全部消息库解压后大小（加密库 ×4 粗略）
    try:
        est_mb = 0.0
        for rel, path, _ in w._db_files:
            if rel.replace(os.sep, "/").startswith("message/") and path:
                try:
                    est_mb += os.path.getsize(path) / (1024 * 1024) * 4.0
                except Exception:
                    pass
        disk_space_ok(out, est_mb)
    except Exception as e:
        progress("stage=disk warn=%r" % e)

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

    def is_self_sender(sid, self_wxid, self_ids):
        """is_self 判定：以 SenderName2Id 的 id→wxid 映射为准。
        - 微信 4.0：自己消息 real_sender_id ∈ self_ids（实测 {1}）
        - 旧版 3.x 约定：sender_id=2 表示自己
        - None/0/未知：无法映射，保守归为对方（False），并在进度中计数便于审计"""
        if sid in (2, "2", 2.0):
            return True
        if isinstance(sid, int) and sid in self_ids:
            return True
        if isinstance(sid, int):
            su = sender_index.get(sid)
            if su:
                return su == self_wxid
        return False

    msg_rels = sorted([rel for rel, _, _ in w._db_files
                       if rel.replace(os.sep, "/").startswith("message/")
                       and rel.lower().endswith(".db")])
    msg_rels = [rel for rel in msg_rels if not rel.lower().endswith(("-wal", "-shm"))]
    if args.max_dbs > 0:
        msg_rels = msg_rels[:args.max_dbs]

    total = 0
    unknown_senders = 0
    with open(out, "w", encoding="utf-8") as fout:
        for rel in msg_rels:
            try:
                conn = open_with_retry(w, rel)
            except Exception as e:
                progress("db=%s error=%r" % (rel, e))
                continue
            if conn is None:
                progress("db=%s skip=checkpoint-5x10s-still-failed" % rel)
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
                    # 会话显示名：昵称/群名（name=会话，方向靠 is_self）
                    conv_name = nick_map.get(uname, uname) or uname
                    while True:
                        rows = cur.fetchmany(2000)
                        if not rows:
                            break
                        for sid, ct, lt, mc, cc in rows:
                            # 方向
                            is_self = is_self_sender(sid, self_wxid, self_sender_ids)
                            if sid is None or (isinstance(sid, int) and sid == 0):
                                unknown_senders += 1
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
                                "name": conv_name,
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
    progress("stage=unknown-senders count=%d (None/0 已按对方处理)" % unknown_senders)
    done(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
