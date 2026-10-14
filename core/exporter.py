# -*- coding: utf-8 -*-
"""导出微信聊天记录（基于 wechatauto，自动提取内存密钥）"""
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def infer_self_wxid(db):
    """从数据库句柄获取自己的 wxid（优先 db.wxid，其次账号目录名去后缀）"""
    try:
        if getattr(db, "wxid", None):
            return db.wxid
        acct = getattr(db, "account", None)
        if acct:
            m = re.match(r"(wxid_[a-zA-Z0-9]+)", acct)
            if m:
                return m.group(1)
    except Exception:
        pass
    return ""


def load_db(db_dir=None):
    """加载微信数据库句柄；db_dir 可指定微信数据根目录（默认自动检测）"""
    from wechatauto import WeChatDB
    if db_dir:
        db = WeChatDB(db_dir=db_dir)
    else:
        db = WeChatDB()
    return db


def list_chats(db):
    """列出会话：username, name, message_count"""
    try:
        chats = db.list_message_chats()
    except Exception as e:
        return [], "列会话失败: %s" % e
    out = []
    for c in chats:
        out.append({
            "username": c.get("username", ""),
            "name": c.get("name", "") or c.get("username", ""),
            "message_count": c.get("message_count") or 0,
        })
    return out, ""


def _is_self(sender, self_wxid):
    if not sender or not self_wxid:
        return 0
    if sender == self_wxid:
        return 1
    # 兼容 wxid + 变体后缀
    return 1 if sender.startswith(self_wxid + "_") else 0


def export_chats(db, chat_list, out_path, progress_cb=None, self_wxid=None):
    """导出选中的会话到 JSONL。
    chat_list: [{"username":.., "name":..}, ...]
    self_wxid: 自己的 wxid（用于标记 is_self），None 时自动推断
    """
    if self_wxid is None:
        self_wxid = infer_self_wxid(db)
    t0 = time.time()
    total_msgs = 0
    self_msgs = 0
    name_map = {c["username"]: c["name"] for c in chat_list}
    with open(out_path, "w", encoding="utf-8") as f:
        for ui, chat in enumerate(chat_list):
            user = chat["username"]
            offset = 0
            batch = 2000
            while True:
                try:
                    msgs = db.get_messages(user, limit=batch, offset=offset)
                except Exception as e:
                    if progress_cb:
                        progress_cb(ui + 1, len(chat_list), "会话 %s 出错: %s" % (user, e))
                    break
                if not msgs:
                    break
                for m in msgs:
                    total_msgs += 1
                    sender = m.get("sender_username", "") or ""
                    is_self = _is_self(sender, self_wxid)
                    if is_self:
                        self_msgs += 1
                    rec = {
                        "user": user,
                        "name": name_map.get(user, user),
                        "sender": sender,
                        "is_self": is_self,
                        "type": m.get("type", ""),
                        "ts": m.get("create_time", 0),
                        "content": m.get("content", ""),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                offset += len(msgs)
                if len(msgs) < batch:
                    break
            if progress_cb:
                progress_cb(ui + 1, len(chat_list), "已导出 %s / %s 个会话" % (ui + 1, len(chat_list)))
    return {
        "path": out_path,
        "total": total_msgs,
        "self": self_msgs,
        "chats": len(chat_list),
        "elapsed": round(time.time() - t0, 1),
        "self_wxid": self_wxid,
    }
