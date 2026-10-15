# -*- coding: utf-8 -*-
"""数据清洗与训练集构建（对话上下文 → 自己回复）"""
import json
import os
import random
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DEFAULT_NOISE_KW = [
    "[图片]", "[表情]", "[视频]", "[语音]", "[文件]", "[链接]", "[动画表情]",
    "拍了拍", "撤回了一条消息", "开启了朋友验证", "邀请你加入群聊", "发起了语音通话",
    "发起了视频通话", "转账", "收款", "该消息已删除", "请先添加对方为好友",
]
CUSTOMER_KW = ["客服", "接单", "代做", "下单", "报价", "开发", "软件", "程序", "咸鱼"]


def make_filter(min_len=2, max_len=800, filter_noise=True, noise_kw=None):
    kw = noise_kw if noise_kw is not None else DEFAULT_NOISE_KW

    def filt(c):
        if not c or not c.strip():
            return True
        if len(c) < min_len or len(c) > max_len:
            return True
        if filter_noise and any(k in c for k in kw):
            return True
        return False
    return filt


def build_dataset(jsonl_path, out_dir, min_len=2, max_len=800, filter_noise=True,
                  max_samples=15000, balance=True, long_first=True,
                  long_threshold=40, context_rounds=3, context_messages=8,
                  val_ratio=0.1, seed=42, self_wxid=""):
    """从导出 JSONL 构建 train/val（messages 格式）
    返回统计 dict
    """
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")

    def _is_self(r):
        v = r.get("is_self")
        if v is not None:
            return bool(v)
        if self_wxid:
            s = r.get("sender", "") or ""
            return s == self_wxid or s.startswith(self_wxid + "_")
        return False

    filt = make_filter(min_len, max_len, filter_noise)
    chats = defaultdict(list)  # chat_key -> [(ts, is_self, content)]
    total_raw = 0
    self_raw = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total_raw += 1
            c = r.get("content", "") or ""
            if r.get("type") != "文本":
                continue
            if filt(c):
                continue
            is_self = _is_self(r)
            if is_self:
                self_raw += 1
            key = r.get("name") or r.get("user") or "?"
            chats[key].append((r.get("ts") or 0, is_self, c))

    def cap_for(name):
        if balance:
            if any(k in name for k in ("爱妻", "老婆", "女友", "对象")):
                return 6000
            if any(k in name for k in CUSTOMER_KW) or "客服" in name or "群" in name or "@chatroom" in name:
                return 400
            return 500
        return 10 ** 9

    random.seed(seed)
    samples = []
    for name, msgs in chats.items():
        msgs.sort(key=lambda x: x[0])
        cap = cap_for(name)
        cands = []
        for i in range(len(msgs)):
            ts, is_self, c = msgs[i]
            if not is_self:
                continue
            seq = msgs[max(0, i - context_messages): i + 1]
            roles = [("user" if not s else "assistant", cc) for _, s, cc in seq]
            if not any(r == "user" for r, _ in roles):
                continue
            if roles[-1][0] != "assistant":
                continue
            merged = []
            for r, cc in roles:
                if merged and merged[-1][0] == r:
                    merged[-1] = (r, merged[-1][1] + "\n" + cc)
                else:
                    merged.append((r, cc))
            messages = [{"role": r, "content": cc} for r, cc in merged]
            cands.append(messages)
        if not cands:
            continue
        cands = cands[::-1]  # 最近优先
        chosen = []
        n_long = 0
        n_short = 0
        for m in cands:
            last = m[-1]["content"]
            if long_first and len(last) >= long_threshold:
                if n_long >= min(1200, cap * 3):
                    continue
                n_long += 1
            else:
                if n_short >= cap:
                    continue
                n_short += 1
            chosen.append(m)
        samples.append((name, chosen))

    long_s = []
    short_s = []
    for name, chosen in samples:
        for m in chosen:
            item = {"messages": m, "chat": name}
            if len(m[-1]["content"]) >= long_threshold:
                long_s.append(item)
            else:
                short_s.append(item)
    random.shuffle(short_s)
    all_s = long_s + short_s
    if len(all_s) > max_samples:
        all_s = all_s[:max_samples]
    random.shuffle(all_s)

    by_chat = defaultdict(list)
    for s in all_s:
        by_chat[s["chat"]].append(s)
    train, val = [], []
    for chat, items in by_chat.items():
        random.shuffle(items)
        nv = max(1, len(items) // 10) if len(items) >= 10 else 0
        val.extend(items[:nv])
        train.extend(items[nv:])
    random.shuffle(train)
    random.shuffle(val)

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")

    lens = sorted(len(s["messages"][-1]["content"]) for s in train)
    def pct(p):
        if not lens:
            return 0
        return lens[min(len(lens) - 1, int(len(lens) * p))]
    return {
        "jsonl": jsonl_path,
        "raw_total": total_raw,
        "self_raw": self_raw,
        "chat_count": len(chats),
        "samples": len(all_s),
        "train": len(train),
        "val": len(val),
        "long_reply": len(long_s),
        "reply_len_p50": pct(0.5),
        "reply_len_p90": pct(0.9),
        "train_path": train_path,
        "val_path": val_path,
    }
