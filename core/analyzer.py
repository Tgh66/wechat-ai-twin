# -*- coding: utf-8 -*-
"""统计分析 + 情感分析"""
import collections
import datetime
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False


def _is_self(m, self_wxid=""):
    v = m.get("is_self")
    if v is not None:
        return bool(v)
    if self_wxid:
        s = m.get("sender", "") or ""
        return s == self_wxid or s.startswith(self_wxid + "_")
    return False


def load_messages(jsonl_path, limit=None):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def analyze_chat(messages, self_prefix="wxid_", self_wxid=""):
    """整体统计：返回 dict + 可直接用于图表的数据"""
    total = len(messages)
    self_cnt = 0
    other_cnt = 0
    type_cnt = collections.Counter()
    by_chat = collections.Counter()
    hour_cnt = [0] * 24
    day_cnt = collections.Counter()
    month_cnt = collections.Counter()
    self_lens = []
    ts_list = []
    content_lens = []
    for m in messages:
        is_self = _is_self(m, self_wxid)
        if is_self:
            self_cnt += 1
        else:
            other_cnt += 1
        type_cnt[m.get("type", "")] += 1
        by_chat[m.get("name", "?")] += 1
        c = m.get("content", "") or ""
        content_lens.append(len(c))
        ts = m.get("ts") or 0
        if ts:
            ts_list.append(ts)
            dt = datetime.datetime.fromtimestamp(ts)
            hour_cnt[dt.hour] += 1
            day_cnt[dt.date().isoformat()] += 1
            month_cnt[dt.strftime("%Y-%m")] += 1
        if is_self:
            self_lens.append(len(c))
    self_lens.sort()
    content_lens.sort()

    def pct(arr, p):
        if not arr:
            return 0
        return arr[min(len(arr) - 1, int(len(arr) * p))]

    span_days = 0
    first_ts = min(ts_list) if ts_list else 0
    last_ts = max(ts_list) if ts_list else 0
    if first_ts and last_ts:
        span_days = max(1, int((last_ts - first_ts) / 86400) + 1)

    return {
        "total": total, "self": self_cnt, "other": other_cnt,
        "chats": len(by_chat),
        "type_cnt": dict(type_cnt.most_common(8)),
        "by_chat": by_chat.most_common(20),
        "hour_cnt": hour_cnt,
        "day_cnt": day_cnt,
        "month_cnt": dict(sorted(month_cnt.items())),
        "active_days": len(day_cnt),
        "span_days": span_days,
        "first_ts": first_ts, "last_ts": last_ts,
        "self_len_p50": pct(self_lens, 0.5), "self_len_p90": pct(self_lens, 0.9),
        "content_len_p50": pct(content_lens, 0.5),
    }


def chat_series(messages, chat_name, self_wxid=""):
    """取单个会话的消息序列（按时间排序）"""
    rows = [m for m in messages if m.get("name") == chat_name or m.get("user") == chat_name]
    rows.sort(key=lambda x: x.get("ts") or 0)
    return rows


def top_keywords(messages, topn=30, self_only=True, self_wxid=""):
    if not _HAS_JIEBA:
        return [("（需要安装 jieba）", 0)]
    stop = set("的了是在我和你有就不人都一个上也很到说要去你会着看好自己这那啊吧呢吗嘛嗯哦哈哈哈哈哈".strip())
    cnt = collections.Counter()
    for m in messages:
        if self_only:
            if not _is_self(m, self_wxid):
                continue
        c = m.get("content", "") or ""
        for w in jieba.cut(c):
            w = w.strip()
            if len(w) < 2 or w in stop or not any('\u4e00' <= ch <= '\u9fff' for ch in w):
                continue
            cnt[w] += 1
    return cnt.most_common(topn)


def reply_latency(messages, chat_name, self_wxid=""):
    """对方发言 → 自己回复 的时间间隔（秒）中位数"""
    rows = chat_series(messages, chat_name, self_wxid)
    gaps = []
    last_other_ts = None
    for m in rows:
        is_self = _is_self(m, self_wxid)
        ts = m.get("ts") or 0
        if not is_self:
            last_other_ts = ts
        elif last_other_ts and ts and 0 < ts - last_other_ts < 3600 * 6:
            gaps.append(ts - last_other_ts)
            last_other_ts = None
    if not gaps:
        return 0
    gaps.sort()
    return gaps[len(gaps) // 2]


# ---------------- 情感分析 ----------------

def sentiment_with_twin(chat_text, who="我", other="对方", max_len=600):
    """用微调好的数字分身模型分析对话情感"""
    from core import chat
    text = chat_text[:max_len]
    prompt = (
        "这是%s与%s的微信聊天记录片段：\n---\n%s\n---\n"
        "请分析：1) %s当时的主要情绪；2) %s对%s的感受；3) 整体情感倾向（正面/负面/中性，-100~100打分）。用简短自然的中文回答。"
        % (who, other, text, who, who, other))
    return chat.generate(prompt, system_prompt="你是情绪分析师，读对话判断说话人的真实感受。",
                         max_new_tokens=260, temperature=0.6)


class SentimentPipeline:
    """专用情感分类小模型（可选下载）"""

    def __init__(self, model_name):
        self.model_name = model_name
        from transformers import pipeline
        self._pipe = pipeline("sentiment-analysis", model=model_name,
                              tokenizer=model_name, device=0 if _cuda_ok() else -1)

    def predict(self, text):
        r = self._pipe(text[:512])[0]
        return r["label"], round(r["score"], 4)


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def sentiment_trend(messages, chat_name=None, month=True, self_wxid=""):
    """基于规则+关键词的轻量情感倾向（不依赖模型）：按月返回 正面/负面 计数"""
    pos = {"哈哈", "开心", "好", "棒", "牛", "可以", "行", "没问题", "爱", "喜欢", "期待",
           "笑死", "太强", "优秀", "美滋滋", "爽", "顺利", "恭喜", "谢谢", "感谢", "晚安", "么么"}
    neg = {"烦", "累", "难受", "气", "烦死", "无语", "崩溃", "哭", "难", "亏", "糟",
           "讨厌", "伤心", "难过", "焦虑", "压力", "坑", "生气", "不行", "完了"}
    rows = chat_series(messages, chat_name, self_wxid) if chat_name else messages
    cnt = collections.Counter()
    for m in rows:
        if not _is_self(m, self_wxid):
            continue
        c = (m.get("content") or "")[:200]
        key = datetime.datetime.fromtimestamp(m.get("ts") or 0).strftime("%Y-%m") if m.get("ts") else "?"
        s = 0
        for w in pos:
            if w in c:
                s += 1
        for w in neg:
            if w in c:
                s -= 1
        cnt[(key, "pos" if s > 0 else "neg" if s < 0 else "neu")] += 1
    return cnt


def sentiment_score(messages, chat_name=None, self_wxid=""):
    """会话级情感倾向度：-100（非常负面）~ +100（非常正面）"""
    pos = {"哈哈", "开心", "好", "棒", "牛", "可以", "行", "没问题", "爱", "喜欢", "期待",
           "笑死", "太强", "优秀", "美滋滋", "爽", "顺利", "恭喜", "谢谢", "感谢", "晚安", "么么",
           "嘻嘻", "嘿嘿", "开心心", "耶"}
    neg = {"烦", "累", "难受", "气", "烦死", "无语", "崩溃", "哭", "难", "亏", "糟",
           "讨厌", "伤心", "难过", "焦虑", "压力", "坑", "生气", "不行", "完了", "叹气"}
    rows = chat_series(messages, chat_name, self_wxid) if chat_name else messages
    s = 0
    n = 0
    for m in rows:
        if not _is_self(m, self_wxid):
            continue
        c = (m.get("content") or "")[:200]
        if not c.strip():
            continue
        n += 1
        for w in pos:
            if w in c:
                s += 1
        for w in neg:
            if w in c:
                s -= 1
    if n == 0:
        return 0, 0
    # 归一化到 [-100, 100]，每次发言 ±1（同句多词不叠加爆表）
    score = max(-100, min(100, s / max(1, n) * 40))
    return round(score, 1), n
