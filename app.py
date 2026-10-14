# -*- coding: utf-8 -*-
"""
微信 AI 数字分身工作台
======================
一键完成：自动发现微信数据 → 导出聊天记录 → 数据清洗 → LoRA 微调 → 对话 → 情感分析

启动：python app.py  然后浏览器打开 http://127.0.0.1:7860
以管理员身份运行可访问受保护目录（双击 run.bat）
"""
import json
import os
import queue
import sys
import threading
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(APP_DIR, "models")
OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

import gradio as gr
from core import discover, exporter, cleaner, gpu, trainer as tr, chat as chatmod, analyzer

# ---------------- 全局状态 ----------------
STATE = {
    "db_dir": None,
    "db": None,
    "export_path": None,
    "train_json": None,
    "val_json": None,
    "adapter_dir": None,
    "merged_dir": None,
    "messages_cache": None,
}

DEFAULT_SYSTEM = "你是微信用户\"我\"的数字分身。你正在微信上和好友聊天，请用我真实、自然的说话方式、语气和习惯回复对方。简洁直接，不用解释，不用客套。"


def _default_export_path():
    return os.path.join(DATA_DIR, "chat_messages.jsonl")


def _default_train():
    return os.path.join(DATA_DIR, "train.jsonl")


def _default_val():
    return os.path.join(DATA_DIR, "val.jsonl")


# ================= Tab1 数据发现 =================
def do_discover():
    found = discover.discover_wechat_dirs()
    accs = []
    for d, src in found:
        accs += discover.account_dirs(d)
    text = discover.format_discovery(found, accs)
    choices = [d for d, _ in found] or []
    running = discover.is_wechat_running()
    hint = ""
    if not running:
        hint = "⚠ 未检测到正在运行的微信。请先登录微信（电脑版）再导出（需要从微信进程内存提取密钥）。"
    return text, gr.update(choices=choices, value=choices[0] if choices else None), hint


def do_manual_folder(path):
    if not path:
        return "未选择文件夹", gr.update(choices=[]), ""
    accs = discover.account_dirs(path)
    n_msg = 0
    for a in accs:
        m = os.path.join(a, "db_storage", "message")
        if os.path.isdir(m):
            n_msg += len([f for f in os.listdir(m) if f.endswith(".db")])
    STATE["db_dir"] = path
    return ("已选择: %s\n账号目录 %d 个，消息库文件 %d 个" % (path, len(accs), n_msg),
            gr.update(choices=[path], value=path), "")


def do_scan_db(db_dir):
    if not db_dir:
        return "请先选择微信数据根目录", []
    STATE["db_dir"] = db_dir
    try:
        db = exporter.load_db(db_dir)
        STATE["db"] = db
        return "数据库句柄获取成功（微信需在运行中）", []
    except Exception as e:
        return "连接失败: %s\n请确认：① 微信已登录运行；② 目录正确；③ 必要时以管理员身份运行。" % e, []


# ================= Tab2 导出 =================
def do_load_chats():
    if STATE["db"] is None:
        return [], "请先在「数据发现」页扫描数据库", ""
    chats, err = exporter.list_chats(STATE["db"])
    if err:
        return [], "加载失败: " + err, ""
    rows = [[c["name"], c["username"], c["message_count"]] for c in chats]
    return rows, "共 %d 个会话" % len(chats), ""


def do_export(scope_mode, keyword, single_name, out_path, self_wxid, progress=gr.Progress()):
    if STATE["db"] is None:
        return "请先扫描数据库", None
    chats, err = exporter.list_chats(STATE["db"])
    if err or not chats:
        return "会话列表为空: %s" % err, None
    if scope_mode == "全部会话":
        sel = chats
    elif scope_mode == "关键词筛选":
        kws = [k.strip() for k in keyword.replace("，", ",").split(",") if k.strip()]
        if not kws:
            return "请输入筛选关键词", None
        exclude = [k for k in kws if k.startswith("-")]
        include = [k for k in kws if not k.startswith("-")]
        sel = []
        for c in chats:
            nm = c["name"]
            if include and not any(k in nm for k in include):
                continue
            if any(k in nm for k in exclude):
                continue
            sel.append(c)
    else:  # 单个会话
        hit = [c for c in chats if c["name"] == single_name or c["username"] == single_name]
        if not hit:
            return "未找到会话: %s" % single_name, None
        sel = [hit[0]]

    if not sel:
        return "筛选后没有会话可导出", None
    if not out_path:
        out_path = _default_export_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not self_wxid:
        self_wxid = exporter.infer_self_wxid(STATE["db"])

    def cb(done, total, msg):
        progress((done, total), desc=msg)

    try:
        res = exporter.export_chats(STATE["db"], sel, out_path, progress_cb=cb, self_wxid=self_wxid)
    except Exception as e:
        return "导出失败: %s" % e, None
    STATE["export_path"] = out_path
    return ("导出完成：%d 条消息（自己 %d 条），%d 个会话，耗时 %.1fs\n文件：%s" % (
        res["total"], res["self"], len(sel), res["elapsed"], out_path)), out_path


# ================= Tab3 清洗 =================
def do_clean(jsonl_path, min_len, max_len, filter_noise, max_samples, balance,
             long_first, long_threshold, progress=gr.Progress()):
    if not jsonl_path or not os.path.exists(jsonl_path):
        return "请先导出聊天记录（或填写已有 JSONL 路径）", None, None
    progress(0.1, desc="构建训练集…")
    st = cleaner.build_dataset(
        jsonl_path, DATA_DIR,
        min_len=int(min_len), max_len=int(max_len),
        filter_noise=filter_noise, max_samples=int(max_samples),
        balance=balance, long_first=long_first, long_threshold=int(long_threshold))
    STATE["train_json"], STATE["val_json"] = st["train_path"], st["val_path"]
    progress(1.0, desc="完成")
    text = (
        "【清洗与训练集构建完成】\n"
        "原始消息：%d 条（自己发言 %d 条）\n"
        "参与清洗的会话：%d 个\n"
        "长回复样本（≥%d字）：%d 条\n"
        "最终样本：%d 条（train %d / val %d）\n"
        "训练集回复长度：中位数 %d 字，P90 %d 字\n"
        "训练集：%s\n验证集：%s"
        % (st["raw_total"], st["self_raw"], st["chat_count"], int(long_threshold),
           st["long_reply"], st["samples"], st["train"], st["val"],
           st["reply_len_p50"], st["reply_len_p90"], st["train_path"], st["val_path"]))
    return text, st["train_path"], st["val_path"]


# ================= Tab4 训练 =================
def do_gpu_status():
    return gpu.gpu_summary()


def do_recommend():
    recs = gpu.recommend_model()
    return recs, gr.update(choices=[r[1] for r in recs], value=recs[0][1])


def do_download_model(hf_name, progress=gr.Progress()):
    if not hf_name:
        return "请输入模型名（如 Qwen/Qwen3.5-2B）", ""
    target = os.path.join(MODELS_DIR, hf_name.replace("/", "_"))
    progress(0.05, desc="下载 %s …" % hf_name)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
        tok.save_pretrained(target)
    except Exception as e:
        return "下载失败: %s（请检查网络 / 模型名）" % e, ""
    return "模型下载成功: %s" % target, target


def _list_merged():
    out = []
    if os.path.isdir(MODELS_DIR):
        for n in sorted(os.listdir(MODELS_DIR)):
            if "merged" in n.lower() and os.path.isdir(os.path.join(MODELS_DIR, n)):
                out.append(os.path.join(MODELS_DIR, n))
    return out


def do_train(model_dir, train_json, val_json, epochs, lr, max_len, grad_accum, adapter_dir):
    if not os.path.isdir(model_dir):
        return gr.update(maximum=1, value=0), "模型目录不存在: %s" % model_dir
    if not train_json or not os.path.exists(train_json):
        return gr.update(maximum=1, value=0), "训练集不存在，请先清洗构建"
    adapter_dir = adapter_dir or os.path.join(MODELS_DIR, "wx-lora")
    q = queue.Queue()
    args = dict(
        base_model_dir=model_dir, train_json=train_json, val_json=val_json,
        out_dir=os.path.join(OUTPUT_DIR, "sft"), adapter_dir=adapter_dir,
        epochs=float(epochs), lr=float(lr), max_len=int(max_len),
        grad_accum=int(grad_accum), progress_cb=q.put)

    def run():
        try:
            tr.train_lora(**args)
        except Exception as e:
            q.put({"error": str(e)})
    t = threading.Thread(target=run, daemon=True)
    t.start()
    yield gr.update(maximum=1, value=0), "训练启动…"
    while t.is_alive():
        try:
            d = q.get(timeout=0.5)
        except queue.Empty:
            yield gr.update(maximum=1, value=0), "训练中… | " + gpu.gpu_summary().replace("\n", " | ")
            continue
        if d.get("error"):
            yield gr.update(maximum=1, value=0), "训练失败: " + d["error"]
            return
        if d.get("done"):
            break
        total = d.get("total") or 1
        speed = d.get("speed") or 0
        loss = d.get("loss")
        loss_s = "%.4f" % loss if loss is not None else "-"
        yield (gr.update(maximum=total, value=d.get("step", 0)),
               "epoch=%s | loss=%s | 速度 %.2f step/s | %s" % (
                   d.get("epoch"), loss_s, speed, gpu.gpu_summary().replace("\n", " | ")))
    t.join()
    STATE["adapter_dir"] = adapter_dir
    yield gr.update(maximum=1, value=1), "训练完成，adapter 已保存: %s" % adapter_dir


def do_merge(model_dir, adapter_dir, merged_dir, progress=gr.Progress()):
    if not os.path.isdir(model_dir):
        return "基座模型目录不存在", ""
    if not os.path.isdir(adapter_dir):
        return "adapter 目录不存在: %s" % adapter_dir, ""
    merged_dir = merged_dir or os.path.join(MODELS_DIR, "wx-merged")
    progress(0.2, desc="合并 LoRA…")
    try:
        tr.merge_lora(model_dir, adapter_dir, merged_dir)
    except Exception as e:
        return "合并失败: %s" % e, ""
    STATE["merged_dir"] = merged_dir
    progress(1.0, desc="完成")
    return "合并完成: %s" % merged_dir, merged_dir


# ================= Tab5 对话 =================
def do_chat_load(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        return "模型目录不存在: %s" % model_dir, None
    try:
        msg = chatmod.load_model(model_dir)
    except Exception as e:
        return "加载失败: %s" % e, None
    return msg, []


def do_chat_unload():
    return chatmod.unload(), None


def do_chat(message, history, temperature, max_tokens, system_prompt):
    history = history or []
    try:
        reply = chatmod.generate(
            message, history=history, system_prompt=system_prompt or DEFAULT_SYSTEM,
            max_new_tokens=int(max_tokens), temperature=float(temperature))
    except Exception as e:
        reply = "（出错：%s）" % e
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


# ================= Tab6 分析 =================
def do_analyze_load(jsonl_path):
    if not jsonl_path or not os.path.exists(jsonl_path):
        return [], "文件不存在: %s" % jsonl_path, gr.update(choices=[]), ""
    msgs = analyzer.load_messages(jsonl_path)
    STATE["messages_cache"] = msgs
    st = analyzer.analyze_chat(msgs)
    names = sorted({m.get("name") for m in msgs})
    text = (
        "【总览】总消息 %d | 自己 %d | 对方 %d | 会话 %d 个\n"
        "时间跨度：%s 至 %s（活跃 %d 天 / 共 %d 天）\n"
        "自己发言长度：中位数 %d 字，P90 %d 字"
        % (st["total"], st["self"], st["other"], st["chats"],
           time.strftime("%Y-%m-%d", time.localtime(st["first_ts"])) if st["first_ts"] else "-",
           time.strftime("%Y-%m-%d", time.localtime(st["last_ts"])) if st["last_ts"] else "-",
           st["active_days"], st["span_days"],
           st["self_len_p50"], st["self_len_p90"]))
    return [[r[0], r[1]] for r in st["by_chat"]], text, gr.update(choices=names, value=None), ""


def _fig_chat_top(messages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    st = analyzer.analyze_chat(messages)
    top = st["by_chat"][:12]
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [t[0][:12] for t in reversed(top)]
    vals = [t[1] for t in reversed(top)]
    ax.barh(names, vals, color="#4f6df5")
    ax.set_title("会话消息条数 TOP12")
    ax.set_xlabel("条数")
    fig.tight_layout()
    return fig


def _fig_hour(messages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    st = analyzer.analyze_chat(messages)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(24), st["hour_cnt"], color="#4f6df5")
    ax.set_title("24 小时消息活跃分布")
    ax.set_xlabel("小时")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig


def _fig_month(messages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    st = analyzer.analyze_chat(messages)
    months = list(st["month_cnt"].keys())
    vals = list(st["month_cnt"].values())
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(vals)), vals, marker="o", color="#e85d75")
    ax.set_title("月度消息量趋势")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m[2:] for m in months], rotation=45, fontsize=8)
    fig.tight_layout()
    return fig


def do_analyze_charts():
    msgs = STATE.get("messages_cache") or []
    if not msgs:
        return None, None, None, "请先加载导出文件"
    return _fig_chat_top(msgs), _fig_hour(msgs), _fig_month(msgs), "图表已生成"


def do_keywords(topn):
    msgs = STATE.get("messages_cache") or []
    if not msgs:
        return "请先加载导出文件"
    try:
        kws = analyzer.top_keywords(msgs, topn=int(topn))
        return "高频词 TOP%d：%s" % (len(kws), "、".join("%s(%d)" % (k, n) for k, n in kws[:20]))
    except Exception as e:
        return "分词失败: %s" % e


def do_sentiment_score(chat_name):
    msgs = STATE.get("messages_cache") or []
    if not msgs:
        return "请先加载导出文件", None
    if not chat_name:
        return "请选择会话", None
    score, n = analyzer.sentiment_score(msgs, chat_name=chat_name)
    label = "非常负面" if score <= -50 else "偏负面" if score < -10 else \
            "偏正面" if score > 10 else "非常正面" if score >= 50 else "中性"
    return "会话「%s」情感倾向度：%+.1f / 100（%s，样本 %d 条自己发言）" % (chat_name, score, label, n), score


def do_sentiment_twin(chat_name, max_len=150):
    msgs = STATE.get("messages_cache") or []
    if not msgs or not chat_name:
        return "请先加载导出文件并选择会话"
    rows = analyzer.chat_series(msgs, chat_name)
    if not rows:
        return "会话无消息"
    text = "\n".join((("我" if (m.get("is_self") or (m.get("sender") or "").startswith("wxid_")) else "对方") + ": " + (m.get("content") or "")[:80])
                     for m in rows[-int(max_len):] if m.get("content"))
    try:
        return analyzer.sentiment_with_twin(text[:2000])
    except Exception as e:
        return "分析失败（需先在对话页加载微调模型）: %s" % e


def do_sentiment_pipe(model_name, text):
    if not model_name:
        return "请输入情感模型名，例如：uer/roberta-base-finetuned-jd-binary-chinese"
    try:
        pipe = analyzer.SentimentPipeline(model_name)
        label, score = pipe.predict(text)
        return "「%s」 → %s (置信度 %.3f)" % (text[:60], label, score)
    except Exception as e:
        return "调用失败: %s" % e


# ================= UI =================
with gr.Blocks(title="微信 AI 数字分身工作台") as demo:
    gr.Markdown("# 🤖 微信 AI 数字分身工作台\n"
                "一键完成：**自动发现微信数据 → 导出聊天记录 → 数据清洗 → LoRA 微调 → 对话 → 情感分析**。"
                "全程本地运行，聊天数据不出本机。")

    with gr.Tab("① 数据发现"):
        gr.Markdown("自动搜索常见官方存储位置（注册表 / 默认目录 / 各盘 xwechat_files 索引），找不到可手动选择。")
        with gr.Row():
            btn_discover = gr.Button("🔍 自动搜索微信数据", variant="primary")
            btn_manual = gr.Button("📁 手动选择文件夹")
        manual_dir = gr.Textbox(label="手动输入微信数据根目录路径（含 wxid_* 账号目录的那层）",
                                placeholder="例如 E:\\xwechat_files 或 D:\\WeChat Files")
        found_text = gr.Textbox(label="发现结果", lines=10)
        db_dir = gr.Dropdown(label="微信数据根目录（用于导出）")
        btn_scan = gr.Button("连接数据库（导出前必做）", variant="primary")
        scan_result = gr.Textbox(label="连接结果", lines=3)

        btn_discover.click(do_discover, outputs=[found_text, db_dir, scan_result])
        btn_manual.click(do_manual_folder, inputs=[manual_dir], outputs=[found_text, db_dir, scan_result])
        btn_scan.click(do_scan_db, inputs=[db_dir], outputs=[scan_result])

    with gr.Tab("② 导出聊天记录"):
        gr.Markdown("选择要导出的会话范围，导出为 JSONL（含条数、发送者、时间、文本）。")
        btn_list = gr.Button("加载会话列表")
        chat_table = gr.Dataframe(headers=["会话名", "username", "消息数"], datatype=["str", "str", "number"],
                                  interactive=False, label="会话列表", wrap=True)
        chat_count = gr.Textbox(label="会话数", interactive=False)
        with gr.Row():
            scope_mode = gr.Radio(["全部会话", "关键词筛选", "单个会话"], value="全部会话", label="导出范围")
            keyword = gr.Textbox(label="关键词（逗号分隔；加 - 前缀表示排除，如：-客服）")
            single_name = gr.Dropdown(label="单个会话（先加载列表）")
        export_path = gr.Textbox(label="导出文件路径", value=_default_export_path())
        self_wxid = gr.Textbox(label="自己的 wxid（留空自动从账号目录推断）",
                               placeholder="例如 wxid_e6o9se8d4ob922")
        btn_export = gr.Button("🚀 开始导出", variant="primary")
        export_result = gr.Textbox(label="导出结果", lines=4)

        btn_list.click(do_load_chats, outputs=[chat_table, chat_count, single_name])
        btn_export.click(do_export, inputs=[scope_mode, keyword, single_name, export_path, self_wxid],
                         outputs=[export_result, export_path])

    with gr.Tab("③ 数据清洗与训练集构建"):
        gr.Markdown("过滤噪声（表情/图片/系统消息）、控制句子长度与样本量，自动切分训练/验证集。")
        clean_src = gr.Textbox(label="源 JSONL（默认用上一步导出文件）", value=_default_export_path())
        with gr.Row():
            min_len = gr.Slider(1, 20, value=2, step=1, label="最短句子长度（字）")
            max_len = gr.Slider(100, 2000, value=800, step=50, label="最长句子长度（字）")
        with gr.Row():
            filter_noise = gr.Checkbox(value=True, label="过滤表情/图片/系统消息等噪声")
            balance = gr.Checkbox(value=True, label="会话平衡（防止大群独霸）")
            long_first = gr.Checkbox(value=True, label="长回复优先（解决句子太短）")
        with gr.Row():
            long_threshold = gr.Slider(20, 200, value=40, step=5, label="长回复阈值（字）")
            max_samples = gr.Slider(1000, 60000, value=15000, step=1000, label="训练样本上限")
        btn_clean = gr.Button("🧹 构建训练集", variant="primary")
        clean_result = gr.Textbox(label="清洗结果", lines=12)
        clean_train = gr.Textbox(label="train.jsonl", interactive=False)
        clean_val = gr.Textbox(label="val.jsonl", interactive=False)

        btn_clean.click(do_clean, inputs=[clean_src, min_len, max_len, filter_noise, max_samples,
                                          balance, long_first, long_threshold],
                        outputs=[clean_result, clean_train, clean_val])

    with gr.Tab("④ 微调训练"):
        gr.Markdown("自动检测显卡并推荐模型；可自行选择模型或本地目录。训练过程显示实时 loss、速度与显存占用。")
        with gr.Row():
            btn_gpu = gr.Button("刷新显卡信息")
            btn_rec = gr.Button("自动推荐模型")
        gpu_text = gr.Textbox(label="显卡信息", lines=3)
        with gr.Row():
            model_choice = gr.Dropdown(label="推荐模型（训练时自动下载到本地 models/ 目录）")
            model_dir = gr.Textbox(label="本地模型目录（可直接填绝对路径，如已下载过）",
                                   value=os.path.join(MODELS_DIR, "Qwen3.5-2B"))
        with gr.Row():
            btn_dl = gr.Button("⬇ 下载模型到本地")
            dl_result = gr.Textbox(label="下载结果", interactive=False)
        train_json = gr.Textbox(label="train.jsonl", value=_default_train())
        val_json = gr.Textbox(label="val.jsonl", value=_default_val())
        with gr.Row():
            epochs = gr.Number(value=1, label="训练轮数（epochs）")
            lr = gr.Number(value=2e-4, label="学习率")
            max_len = gr.Number(value=384, label="最大长度（token）")
            grad_accum = gr.Number(value=8, label="梯度累积步数")
        adapter_dir = gr.Textbox(label="adapter 输出目录", value=os.path.join(MODELS_DIR, "wx-lora"))
        with gr.Row():
            btn_train = gr.Button("🔥 开始训练", variant="primary")
            btn_merge = gr.Button("🧩 合并模型")
        train_progress = gr.Progress()
        train_status = gr.Textbox(label="训练状态（loss / 速度 / 显存）", lines=4)
        train_bar = gr.Slider(minimum=0, maximum=1, value=0, label="训练进度", interactive=False)
        merged_dir = gr.Textbox(label="合并模型输出目录", value=os.path.join(MODELS_DIR, "wx-merged"))
        merge_result = gr.Textbox(label="合并结果", interactive=False)

        btn_gpu.click(do_gpu_status, outputs=[gpu_text])
        btn_rec.click(do_recommend, outputs=[model_choice, model_dir])
        btn_dl.click(do_download_model, inputs=[model_choice], outputs=[dl_result, model_dir])
        btn_train.click(do_train,
                        inputs=[model_dir, train_json, val_json, epochs, lr, max_len, grad_accum, adapter_dir],
                        outputs=[train_bar, train_status])
        btn_merge.click(do_merge, inputs=[model_dir, adapter_dir, merged_dir], outputs=[merge_result, merged_dir])

    with gr.Tab("⑤ 对话（数字分身）"):
        gr.Markdown("加载微调后的模型，模拟你的说话风格与好友聊天。")
        chat_model_dir = gr.Textbox(label="合并模型目录", value=os.path.join(MODELS_DIR, "wx-merged"))
        with gr.Row():
            btn_load = gr.Button("加载模型", variant="primary")
            btn_unload = gr.Button("释放模型（省显存）")
        load_result = gr.Textbox(label="加载状态", interactive=False)
        with gr.Row():
            temperature = gr.Slider(0.3, 1.5, value=0.85, step=0.05, label="temperature")
            max_tokens = gr.Slider(32, 512, value=160, step=16, label="最大回复长度")
        system_prompt = gr.Textbox(label="人设（system prompt）", value=DEFAULT_SYSTEM, lines=3)
        chatbot = gr.Chatbot(label="对话", height=480)
        msg = gr.Textbox(label="输入消息", placeholder="跟你的数字分身说点什么…")
        btn_send = gr.Button("发送", variant="primary")

        btn_load.click(do_chat_load, inputs=[chat_model_dir], outputs=[load_result, chatbot])
        btn_unload.click(do_chat_unload, outputs=[load_result, chatbot])
        btn_send.click(do_chat, inputs=[msg, chatbot, temperature, max_tokens, system_prompt],
                       outputs=[chatbot, msg])

    with gr.Tab("⑥ 统计分析 & 情感"):
        gr.Markdown("对导出的聊天记录做统计分析；用微调模型或专用情感小模型分析对话情感。")
        ana_src = gr.Textbox(label="导出 JSONL", value=_default_export_path())
        with gr.Row():
            btn_ana = gr.Button("加载并统计")
            btn_charts = gr.Button("生成图表")
        ana_overview = gr.Textbox(label="总览", lines=6)
        ana_table = gr.Dataframe(headers=["会话", "消息数"], label="会话排行", wrap=True)
        with gr.Row():
            fig_top = gr.Plot(label="会话条数 TOP")
            fig_hour = gr.Plot(label="24小时分布")
            fig_month = gr.Plot(label="月度趋势")
        with gr.Row():
            topn = gr.Slider(10, 50, value=30, step=5, label="高频词数量")
            btn_kw = gr.Button("高频词分析")
        kw_result = gr.Textbox(label="高频词", lines=4)

        gr.Markdown("### 情感分析")
        ana_chat = gr.Dropdown(label="选择会话")
        with gr.Row():
            btn_score = gr.Button("情感倾向度（规则评分）")
            score_result = gr.Textbox(label="情感倾向", lines=3)
        with gr.Row():
            btn_twin = gr.Button("🤖 用数字分身读心（需先加载模型）")
            twin_result = gr.Textbox(label="数字分身分析", lines=6)
        with gr.Row():
            pipe_model = gr.Textbox(label="情感小模型名（可下载）",
                                    value="uer/roberta-base-finetuned-jd-binary-chinese")
            pipe_text = gr.Textbox(label="要分析的一句话")
        with gr.Row():
            btn_pipe = gr.Button("用情感小模型分析")
            pipe_result = gr.Textbox(label="情感模型结果", lines=3)

        btn_ana.click(do_analyze_load, inputs=[ana_src], outputs=[ana_table, ana_overview, ana_chat])
        btn_charts.click(do_analyze_charts, outputs=[fig_top, fig_hour, fig_month, ana_overview])
        btn_kw.click(do_keywords, inputs=[topn], outputs=[kw_result])
        btn_score.click(do_sentiment_score, inputs=[ana_chat], outputs=[score_result])
        btn_twin.click(do_sentiment_twin, inputs=[ana_chat], outputs=[twin_result])
        btn_pipe.click(do_sentiment_pipe, inputs=[pipe_model, pipe_text], outputs=[pipe_result])

    gr.Markdown("---\n数据全程本地处理，不上传任何服务器。训练建议：6GB 显存可用 Qwen3.5-2B（bf16 LoRA）。")


def main():
    print("=" * 50)
    print("微信 AI 数字分身工作台")
    print("浏览器打开: http://127.0.0.1:7860")
    print("数据目录:", DATA_DIR)
    print("模型目录:", MODELS_DIR)
    print("=" * 50)
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True,
        theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
