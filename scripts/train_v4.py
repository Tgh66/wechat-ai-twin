# -*- coding: utf-8 -*-
"""v4 一站式训练：严格清洗 → 下载基座 → QLoRA 微调 → 合并 → 切换推理模型
后台运行，日志写入 data/train_v4.log
"""
import json, os, re, sys, time, subprocess

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\Administrator\DoubaoWork\chats\2026-10-13\new-chat-1"
DATA = os.path.join(ROOT, "ai-twin", "data", "chat_messages_e.jsonl")
BASE = os.path.join(ROOT, "ai-twin", "models", "Qwen3-4B")
TRAIN = os.path.join(ROOT, "ai-twin", "data", "train_v4.jsonl")
OUT = os.path.join(ROOT, "ai-twin", "output", "sft_v4")
ADAPTER = os.path.join(ROOT, "ai-twin", "models", "qwen3-4b-wx-lora-v4")
MERGED = os.path.join(ROOT, "ai-twin", "models", "Qwen3-4B-wx-merged-v4")

def log(m):
    print(f"[v4 {time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---------- 1. 严格清洗 + 对话对构建 ----------
def build_dataset():
    log("读取全量聊天记录…")
    msgs = []
    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            msgs.append(r)
    log(f"共 {len(msgs)} 条")

    # 按会话分组
    by_chat = {}
    for m in msgs:
        name = (m.get("name") or m.get("user") or "?").strip()
        by_chat.setdefault(name, []).append(m)
    log(f"会话数: {len(by_chat)}")

    # 群聊排除词（用户消息占比低/明显群名）
    GRP = ("群", "圈", "捡漏", "攻略", "羊毛", "后宫", "论坛", "代做", "交流", "福利", "任务", "班级通知")
    def is_private(name, ms):
        if any(g in name for g in GRP):
            return False
        self_n = sum(1 for m in ms if m.get("is_self"))
        if len(ms) < 50:
            return False
        ratio = self_n / len(ms)
        return 0.08 <= ratio <= 0.95

    def clean_text(t):
        if not isinstance(t, str):
            return ""
        t = t.strip()
        if not (2 <= len(t) <= 150):
            return ""
        # 纯表情/纯符号/纯URL
        if re.fullmatch(r"(\[[^\]]{1,10}\]|[\s！？?。，,.~～…!]+|https?://\S+)+", t):
            return ""
        # 去 URL
        t = re.sub(r"https?://\S+", "", t).strip()
        if len(t) < 2:
            return ""
        # 连续重复整句（如 "哈哈哈哈哈哈"）
        if len(set(t)) <= 2:
            return ""
        # 系统消息/引用
        if t.startswith("[") and "]" in t[:12] and len(t) <= 14:
            return ""
        return t

    pairs = []
    for name, ms in by_chat.items():
        if not is_private(name, ms):
            continue
        ms.sort(key=lambda x: x.get("ts", 0))
        cleaned = []
        for m in ms:
            if m.get("type") != "文本":
                continue
            c = clean_text(m.get("content", ""))
            if not c:
                continue
            cleaned.append((bool(m.get("is_self")), c))
        # 合并连续同方消息；构建 对方→我 回复对，保留最近 4 轮上下文
        merged = []
        for is_self, c in cleaned:
            if merged and merged[-1][0] == is_self:
                if len(merged[-1][1]) + len(c) < 90:
                    merged[-1][1] += "。" + c
                    continue
            merged.append([is_self, c])
        ctx = []
        for i in range(len(merged) - 1):
            is_self, c = merged[i]
            if is_self:
                continue  # 找对方发言
            nxt = merged[i + 1]
            if not nxt[0]:
                continue  # 需要我回复
            ctx.append({"role": "user", "content": c})
            ctx.append({"role": "assistant", "content": nxt[1]})
            if len(ctx) >= 8:
                pairs.append({"messages": ctx[-8:]})
                ctx = []
        if len(ctx) >= 2:
            pairs.append({"messages": ctx})

    log(f"原始回复对: {len(pairs)}")
    # 去重 + 过滤超长
    seen = set()
    out = []
    for p in pairs:
        key = json.dumps(p["messages"], ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        total_len = sum(len(m["content"]) for m in p["messages"])
        if total_len <= 700:
            out.append(p)
    log(f"清洗后训练样本: {len(out)}")

    import random
    random.seed(42)
    random.shuffle(out)
    n_train = min(30000, int(len(out) * 0.95))
    with open(TRAIN, "w", encoding="utf-8") as f:
        for p in out[:n_train]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"训练集: {n_train} → {TRAIN}")
    return n_train

# ---------- 2. 下载基座 ----------
def ensure_base():
    if os.path.isdir(BASE) and any(x.endswith(".safetensors") for x in os.listdir(BASE)):
        log("基座已存在")
        return
    log("通过 ModelScope 下载 Qwen/Qwen3-4B …")
    from modelscope import snapshot_download
    os.makedirs(BASE, exist_ok=True)
    snapshot_download("Qwen/Qwen3-4B", local_dir=BASE)
    log("基座下载完成")

# ---------- 3. 训练 ----------
def train():
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        BitsAndBytesConfig, DataCollatorForCompletionOnlyLM,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True,
                                                 quantization_config=bnb,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = []
    with open(TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log(f"加载训练样本 {len(rows)}")

    def fmt(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    texts = [fmt(r["messages"]) for r in rows]
    log("模板转换完成，示例：" + texts[0][:200].replace("\n", "\\n"))

    enc = tok(texts, truncation=True, max_length=512, padding=False)
    ds = [{"input_ids": ids} for ids in enc["input_ids"]]

    args = TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=20,
        save_steps=500,
        save_total_limit=1,
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant",
        tokenizer=tok,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    log("开始训练…")
    trainer.train()
    trainer.save_model(ADAPTER)
    log("LoRA 保存完成")

# ---------- 4. 合并 ----------
def merge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    log("合并 LoRA → 全量模型")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="cpu")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload()
    os.makedirs(MERGED, exist_ok=True)
    model.save_pretrained(MERGED, safe_serialization=True)
    tok.save_pretrained(MERGED)
    log(f"合并完成: {MERGED}")

# ---------- 5. 切换 serve ----------
def swap_serve():
    # 更新 serve 启动脚本默认模型路径（下一轮 serve 用 v4）
    serve = os.path.join(ROOT, "wechat-ai-twin", "scripts", "serve_twin.py")
    with open(serve, "r", encoding="utf-8") as f:
        s = f.read()
    if "Qwen3-4B-wx-merged-v4" not in s:
        s = s.replace('parser.add_argument("--model", default=r"models/Qwen3-4B-wx-merged")',
                      'parser.add_argument("--model", default=r"..\\ai-twin\\models\\Qwen3-4B-wx-merged-v4")')
        with open(serve, "w", encoding="utf-8") as f:
            f.write(s)
        log("serve 默认模型已切到 v4")
    # 启动 serve
    subprocess.Popen([
        os.path.join(ROOT, "ai-twin", "venv", "Scripts", "python.exe"),
        os.path.join(ROOT, "wechat-ai-twin", "scripts", "serve_twin.py"),
        "--model", MERGED, "--port", "5000", "--bits", "8",
    ], creationflags=subprocess.CREATE_NO_WINDOW)
    log("serve 已启动 (v4)")

if __name__ == "__main__":
    t0 = time.time()
    build_dataset()
    ensure_base()
    train()
    merge()
    swap_serve()
    log(f"v4 全流程完成，总耗时 {(time.time()-t0)/3600:.1f}h")
