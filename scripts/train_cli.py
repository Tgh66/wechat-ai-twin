# -*- coding: utf-8 -*-
"""参数化训练 CLI：任意会话 → 训练"我"或"对方"的数字分身
由 Go 工作台调用。日志按结构化行输出，供界面解析进度。
用法:
  python train_cli.py --data <jsonl> --chat <会话名> --persona me|other
                      --model <基座目录> --epochs 2 --samples 15000 --lr 2e-4
                      --out <输出目录> --port 5000 --log <日志>
"""
import argparse, json, os, re, sys, time, subprocess

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def log(msg):
    line = f"[TRAIN] {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)

def stage(name, **kv):
    parts = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[STAGE] {name} {parts}".strip(), flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--chat", required=True)
    p.add_argument("--persona", default="me", choices=["me", "other"])
    p.add_argument("--model", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--samples", type=int, default=15000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--maxlen", type=int, default=512)
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--log", default="")
    return p.parse_args()

# ---------- 清洗：任意会话 + persona ----------
def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.strip()
    if not (2 <= len(t) <= 150):
        return ""
    if re.fullmatch(r"(\[[^\]]{1,10}\]|[\s！？?。，,.~～…!]+|https?://\S+)+", t):
        return ""
    t = re.sub(r"https?://\S+", "", t).strip()
    if len(t) < 2:
        return ""
    if len(set(t)) <= 2:
        return ""
    return t

def build_dataset(args):
    stage("clean", chat=args.chat, persona=args.persona)
    log(f"读取 {args.data}")
    msgs = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("name") or r.get("user") or "").strip() == args.chat:
                msgs.append(r)
    log(f"会话 {args.chat} 共 {len(msgs)} 条")

    cleaned = []
    for m in msgs:
        if m.get("type") != "文本":
            continue
        c = clean_text(m.get("content", ""))
        if not c:
            continue
        cleaned.append((bool(m.get("is_self")), c))

    # 合并连续同方消息
    merged = []
    for is_self, c in cleaned:
        if merged and merged[-1][0] == is_self:
            if len(merged[-1][1]) + len(c) < 90:
                merged[-1][1] += "。" + c
                continue
        merged.append([is_self, c])

    # persona 决定 assistant 是谁
    # me:   我(is_self=1) 是 assistant，对方是 user
    # other:对方(is_self=0) 是 assistant，我是 user
    asst_self = 1 if args.persona == "me" else 0
    ctx = []
    pairs = []
    for i in range(len(merged) - 1):
        is_self, c = merged[i]
        if is_self == asst_self:
            continue  # 等待 user 发言
        nxt = merged[i + 1]
        if nxt[0] != asst_self:
            continue  # 需要 assistant 回复
        ctx.append({"role": "user", "content": c})
        ctx.append({"role": "assistant", "content": nxt[1]})
        if len(ctx) >= 8:
            pairs.append({"messages": ctx[-8:]})
            ctx = []
    if len(ctx) >= 2:
        pairs.append({"messages": ctx})

    # 去重 + 截断
    seen, out = set(), []
    for p in pairs:
        key = json.dumps(p["messages"], ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        if sum(len(m["content"]) for m in p["messages"]) <= 700:
            out.append(p)
    log(f"清洗后对话对: {len(out)}")

    import random
    random.seed(42)
    random.shuffle(out)
    n = min(args.samples, len(out))
    train_path = os.path.join(os.path.dirname(args.out), "train_cli.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for p in out[:n]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"训练集 {n} 条 → {train_path}")
    return train_path, n

# ---------- 训练 ----------
def train(args, train_path, n):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
        BitsAndBytesConfig, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    stage("train", samples=n, epochs=args.epochs, lr=args.lr, model=args.model, device=device)
    log(f"加载基座 {args.model}（设备 {device}）")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cuda:
        # NVIDIA GPU：4bit QLoRA（现有方案）
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="cuda")
        model = prepare_model_for_kbit_training(model)
    else:
        # 无 GPU：CPU fp32 训练（明确降级，不静默失败）
        stage("train", note="未检测到 NVIDIA GPU，回退 CPU fp32 训练（速度较慢但可完成；建议小样本/短对话）")
        log("无 NVIDIA GPU → CPU fp32 训练（去 4bit 量化，bf16 改 fp32，batch=1）")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True,
            torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def fmt(msgs):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    texts = [fmt(r["messages"]) for r in rows]

    # 手动 loss masking：assistant 段才计算 loss
    def tokenize_with_mask(text):
        enc = tok(text, truncation=True, max_length=args.maxlen)
        ids = enc["input_ids"]
        labels = [-100] * len(ids)
        # 找到最后一个 <|im_start|>assistant 之后的位置
        marker = tok("<|im_start|>assistant", add_special_tokens=False)["input_ids"]
        for i in range(len(ids) - len(marker), -1, -1):
            if ids[i:i + len(marker)] == marker:
                for j in range(i + len(marker), len(ids)):
                    labels[j] = ids[j]
                break
        return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}

    ds = [tokenize_with_mask(t) for t in texts]

    total_steps = int(len(ds) / 8 * args.epochs) + 1
    log(f"总步数约 {total_steps}")

    args_t = TrainingArguments(
        output_dir=os.path.join(args.out, "ckpt"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=100000,
        bf16=cuda,
        fp16=False,
        optim="paged_adamw_8bit" if cuda else "adamw_torch",
        max_grad_norm=1.0,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    class LoggingCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            loss = logs.get("loss", 0)
            print(f"[TRAIN] step={step} total={total_steps} loss={loss:.4f} lr={logs.get('learning_rate', 0):.2e}", flush=True)

    from transformers import TrainerCallback
    class _C(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            loss = logs.get("loss", 0) if logs else 0
            print(f"[TRAIN] step={step} total={total_steps} loss={loss:.4f}", flush=True)

    trainer = Trainer(
        model=model, args=args_t, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        callbacks=[_C()])
    log("开始训练")
    trainer.train()
    adapter = os.path.join(args.out, "adapter")
    trainer.save_model(adapter)
    tok.save_pretrained(adapter)
    log("LoRA 保存完成")

# ---------- 合并 ----------
def merge(args):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    stage("merge", out=args.out)
    log("合并 LoRA → 全量模型")
    # 先释放训练阶段残留（GPU 4bit 模型、优化器状态等），避免合并时 CPU 内存峰值
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # low_cpu_mem_usage: 流式加载权重，避免整模型一次性载入内存导致峰值爆掉
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="cpu", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, os.path.join(args.out, "adapter"))
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    # 分片保存（每片 ≤2GB），避免写盘时单文件过大引发内存暴涨
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(args.out)
    # 合并完立即释放，降低峰值占用
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"合并完成 → {args.out}")

# ---------- 启动推理 ----------
def serve(args, py, cwd):
    stage("serve", port=args.port)
    log(f"启动推理服务 :{args.port}（模型 {args.out}）")
    serve_script = os.path.join(cwd, "scripts", "serve_twin.py")
    subprocess.Popen([
        py, serve_script, "--model", args.out, "--port", str(args.port), "--bits", "8",
    ], cwd=cwd, creationflags=subprocess.CREATE_NO_WINDOW)

def find_project_python():
    """从脚本位置推导项目根（scripts/ 上一级），探测可用的训练/推理 python。
    兼容三种部署：源码目录 / release zip 解压目录 / 开发机布局。"""
    here = os.path.dirname(os.path.abspath(__file__))        # .../scripts
    root = os.path.dirname(here)                              # 项目根
    cands = [
        os.path.join(root, "..", "ai-twin", "venv", "Scripts", "python.exe"),
        os.path.join(root, "..", "ai-twin", "venv", "python.exe"),
        os.path.join(root, "venv", "Scripts", "python.exe"),
        os.path.join(root, "venv", "python.exe"),
        os.path.join(root, "runtime", "python.exe"),          # 内置运行时（解密用，无 torch；仅作兜底探测）
    ]
    for c in cands:
        if os.path.exists(c):
            return c, root
    return sys.executable, root

if __name__ == "__main__":
    args = parse_args()
    if args.log:
        # 重定向 stdout 到日志文件（同时保留控制台）
        _old = sys.stdout
        sys.stdout = open(args.log, "a", encoding="utf-8")
        print(f"=== 训练启动 {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    try:
        t0 = time.time()
        train_path, n = build_dataset(args)
        train(args, train_path, n)
        merge(args)
        py, cwd = find_project_python()
        serve(args, py, cwd)
        stage("done", cost_h=round((time.time() - t0) / 3600, 2))
        log(f"全流程完成，耗时 {(time.time()-t0)/3600:.2f}h")
    except Exception as e:
        import traceback
        traceback.print_exc()
        stage("error", msg=str(e))
        sys.exit(1)
