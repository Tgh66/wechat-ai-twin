# -*- coding: utf-8 -*-
"""LoRA 微调 + 合并（带进度回调）"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ProgressCallback:
    """Transformers TrainerCallback 包装：把日志推给 ui 回调"""

    def __init__(self, emit, total_steps):
        self.emit = emit
        self.total_steps = total_steps
        self.last_time = time.time()
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        dt = max(now - self.last_time, 1e-6)
        ds = state.global_step - self.last_step
        self.last_time = now
        self.last_step = state.global_step
        speed = ds / dt if dt > 0 else 0
        try:
            self.emit({
                "step": state.global_step,
                "total": self.total_steps,
                "epoch": round(state.epoch, 2) if state.epoch else None,
                "loss": logs.get("loss") if logs else None,
                "lr": logs.get("learning_rate") if logs else None,
                "speed": round(speed, 2),
            })
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self.emit({"step": self.total_steps, "total": self.total_steps, "done": True})
        except Exception:
            pass


def train_lora(base_model_dir, train_json, val_json, out_dir, adapter_dir,
               epochs=1, lr=2e-4, max_len=384, grad_accum=8, batch=1,
               lora_r=16, lora_alpha=32, eval_steps=300,
               progress_cb=None, device="cuda"):
    """执行 LoRA 训练；progress_cb(dict) 接收进度；返回 adapter 路径"""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, dtype=torch.bfloat16, trust_remote_code=True, device_map=device)
    model.eval()

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules="all-linear",
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    model.gradient_checkpointing_enable()

    def build_examples(rows):
        out = []
        for r in rows:
            full = tok.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)
            prompt = tok.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            out.append((full, prompt))
        return out

    def tokenize_pair(full, prompt):
        fid = tok(full, truncation=True, max_length=max_len)["input_ids"]
        pid = tok(prompt, truncation=True, max_length=max_len)["input_ids"]
        if len(pid) >= len(fid):
            return None
        labels = [-100] * len(fid)
        labels[len(pid):] = fid[len(pid):]
        return {"input_ids": fid, "attention_mask": [1] * len(fid), "labels": labels}

    def gen_split(examples):
        for i in range(len(examples)):
            item = tokenize_pair(*examples[i])
            if item is not None:
                yield item

    train_rows = load_jsonl(train_json)
    val_rows = load_jsonl(val_json)
    train_ex = build_examples(train_rows)
    val_ex = build_examples(val_rows)

    import datasets as hf_ds
    train_ds = hf_ds.Dataset.from_generator(lambda: gen_split(train_ex))
    val_ds = hf_ds.Dataset.from_generator(lambda: gen_split(val_ex))

    effective_batch = batch * grad_accum
    max_steps = max(1, int(len(train_ds) / effective_batch) * epochs)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=min(50, max(1, max_steps // 20)),
        weight_decay=0.01,
        bf16=True,
        optim="adamw_8bit",
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps" if val_ds and len(val_ds) > 0 else "no",
        eval_steps=eval_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    callbacks = []
    if progress_cb:
        callbacks.append(ProgressCallback(progress_cb, max_steps))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if val_ds and len(val_ds) > 0 else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        callbacks=callbacks,
    )

    trainer.train()
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    return adapter_dir


def merge_lora(base_model_dir, adapter_dir, merged_dir, progress_cb=None):
    """LoRA 合并为独立模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_dir, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    return merged_dir
