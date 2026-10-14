# -*- coding: utf-8 -*-
"""对话：加载微调模型并推理"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_MODEL = None
_TOK = None
_MODEL_LOCK = None


def get_lock():
    global _MODEL_LOCK
    if _MODEL_LOCK is None:
        import threading
        _MODEL_LOCK = threading.Lock()
    return _MODEL_LOCK


def unload():
    """释放模型（节省显存）"""
    global _MODEL, _TOK
    with get_lock():
        if _MODEL is not None:
            import torch
            _MODEL.cpu()
            del _MODEL
            _TOK = None
            _MODEL = None
            torch.cuda.empty_cache()
    return "模型已释放"


def load_model(model_dir, device="cuda"):
    """加载模型（幂等：已加载同目录则复用）"""
    global _MODEL, _TOK
    with get_lock():
        if _MODEL is not None and getattr(_MODEL, "_loaded_from", None) == model_dir:
            return "已加载: %s" % model_dir
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, trust_remote_code=True, device_map=device)
        model.eval()
        model._loaded_from = model_dir
        _MODEL, _TOK = model, tokenizer
        return "模型加载成功: %s" % model_dir


def generate(user_text, history=None, system_prompt=None,
             max_new_tokens=160, temperature=0.85, top_p=0.9,
             repetition_penalty=1.1, enable_thinking=False):
    """对话生成。history: [{"user":..,"assistant":..}]"""
    global _MODEL, _TOK
    if _MODEL is None:
        return "（请先在训练页/对话页加载模型）"
    import torch
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if history:
        for h in history[-6:]:
            msgs.append({"role": "user", "content": h["user"]})
            msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": user_text})
    try:
        text = _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=enable_thinking)
    except TypeError:
        text = _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = _TOK(text, return_tensors="pt").to(next(_MODEL.parameters()).device)
    # 停止符：优先使用 tokenizer 的 eos（模板的 im_end），pad 取 eos 或 config
    eos = _TOK.eos_token_id
    pad = _TOK.pad_token_id if _TOK.pad_token_id is not None else eos
    with torch.no_grad():
        out = _MODEL.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
            eos_token_id=eos, pad_token_id=pad)
    gen = _TOK.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.replace("<think>", "").replace("</think>", "").strip()


def chat_analysis(chat_text, question="分析这段对话中我（说话人）的情绪状态和感受", system_prompt=None):
    """用微调模型分析一段对话"""
    return generate(chat_text, history=None,
                    system_prompt=system_prompt or question,
                    max_new_tokens=300, temperature=0.6)
