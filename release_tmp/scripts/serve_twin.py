# -*- coding: utf-8 -*-
"""数字分身推理服务：加载微调模型，提供 HTTP 对话接口（供 Go 工作台连接）
用法: python scripts/serve_twin.py --model models/Qwen3-4B-wx-merged --port 5000
CPU/GPU 自适应：无 NVIDIA GPU 时自动走 CPU fp32（去 8/4bit 量化）。
"""
import sys, os, time, argparse, json
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 依赖检查：缺失时打印明确安装命令后退出，禁止裸抛 ModuleNotFoundError ----
REQ_TXT = "请先在项目根目录执行以下命令安装推理依赖（无 GPU 机器选 CPU 版）：\n" \
          "  无 GPU:  <venv>\\Scripts\\python.exe -m pip install -r scripts\\requirements-train-cpu.txt\n" \
          "  有 GPU:  <venv>\\Scripts\\python.exe -m pip install -r scripts\\requirements-train-gpu.txt"
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"[serve] 缺少推理依赖: {e}", flush=True)
    print(f"[serve] {REQ_TXT}", flush=True)
    sys.exit(2)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=r"models/Qwen3-4B-wx-merged")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
args = parser.parse_args()

# ---- 模型路径校验：不存在/缺 config.json 时给出明确提示 ----
if not os.path.isdir(args.model) or not os.path.isfile(os.path.join(args.model, "config.json")):
    print(f"[serve] 模型目录无效或不存在: {args.model}", flush=True)
    print(f"[serve] 请先在训练页「数字分身」中选择/下载模型（或完成一次训练生成合并模型），再启动推理服务。", flush=True)
    sys.exit(3)

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print(f"[serve] 加载模型: {args.model} 设备={device}（{'8/4bit 量化' if cuda else 'CPU fp32'}）", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if cuda:
    try:
        from transformers import BitsAndBytesConfig
        if args.bits == 8:
            quant = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        elif args.bits == 4:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="cuda",
            quantization_config=quant, torch_dtype=torch.bfloat16)
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, device_map="cuda", torch_dtype=torch.bfloat16)
else:
    # CPU：去量化、fp32、低内存流式加载
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map="cpu",
        torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
# 禁用 thinking（Qwen3 默认开启思考块，数字分身不需要）
try:
    model.generation_config.enable_thinking = False
except Exception:
    pass

# pad/eos 从 tokenizer 读取（兼容 Qwen2.5/Qwen3 等不同模型）
if tok.pad_token_id is None or tok.pad_token_id < 0:
    tok.pad_token_id = tok.eos_token_id
eos_ids = [tok.eos_token_id] if tok.eos_token_id is not None else []
print(f"[serve] 加载完成 {time.time()-t0:.1f}s, eos={eos_ids}, pad={tok.pad_token_id}", flush=True)

SYSTEM = "你是微信用户\"我\"的数字分身。你正在微信上和好友聊天，请用我真实、自然的说话方式、语气和习惯回复对方。要求：1) 像真人微信聊天一样，把回复拆成 1-8 条短消息（每条 5-40 字），每条消息单独占一行，模拟连发多条微信；2) 每条都要有实质内容，针对对方的话给出具体回应；3) 简洁直接，可以偶尔用“哈哈”“嗯嗯”等口语词，但绝不能整条只发表情或只回“？？”。"

@torch.inference_mode()
def generate_once(prompt_tokens, max_tokens=240, temp=0.8):
    inputs = tok(prompt_tokens, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
        repetition_penalty=1.12,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )
    out = model.generate(**inputs, **gen_kwargs)
    new = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    for e in eos_ids:
        if e in new.tolist():
            text = tok.decode(new[:new.tolist().index(e)], skip_special_tokens=True)
            break
    return clean_reply(text)

def clean_reply(t: str):
    t = t.replace("<think>", "").replace("</think>", "")
    t = t.replace("user", "").replace("assistant", "").replace("system", "")
    lines = []
    for ln in t.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if lines and ln == lines[-1]:
            continue
        lines.append(ln)
    t = "\n".join(lines)
    import re
    t = re.sub(r"(\[[^\]]+\])(\1{2,})", r"\1\1", t)
    return t.strip()[:400]

def quality_score(t: str) -> float:
    """回复质量评分：越高越好"""
    import re
    if not t:
        return -1
    n = len(t)
    if n < 4:
        return -2
    # 问号/感叹号/表情占比过高 → 差
    punct = len(re.findall(r"[？?！!。，,～~…]", t))
    emoji = len(re.findall(r"\[[^\]]{1,8}\]", t))
    if (punct + emoji) / n > 0.45:
        return -3
    # 重复模式
    if len(set(t)) <= 3:
        return -4
    if "？？" in t and t.count("？") > 4:
        return -5
    # 有实质内容词
    zh = len(re.findall(r"[\u4e00-\u9fff]", t))
    if zh < 3:
        return -6  # 纯表情/符号 → 差
    return zh

def chat(msg: str, history: list):
    # 组装 Qwen chat 模板
    convo = [{"role": "system", "content": SYSTEM}]
    for h in history[-6:]:
        convo.append(h)
    convo.append({"role": "user", "content": msg})
    prompt = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    # 重采样：最多 3 次，取质量最高
    best, best_q = "", -99
    for temp in [0.8, 0.9, 0.7]:
        try:
            cand = generate_once(prompt, max_tokens=320, temp=temp)
        except Exception:
            continue
        q = quality_score(cand)
        if q > best_q:
            best, best_q = cand, q
        if q >= 5:
            break
    if best_q < 0:
        best = "在呢，刚看到，咋啦？"
    return split_replies(best)


def split_replies(t: str):
    """把生成结果拆成 1-8 条微信短消息（每条一行）"""
    import re
    t = t.strip()
    parts = []
    for ln in t.split("\n"):
        ln = ln.strip()
        ln = re.sub(r"^[\d一二三四五六七八九十][.、)）]?\s*", "", ln)  # 去序号
        ln = ln.lstrip("-•·* ")
        if not ln:
            continue
        parts.append(ln)
    # 去重（保留顺序）
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    # 过短碎片合并到上一条，避免一堆“嗯”“好”
    merged = []
    for p in out:
        if merged and len(p) <= 4 and len(merged[-1]) > 6:
            merged[-1] += "，" + p
        else:
            merged.append(p)
    while len(merged) > 8:
        merged.pop()
    if not merged:
        merged = ["在呢，刚看到，咋啦？"]
    return [m.strip()[:60] for m in merged]


def first_reply(t: str):
    """取第一条自然回复（第一行；过长的行截到第一个句末标点）"""
    import re
    t = t.strip()
    lines = [l for l in t.split("\n") if l.strip()]
    if not lines:
        return ""
    one = lines[0].strip()
    if len(one) <= 60:
        return one
    m = re.search(r"[。！？!?]", one)
    if m and m.start() > 4:
        return one[: m.start() + 1]
    return one[:60]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/api/twin_chat":
            self.send_error(404)
            return
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln) or b"{}")
        msg = (body.get("msg") or "").strip()
        if not msg:
            self._json({"error": "empty message"})
            return
        try:
            t0 = time.time()
            replies = chat(msg, body.get("history") or [])
            print(f"[chat] {time.time()-t0:.1f}s: {msg[:30]} -> {len(replies)} 条", flush=True)
            self._json({"reply": replies[0], "replies": replies, "count": len(replies)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._json({"error": str(e)})

    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            self._json({"ok": True, "model": args.model})
        else:
            self.send_error(404)

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

print(f"[serve] HTTP 服务: http://127.0.0.1:{args.port}  (POST /api/twin_chat)", flush=True)
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
