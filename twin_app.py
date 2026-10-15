# -*- coding: utf-8 -*-
"""
微信 AI 数字分身 · 全功能工作台
================================
单页应用，包含：
  ① 对话   —— 数字分身聊天（一次回多条微信消息，气泡展示）
  ② 情感   —— 输入文字/选会话做情感分析（规则评分 + 数字分身读心 + 情感小模型）
  ③ 分析   —— 指定会话的聊天记录分析（条数/占比/时间跨度/高频词/情感倾向/最近消息）

启动：python twin_app.py   →  http://127.0.0.1:7860
自动识别 models/ 下所有已训练好的 merged 模型，可下拉切换。
"""
import json
import os
import re
import sys
import threading
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(APP_DIR, "models")
DEFAULT_JSONL = os.path.join(DATA_DIR, "chat_messages_e.jsonl")

IM_END = 248046  # <|im_end|>

_MODEL = None
_TOK = None
_MODEL_NAME = None
_LOCK = threading.Lock()

# ---------------- 模型管理 ----------------

def list_merged_models():
    """识别 models/ 下所有已合并（可用）的模型目录"""
    out = []
    if os.path.isdir(MODELS_DIR):
        for n in sorted(os.listdir(MODELS_DIR)):
            p = os.path.join(MODELS_DIR, n)
            if os.path.isdir(p) and any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(p)):
                out.append(p)
    return out


def pick_default_model():
    merged = [p for p in list_merged_models() if "merged" in os.path.basename(p).lower()]
    merged.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return merged[0] if merged else (list_merged_models() or [None])[0]


def load_model(model_dir):
    global _MODEL, _TOK, _MODEL_NAME
    with _LOCK:
        if _MODEL is not None and _MODEL_NAME == model_dir:
            return "已加载: %s" % os.path.basename(model_dir)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if _MODEL is not None:
            _MODEL.cpu()
            del _MODEL
            torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda")
        model.eval()
        _MODEL, _TOK, _MODEL_NAME = model, tok, model_dir
        return "已加载: %s（%s）" % (os.path.basename(model_dir), _gpu_short())


def _gpu_short():
    try:
        import torch
        if torch.cuda.is_available():
            return "GPU 显存可用 %.1fGB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    except Exception:
        pass
    return "CPU"


def generate(user_text, history=None, system_prompt=None, max_new_tokens=220, temperature=0.85):
    if _MODEL is None:
        return "（请先加载模型）"
    import torch
    msgs = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM}]
    if history:
        for h in history[-8:]:
            msgs.append({"role": "user", "content": h["user"]})
            msgs.append({"role": "assistant", "content": h["assistant"]})
    msgs.append({"role": "user", "content": user_text})
    text = _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = _TOK(text, return_tensors="pt").to(next(_MODEL.parameters()).device)
    eos = _TOK.eos_token_id
    pad = _TOK.pad_token_id if _TOK.pad_token_id is not None else eos
    with torch.no_grad():
        out = _MODEL.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                              temperature=temperature, top_p=0.9, repetition_penalty=1.08,
                              eos_token_id=eos, pad_token_id=pad)
    gen = _TOK.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = gen.replace("<think>", "").replace("</think>", "").strip()
    # 微信连发风格：按换行拆成多条消息（去掉首条反问/复读，最多 5 条）
    parts = [p.strip() for p in re.split(r"[\n]+", gen) if p.strip()]
    keep = []
    for p in parts:
        if not keep and len(p) <= 8 and p.endswith(("？", "吗", "呢", "嘛", "？")) and "我" in p:
            continue
        keep.append(p[:120])
        if len(keep) >= 5:
            break
    return keep if keep else [gen[:120]]


DEFAULT_SYSTEM = ("你是微信用户\"挽星宴君安\"的数字分身。你正在微信上和朋友聊天，请用你自己一贯的真实、自然的"
                  "说话方式、语气和习惯回复。像微信聊天一样：可以一次连着发 2~5 条短消息，每条一句话，用换行分隔。"
                  "不要长篇大论，不要客套，不要替对方说话。")

# ---------------- 数据分析（流式，不整文件入内存） ----------------

def _scan_chats(jsonl_path):
    """返回 {会话名: {total, self, other, first_ts, last_ts}}（流式）"""
    chats = {}
    if not os.path.exists(jsonl_path):
        return chats
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            name = r.get("name") or r.get("user") or "?"
            c = chats.setdefault(name, {"total": 0, "self": 0, "other": 0, "first": 0, "last": 0})
            c["total"] += 1
            if r.get("is_self"):
                c["self"] += 1
            else:
                c["other"] += 1
            ts = r.get("ts") or 0
            if ts:
                if not c["first"] or ts < c["first"]:
                    c["first"] = ts
                if ts > c["last"]:
                    c["last"] = ts
    return chats


def _analyze_chat(jsonl_path, chat_name, limit=20000):
    """流式统计指定会话：消息序列、自己/对方、情感、高频词、最近消息"""
    pos = {"哈哈", "开心", "好", "棒", "牛", "可以", "行", "没问题", "爱", "喜欢", "期待",
           "笑死", "太强", "优秀", "爽", "顺利", "恭喜", "谢谢", "感谢", "晚安", "么么", "嘻嘻", "嘿嘿", "耶"}
    neg = {"烦", "累", "难受", "气", "烦死", "无语", "崩溃", "哭", "难", "亏", "糟",
           "讨厌", "伤心", "难过", "焦虑", "压力", "坑", "生气", "不行", "完了", "叹气"}
    out = {"total": 0, "self": 0, "other": 0, "first": 0, "last": 0,
           "sentiment": 0, "sent_n": 0, "words": {}, "recent": [], "type_cnt": {}}
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("name") or r.get("user")) != chat_name:
                continue
            out["total"] += 1
            is_self = bool(r.get("is_self"))
            if is_self:
                out["self"] += 1
            else:
                out["other"] += 1
            ts = r.get("ts") or 0
            if ts:
                if not out["first"] or ts < out["first"]:
                    out["first"] = ts
                if ts > out["last"]:
                    out["last"] = ts
            c = r.get("content") or ""
            t = r.get("type") or "?"
            out["type_cnt"][t] = out["type_cnt"].get(t, 0) + 1
            if is_self and t == "文本" and 2 <= len(c) <= 150:
                s = 0
                for w in pos:
                    if w in c:
                        s += 1
                for w in neg:
                    if w in c:
                        s -= 1
                if c.strip():
                    out["sentiment"] += s
                    out["sent_n"] += 1
                try:
                    import jieba
                    for w in jieba.cut(c):
                        w = w.strip()
                        if len(w) >= 2:
                            out["words"][w] = out["words"].get(w, 0) + 1
                except Exception:
                    pass
            if t == "文本" and c.strip():
                out["recent"].append({"self": is_self, "content": c[:100], "ts": ts})
    if out["sent_n"]:
        out["sentiment"] = round(max(-100, min(100, out["sentiment"] / out["sent_n"] * 40)), 1)
    out["words"] = sorted(out["words"].items(), key=lambda x: -x[1])[:30]
    out["recent"] = out["recent"][-50:]
    return out


# ---------------- HTTP 服务 ----------------

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>微信 AI 数字分身 · 全功能工作台</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:linear-gradient(135deg,#f5f7ff,#eef1fb 40%,#fdeef0);min-height:100vh}
.top{background:linear-gradient(90deg,#1a1a2e,#3a2e6e);color:#fff;padding:14px 24px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:9}
.top h1{font-size:18px;font-weight:700}
.top .sub{font-size:12px;opacity:.75}
.tabs{display:flex;gap:6px;margin-left:auto}
.tab{padding:8px 18px;border-radius:20px;cursor:pointer;font-size:14px;background:rgba(255,255,255,.12);color:#fff;border:none}
.tab.active{background:#4f6df5;font-weight:600}
.wrap{max-width:980px;margin:0 auto;padding:20px}
.page{display:none}
.page.active{display:block}
.card{background:#fff;border-radius:14px;box-shadow:0 4px 20px rgba(30,40,90,.06);padding:18px;margin-bottom:16px}
.card h3{font-size:15px;color:#1a1a2e;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.badge{background:#eef1fb;color:#4f6df5;border-radius:8px;padding:2px 8px;font-size:12px}
/* 对话 */
#chatbox{height:56vh;overflow-y:auto;padding:8px 2px;display:flex;flex-direction:column;gap:10px}
.msg{display:flex;flex-direction:column;max-width:78%}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.twin{align-self:flex-start;align-items:flex-start}
.msg .row{background:#f2f4f8;border-radius:14px;padding:8px 12px;margin:3px 0;line-height:1.5;font-size:14px;white-space:pre-wrap;word-break:break-word;border-bottom-left-radius:3px}
.msg.user .row{background:#4f6df5;color:#fff;border-radius:14px;border-bottom-right-radius:3px}
.msg .time{font-size:11px;color:#999;margin:0 6px}
.inputbar{display:flex;gap:10px;margin-top:10px}
#inp{flex:1;border:1px solid #d5d9e2;border-radius:12px;padding:12px 14px;font-size:15px;outline:none}
#inp:focus{border-color:#4f6df5}
button.primary{background:#4f6df5;color:#fff;border:none;border-radius:12px;padding:10px 22px;font-size:14px;cursor:pointer;font-weight:600}
button.primary:disabled{opacity:.5}
button.ghost{background:#f2f4f8;color:#333;border:1px solid #dfe3ea;border-radius:10px;padding:8px 16px;font-size:13px;cursor:pointer}
select,input[type=text]{border:1px solid #d5d9e2;border-radius:10px;padding:9px 12px;font-size:14px;outline:none;background:#fff}
textarea{border:1px solid #d5d9e2;border-radius:10px;padding:10px 12px;font-size:14px;outline:none;width:100%;font-family:inherit;resize:vertical}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.metric{display:flex;flex-direction:column;background:#f7f8fc;border-radius:10px;padding:12px 14px}
.metric .v{font-size:22px;font-weight:800;color:#1a1a2e}
.metric .k{font-size:12px;color:#777;margin-top:2px}
.senti{display:flex;align-items:center;gap:12px;margin-top:8px}
.senti .bar{flex:1;height:12px;border-radius:6px;background:linear-gradient(90deg,#e85d75,#f5c75c 50%,#27ae60);position:relative}
.senti .mark{position:absolute;top:-4px;width:4px;height:20px;background:#1a1a2e;border-radius:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #eef0f4}
th{color:#666;font-weight:600;background:#fafbfd}
.tag{display:inline-block;background:#eef1fb;color:#4f6df5;border-radius:6px;padding:2px 8px;font-size:12px;margin:2px}
.warn{background:#fff7e8;border:1px solid #f5d98a;color:#8a6d1f;border-radius:10px;padding:10px 14px;font-size:13px;margin-bottom:14px}
a{color:#4f6df5;text-decoration:none}
</style></head><body>

<div class="top">
  <h1>💬 微信 AI 数字分身</h1>
  <span class="sub" id="modelinfo">正在检测本地模型…</span>
  <div class="tabs">
    <button class="tab active" onclick="sw('chat')">对话</button>
    <button class="tab" onclick="sw('senti')">情感分析</button>
    <button class="tab" onclick="sw('ana')">聊天记录分析</button>
  </div>
</div>

<div class="wrap">
<div id="chat" class="page active">
  <div class="card">
    <h3>和「数字分身」聊天 <span class="badge">像微信一样，一次回 2~5 条</span></h3>
    <div id="chatbox"><div class="twin" style="display:flex;flex-direction:column"><div class="row" style="max-width:78%">你好，我是你的微信数字分身。随便聊聊~</div></div></div>
    <div class="inputbar">
      <input id="inp" placeholder="输入消息，回车发送…" onkeydown="if(event.key==='Enter')send()">
      <button class="primary" id="btn" onclick="send()">发送</button>
      <button class="ghost" onclick="clearChat()">清空</button>
    </div>
  </div>
  <div class="card">
    <h3>模型设置</h3>
    <div class="grid2">
      <div>
        <div style="font-size:13px;color:#666;margin-bottom:6px">已训练的本地模型（自动识别）</div>
        <select id="model_sel" style="width:100%" onchange="loadModel()"></select>
      </div>
      <div>
        <div style="font-size:13px;color:#666;margin-bottom:6px">人设（system prompt）</div>
        <textarea id="sysp" rows="3">你是微信用户"挽星宴君安"的数字分身。你正在微信上和朋友聊天，请用你自己一贯的真实、自然的说话方式、语气和习惯回复。像微信聊天一样：可以一次连着发 2~5 条短消息，每条一句话，用换行分隔。不要长篇大论，不要客套，不要替对方说话。</textarea>
      </div>
    </div>
    <div style="display:flex;gap:10px;margin-top:10px;align-items:center">
      <button class="ghost" onclick="loadModel()">🔄 切换/加载模型</button>
      <span id="model_status" style="font-size:13px;color:#666"></span>
    </div>
  </div>
</div>

<div id="senti" class="page">
  <div class="warn">情感分析<strong>不需要微调模型</strong>也能用（规则评分）；"数字分身读心"会用你微调好的模型分析聊天记录里的真实情绪。</div>
  <div class="card">
    <h3>🧠 情感倾向度（规则评分，-100~+100）</h3>
    <div class="grid2">
      <div>
        <div style="font-size:13px;color:#666;margin-bottom:6px">选择会话</div>
        <select id="senti_chat" style="width:100%"></select>
      </div>
      <div style="display:flex;align-items:flex-end">
        <button class="primary" style="width:100%" onclick="sentimentScore()">分析情感倾向度</button>
      </div>
    </div>
    <div id="senti_res" style="margin-top:12px"></div>
  </div>
  <div class="card">
    <h3>🔮 数字分身读心（用你的模型分析对话里的真实感受）</h3>
    <div style="font-size:13px;color:#666;margin-bottom:8px">取所选会话最近 N 条消息，让数字分身分析：你当时的情绪、对对方的感受。</div>
    <div class="grid2">
      <select id="twin_chat" style="width:100%"></select>
      <div style="display:flex;gap:8px">
        <input type="text" id="twin_n" value="60" style="width:90px" placeholder="条数">
        <button class="primary" style="flex:1" onclick="twinMind()">读心</button>
      </div>
    </div>
    <div id="twin_res" style="margin-top:12px;background:#f7f8fc;border-radius:10px;padding:12px;font-size:14px;line-height:1.7;white-space:pre-wrap"></div>
  </div>
</div>

<div id="ana" class="page">
  <div class="card">
    <h3>📊 聊天记录分析</h3>
    <div class="grid2">
      <div>
        <div style="font-size:13px;color:#666;margin-bottom:6px">选择会话</div>
        <select id="ana_chat" style="width:100%"></select>
      </div>
      <div style="display:flex;align-items:flex-end">
        <button class="primary" style="width:100%" onclick="analyze()">分析</button>
      </div>
    </div>
    <div id="ana_res" style="margin-top:14px"></div>
  </div>
  <div class="card">
    <h3>📈 高频词 TOP</h3>
    <div id="ana_words" style="font-size:13px;line-height:2"></div>
  </div>
</div>
</div>

<script>
let hist=[];
const $=id=>document.getElementById(id);
function sw(id){document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
$(id).classList.add('active');
document.querySelector('.tab'+(id==='chat'?':nth-child(1)':id==='senti'?':nth-child(2)':':nth-child(3)')).classList.add('active');}

// 模型
async function refreshModels(){const r=await fetch('/api/models');const d=await r.json();
const sel=$('model_sel');sel.innerHTML='';
(d.models||[]).forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m.split(/[\\/]/).pop();sel.appendChild(o);});
$('modelinfo').textContent='已识别 '+d.models.length+' 个本地模型'+(d.current?' · 当前: '+d.current.split(/[\\/]/).pop():'');
if(d.models.length){$('model_status').textContent='请点击"加载模型"开始对话';}}
async function loadModel(){const m=$('model_sel').value;if(!m)return;
$('model_status').textContent='加载中…（4B 模型约需 30 秒）';
const r=await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:m})});
const d=await r.json();$('model_status').textContent=d.msg;
$('modelinfo').textContent='模型: '+m.split(/[\\/]/).pop();}
// 对话
function addBubbles(who,parts){const box=$('chatbox');
const wrap=document.createElement('div');wrap.className='msg '+who;
parts.forEach(t=>{const row=document.createElement('div');row.className='row';row.textContent=t;wrap.appendChild(row);});
box.appendChild(wrap);box.scrollTop=1e9;}
async function send(){const inp=$('inp');const txt=inp.value.trim();if(!txt)return;
inp.value='';addBubbles('user',[txt]);const btn=$('btn');btn.disabled=true;
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:txt,history:hist,sys:$('sysp').value})});
const d=await r.json();
if(d.reply&&d.reply.length){addBubbles('twin',d.reply);hist.push({user:txt,assistant:d.reply.join('\n')});hist=hist.slice(-12);}
else addBubbles('twin',['（模型未加载或出错：'+(d.error||'')+'）']);}
catch(e){addBubbles('twin',['（出错了：'+e.message+'）']);}
btn.disabled=false;}
function clearChat(){$('chatbox').innerHTML='';hist=[];}
// 会话列表
async function refreshChats(){const r=await fetch('/api/chats');const d=await r.json();
const cs=d.chats||[];
['senti_chat','twin_chat','ana_chat'].forEach(id=>{const sel=$(id);sel.innerHTML='';
cs.forEach(c=>{const o=document.createElement('option');o.value=c.name;o.textContent=c.name+'（'+c.total+'条）';sel.appendChild(o);});});}
// 情感
async function sentimentScore(){const name=$('senti_chat').value;if(!name)return;
const r=await fetch('/api/sentiment?chat='+encodeURIComponent(name));const d=await r.json();
const s=d.sentiment;
const label=s<=-50?'非常负面':s<-10?'偏负面':s>10?'偏正面':s>=50?'非常正面':'中性';
$('senti_res').innerHTML='<div class="metric"><div class="k">会话「'+name+'」情感倾向度（样本 '+d.n+' 条自己发言）</div>'+
'<div class="senti"><div style="font-size:26px;font-weight:800;min-width:120px">'+s+' / 100（'+label+'）</div>'+
'<div class="bar"><div class="mark" style="left:'+(s+100)/2+'%"></div></div></div></div>';}
async function twinMind(){const name=$('twin_chat').value;const n=$('twin_n').value||60;if(!name)return;
$('twin_res').textContent='数字分身正在读心…（约 20~60 秒）';
try{const r=await fetch('/api/twin?chat='+encodeURIComponent(name)+'&n='+n);const d=await r.json();
$('twin_res').textContent=d.result||('（'+d.error+'）');}catch(e){$('twin_res').textContent='出错：'+e.message;}}
// 分析
async function analyze(){const name=$('ana_chat').value;if(!name)return;
const r=await fetch('/api/analyze?chat='+encodeURIComponent(name));const d=await r.json();
const a=d.data;
const fmt=ts=>ts?new Date(ts*1000).toLocaleDateString('zh-CN'):'-';
$('ana_res').innerHTML='<div class="grid2">'+
'<div class="metric"><div class="v">'+a.total+'</div><div class="k">总消息</div></div>'+
'<div class="metric"><div class="v">'+a.self+' / '+a.other+'</div><div class="k">自己 / 对方</div></div>'+
'<div class="metric"><div class="v">'+a.sentiment+'</div><div class="k">情感倾向度</div></div>'+
'<div class="metric"><div class="v">'+fmt(a.first)+' ~ '+fmt(a.last)+'</div><div class="k">时间跨度</div></div></div>'+
'<h3 style="margin:16px 0 8px">最近消息</h3><table><tr><th>时间</th><th>谁</th><th>内容</th></tr>'+
(a.recent||[]).slice(-12).reverse().map(m=>'<tr><td>'+fmt(m.ts)+'</td><td>'+(m.self?'我':'对方')+'</td><td>'+m.content.replace(/</g,'&lt;')+'</td></tr>').join('')+
'</table>';
$('ana_words').innerHTML=(a.words||[]).slice(0,25).map(w=>'<span class="tag">'+w[0]+' ×'+w[1]+'</span>').join(' ');}

refreshModels();refreshChats();
setInterval(()=>{refreshChats();},60000);
</script>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        q = {}
        if "?" in self.path:
            for kv in self.path.split("?", 1)[1].split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    q[k] = v
        try:
            if path == "/api/models":
                self._send({"models": list_merged_models(),
                            "current": _MODEL_NAME})
            elif path == "/api/chats":
                chats = _scan_chats(DEFAULT_JSONL)
                items = sorted([{"name": k, "total": v["total"]} for k, v in chats.items()
                                if v["total"] >= 50], key=lambda x: -x["total"])[:500]
                self._send({"chats": items})
            elif path == "/api/sentiment":
                name = q.get("chat", "")
                data = _analyze_chat(DEFAULT_JSONL, name)
                self._send({"sentiment": data["sentiment"], "n": data["sent_n"]})
            elif path == "/api/twin":
                name = q.get("chat", "")
                n = int(q.get("n", "60") or "60")
                data = _analyze_chat(DEFAULT_JSONL, name)
                if _MODEL is None:
                    self._send({"error": "请先在对话页加载模型"}, 400)
                    return
                rows = data["recent"][-n:]
                if not rows:
                    self._send({"error": "会话无文本消息"}, 400)
                    return
                text = "\n".join((("我" if m["self"] else "对方") + ": " + m["content"]) for m in rows)
                prompt = ("这是\"我\"与对方的微信聊天记录片段：\n---\n%s\n---\n"
                          "请分析：1) \"我\"当时的主要情绪；2) \"我\"对对方的真实感受；"
                          "3) 整体情感倾向（正面/负面/中性，-100~100 打分）。简短自然的中文回答。" % text[:1800])
                try:
                    result = generate(prompt, system_prompt="你是情绪分析师，读对话判断说话人的真实感受。",
                                      max_new_tokens=260, temperature=0.6)
                    self._send({"result": "\n".join(result) if isinstance(result, list) else result})
                except Exception as e:
                    self._send({"error": str(e)}, 500)
            elif path == "/api/analyze":
                name = q.get("chat", "")
                data = _analyze_chat(DEFAULT_JSONL, name)
                self._send({"data": data})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": str(e)}, 500)

    def do_POST(self):
        path = self.path.split("?")[0]
        ln = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            data = {}
        try:
            if path == "/api/load":
                model_dir = data.get("model") or ""
                if not model_dir or not os.path.isdir(model_dir):
                    self._send({"msg": "模型目录不存在: %s" % model_dir}, 400)
                    return
                try:
                    msg = load_model(model_dir)
                    self._send({"msg": msg})
                except Exception as e:
                    self._send({"msg": "加载失败: %s" % e}, 500)
            elif path == "/api/chat":
                if _MODEL is None:
                    self._send({"error": "模型未加载，请先在模型设置中加载"}, 400)
                    return
                reply = generate(data.get("text", ""), data.get("history") or [],
                                 system_prompt=data.get("sys") or None)
                self._send({"reply": reply if isinstance(reply, list) else [reply]})
            else:
                self._send({"error": "not found"}, 404)
        except Exception as e:
            self._send({"error": str(e)}, 500)


def main():
    default = pick_default_model()
    print("=" * 54)
    print("微信 AI 数字分身 · 全功能工作台")
    print("浏览器打开: http://127.0.0.1:7860")
    print("自动识别模型:", default or "（models/ 下暂无模型）")
    print("聊天记录数据:", DEFAULT_JSONL)
    print("=" * 54)
    if default:
        try:
            print("预加载模型…", flush=True)
            msg = load_model(default)
            print(msg, flush=True)
        except Exception as e:
            print("预加载失败（可稍后在页面手动加载）: %s" % e, flush=True)
    ThreadingHTTPServer(("127.0.0.1", 7860), H).serve_forever()


if __name__ == "__main__":
    main()
