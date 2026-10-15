# -*- coding: utf-8 -*-
"""ModelScope 模型下载脚本（供 WeChatTwin Go 端调用）
用法: python model_download.py <model_id> <target_dir>
输出协议: [DLTOTAL] <bytes> / [DL] start / [DL] done（stdout，写入 log 供界面解析）
"""
import sys, os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

mid = sys.argv[1]
target = sys.argv[2]
os.makedirs(target, exist_ok=True)

# 探测总大小（失败不影响下载）
try:
    import urllib.request, json
    url = "https://modelscope.cn/api/v1/models/%s/repo/files?Revision=master" % mid
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    files = data.get("Data", {}).get("Files", []) or data.get("data", {}).get("files", [])
    tot = sum(int(f.get("Size", 0) or 0) for f in files)
    print("[DLTOTAL] %d" % tot, flush=True)
except Exception:
    print("[DLTOTAL] 0", flush=True)

print("[DL] start " + mid, flush=True)
from modelscope import snapshot_download
snapshot_download(mid, local_dir=target)
print("[DL] done", flush=True)
