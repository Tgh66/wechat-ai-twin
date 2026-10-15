# -*- coding: utf-8 -*-
"""ModelScope 模型下载脚本（供 WeChatTwin Go 端调用）
用法: python model_download.py <model_id> <target_dir>
输出协议:
  [DLTOTAL] <bytes>          总大小探测（0=未知）
  [DL] start <model_id>      开始下载
  [DL] progress <bytes>      已下载字节（Go 端据此算百分比/速度）
  [DL] verify <actual> <expected>  下载后完整性校验（不一致会清理重下标记）
  [DL] done                  完成（已清理 .incomplete 残留）
  [DL] error <msg>           失败
"""
import sys, os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def dl_progress():
    # 探测已下载大小（snapshot_download 的临时文件 *.incomplete 不计入）
    total = 0
    for root, dirs, files in os.walk(target):
        for f in files:
            if f.endswith(".incomplete"):
                continue
            total += os.path.getsize(os.path.join(root, f))
    return total


mid = sys.argv[1]
target = sys.argv[2]
os.makedirs(target, exist_ok=True)

expected = 0
# 探测总大小（失败不影响下载）
try:
    import urllib.request, json
    url = "https://modelscope.cn/api/v1/models/%s/repo/files?Revision=master" % mid
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    files = data.get("Data", {}).get("Files", []) or data.get("data", {}).get("files", [])
    expected = sum(int(f.get("Size", 0) or 0) for f in files)
    print("[DLTOTAL] %d" % expected, flush=True)
except Exception:
    print("[DLTOTAL] 0", flush=True)

print("[DL] start " + mid, flush=True)
from modelscope import snapshot_download
try:
    snapshot_download(mid, local_dir=target)
    # 清理 .incomplete 残留（下载中断留下的分片标记文件）
    cleaned = 0
    for root, dirs, files in os.walk(target):
        for f in files:
            if f.endswith(".incomplete"):
                try:
                    os.remove(os.path.join(root, f))
                    cleaned += 1
                except Exception:
                    pass
    # 完整性校验：实际大小 vs 期望（容差 1MB；expected=0 时跳过）
    actual = dl_progress()
    if expected > 0 and abs(actual - expected) > 1 * 1024 * 1024:
        print("[DL] verify %d %d" % (actual, expected), flush=True)
        print("[DL] error 下载文件不完整（实际 %d / 期望 %d），请重新下载" % (actual, expected), flush=True)
        sys.exit(4)
    if cleaned:
        print("[DL] cleanup incomplete=%d" % cleaned, flush=True)
    print("[DL] done", flush=True)
except Exception as e:
    print("[DL] error %r" % e, flush=True)
    sys.exit(1)

