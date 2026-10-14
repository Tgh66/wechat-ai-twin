# -*- coding: utf-8 -*-
"""GPU 检测与模型推荐"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


def gpu_info():
    """返回 [{index, name, total_gb, used_gb, util}]；无 NVIDIA GPU 返回 []"""
    if not _HAS_NVML:
        return []
    out = []
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = 0
            out.append({
                "index": i, "name": name,
                "total_gb": round(mem.total / 1024 ** 3, 1),
                "used_gb": round(mem.used / 1024 ** 3, 1),
                "free_gb": round((mem.total - mem.used) / 1024 ** 3, 1),
                "util": util,
            })
        pynvml.nvmlShutdown()
    except Exception as e:
        return [{"error": str(e)}]
    return out


def gpu_summary():
    infos = gpu_info()
    if not infos:
        return "未检测到 NVIDIA GPU（可用 CPU 训练，速度较慢）"
    lines = []
    for g in infos:
        if "error" in g:
            return "GPU 读取失败: %s" % g["error"]
        lines.append("GPU%d %s | 显存 %.1f/%.1f GB | 占用 %d%%" % (
            g["index"], g["name"], g["used_gb"], g["total_gb"], g["util"]))
    return "\n".join(lines)


def recommend_model():
    """根据显存推荐可用模型（bf16/LoRA 可训练）"""
    infos = gpu_info()
    if not infos or "error" in infos[0]:
        return [("Qwen/Qwen3.5-2B", "Qwen3.5-2B（CPU 可用，2B 参数量）")]
    total = max(g["total_gb"] for g in infos)
    if total >= 20:
        return [
            ("Qwen/Qwen3-8B", "Qwen3-8B（16GB 显存训练）"),
            ("Qwen/Qwen3-4B", "Qwen3-4B（8GB 显存训练）"),
            ("Qwen/Qwen3.5-2B", "Qwen3.5-2B（6GB 显存训练）"),
        ]
    if total >= 10:
        return [
            ("Qwen/Qwen3-4B", "Qwen3-4B（8GB 显存训练）"),
            ("Qwen/Qwen3.5-2B", "Qwen3.5-2B（6GB 显存训练）"),
            ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B（4GB 显存训练）"),
        ]
    if total >= 6:
        return [
            ("Qwen/Qwen3.5-2B", "Qwen3.5-2B（6GB 显存训练）"),
            ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B（4GB 显存训练）"),
            ("Qwen/Qwen3-0.6B", "Qwen3-0.6B（2GB 显存训练）"),
        ]
    return [
        ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B（4GB 显存训练）"),
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B（2GB 显存训练）"),
        ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B（1GB 显存训练）"),
    ]
