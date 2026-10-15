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
            name = pynvml.nvmlDeviceGetName(h).decode() if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else pynvml.nvmlDeviceGetName(h)
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
    """根据显存推荐可用模型（QLoRA 4bit 已在 6GB 显卡验证可训练 Qwen3-4B）"""
    infos = gpu_info()
    if not infos or "error" in infos[0]:
        return [("Qwen/Qwen3-4B", "Qwen3-4B（CPU 可用，QLoRA 4bit 训练）")]
    total = max(g["total_gb"] for g in infos)
    if total >= 20:
        return [
            ("Qwen/Qwen3-8B", "Qwen3-8B（QLoRA 4bit，16GB 训练）"),
            ("Qwen/Qwen3-4B", "Qwen3-4B（QLoRA 4bit，8GB 训练）"),
            ("Qwen/Qwen3.5-2B", "Qwen3.5-2B（bf16 LoRA，6GB 训练）"),
        ]
    if total >= 10:
        return [
            ("Qwen/Qwen3-8B", "Qwen3-8B（QLoRA 4bit，10GB 训练）"),
            ("Qwen/Qwen3-4B", "Qwen3-4B（QLoRA 4bit，6GB 训练）"),
            ("Qwen/Qwen3.5-2B", "Qwen3.5-2B（bf16 LoRA，6GB 训练）"),
        ]
    if total >= 6:
        return [
            ("Qwen/Qwen3-4B", "Qwen3-4B（QLoRA 4bit，6GB 训练，已验证）"),
            ("Qwen/Qwen3-1.7B", "Qwen3-1.7B（bf16 LoRA，5GB 训练）"),
            ("Qwen/Qwen3.5-2B", "Qwen3.5-2B（bf16 LoRA，6GB 训练）"),
        ]
    if total >= 4:
        return [
            ("Qwen/Qwen3-1.7B", "Qwen3-1.7B（bf16 LoRA，5GB 训练）"),
            ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B（bf16 LoRA，4GB 训练）"),
            ("Qwen/Qwen3-0.6B", "Qwen3-0.6B（bf16 LoRA，2GB 训练）"),
        ]
    return [
        ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B（bf16 LoRA，4GB 训练）"),
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B（bf16 LoRA，2GB 训练）"),
        ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B（bf16 LoRA，1GB 训练）"),
    ]


# 预置模型下载表（ModelScope，国内可直接下载；确定可用）
MODEL_PRESETS = [
    ("Qwen/Qwen3-4B", "Qwen3-4B · 4B 参数 · 6GB 显存 QLoRA 训练（推荐）"),
    ("Qwen/Qwen3.5-2B", "Qwen3.5-2B · 2B 参数 · 6GB 显存 bf16 训练"),
    ("Qwen/Qwen3-1.7B", "Qwen3-1.7B · 1.7B 参数 · 5GB 显存 bf16 训练"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B · 1.5B 参数 · 4GB 显存 bf16 训练"),
    ("Qwen/Qwen3-0.6B", "Qwen3-0.6B · 0.6B 参数 · 2GB 显存 bf16 训练"),
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B · 0.5B 参数 · 1GB 显存 bf16 训练"),
]
