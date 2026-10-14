# -*- coding: utf-8 -*-
"""微信数据目录发现：注册表 + 常见路径 + 磁盘索引扫描 + 手动选择 + 管理员提示"""
import os
import string
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def _expand_user_paths():
    """常见官方存储位置（含用户目录）"""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Documents", "xwechat_files"),     # 微信 4.x 默认
        os.path.join(home, "Documents", "WeChat Files"),      # 微信 3.x 默认
        os.path.join(home, "xwechat_files"),
        os.path.join(home, "WeChat Files"),
        os.path.join(home, "Documents", "Tencent Files"),
        os.path.join(home, "Documents", "WeChat"),
    ]
    return [p for p in candidates if os.path.isdir(p)]


def _registry_paths():
    """从注册表读取微信文件保存路径（3.x/4.x 均可能写入）"""
    out = []
    try:
        import winreg
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for key_path in (
                    r"Software\Tencent\WeChat",
                    r"Software\WOW6432Node\Tencent\WeChat",
                    r"Software\Tencent\xwechat",
                    r"Software\WOW6432Node\Tencent\xwechat",
            ):
                try:
                    with winreg.OpenKey(root, key_path) as k:
                        for name in ("FileSavePath", "FileSavePathOld", "FileSavePathX", "DataPath"):
                            try:
                                v, _ = winreg.QueryValueEx(k, name)
                                if v and isinstance(v, str) and os.path.isdir(v):
                                    out.append(v)
                            except OSError:
                                pass
                except OSError:
                    pass
    except Exception:
        pass
    return out


def _scan_drive_roots():
    """扫描所有磁盘根目录下的 xwechat_files / WeChat Files"""
    out = []
    for d in string.ascii_uppercase:
        root = d + ":\\"
        if not os.path.exists(root):
            continue
        for name in ("xwechat_files", "WeChat Files", "wechat_files"):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                out.append(p)
    return out


def discover_wechat_dirs():
    """自动发现所有候选微信数据根目录，返回 [(dir, 来源)]"""
    found = {}
    for p in _registry_paths():
        found[p] = "注册表"
    for p in _expand_user_paths():
        found.setdefault(p, "默认位置")
    for p in _scan_drive_roots():
        found.setdefault(p, "磁盘扫描")
    return list(found.items())


def account_dirs(root_dir):
    """列出数据根目录下的账号目录（wxid_*）"""
    if not root_dir or not os.path.isdir(root_dir):
        return []
    try:
        names = os.listdir(root_dir)
    except PermissionError:
        return []
    acc = []
    for n in sorted(names):
        p = os.path.join(root_dir, n)
        if os.path.isdir(p) and (n.startswith("wxid_") or n == "all_users"):
            if n == "all_users":
                continue
            if os.path.isdir(os.path.join(p, "db_storage")):
                acc.append(p)
    return acc


def is_wechat_running():
    try:
        for exe in ("Weixin.exe", "WeChat.exe"):
            r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq " + exe],
                               capture_output=True, text=True, timeout=10,
                               encoding="gbk", errors="ignore",
                               creationflags=0x08000000 if os.name == "nt" else 0)
            if exe in r.stdout:
                return True
        return False
    except Exception:
        return False


def need_admin():
    """检测是否有管理员权限"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin()) is False
    except Exception:
        return False


def scan_for_db_dirs(root_dir):
    """在数据根目录下递归（浅层）寻找 db_storage/message 目录"""
    hits = []
    if not root_dir or not os.path.isdir(root_dir):
        return hits
    for acc in account_dirs(root_dir):
        m = os.path.join(acc, "db_storage", "message")
        if os.path.isdir(m):
            hits.append(m)
    return hits


def format_discovery(found, accounts):
    """把发现结果格式化为可读文本"""
    lines = ["【已发现微信数据位置】"]
    for d, src in found:
        accs = account_dirs(d)
        lines.append("• %s （来源：%s）" % (d, src))
        for a in accs:
            n_msg = len([f for f in os.listdir(os.path.join(a, "db_storage", "message"))
                         if f.endswith(".db")]) if os.path.isdir(os.path.join(a, "db_storage", "message")) else 0
            lines.append("    ├ %s  （消息库 %d 个）" % (os.path.basename(a), n_msg))
    if not found:
        lines.append("（未自动找到，请手动选择文件夹）")
    if need_admin():
        lines.append("")
        lines.append("⚠ 当前不是管理员权限。若下方目录无权限访问，请用「以管理员身份运行」重启本程序。")
    return "\n".join(lines)
