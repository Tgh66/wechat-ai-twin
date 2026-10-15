package main

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ---------------- 时间工具 ----------------

func timeUnix(ts int64) time.Time {
	return time.Unix(ts, 0).Local()
}
func timeUnixNow() int64 {
	return time.Now().Unix()
}

// ---------------- 微信目录自动发现 ----------------

// DiscoverWeChatDirs 自动搜索微信聊天记录目录（多适配）
// 1) 注册表微信安装信息 2) 已知默认路径 3) 各磁盘根目录 xwechat_files / WeChat Files
func DiscoverWeChatDirs() []string {
	dirs := []string{}
	seen := map[string]bool{}
	add := func(d string) {
		d = strings.TrimSpace(d)
		if d == "" {
			return
		}
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			if !seen[d] {
				seen[d] = true
				dirs = append(dirs, d)
			}
		}
	}

	// 1) 微信默认数据目录（新版微信 4.x：Documents\xwechat_files）
	home, _ := os.UserHomeDir()
	add(filepath.Join(home, "Documents", "xwechat_files"))
	// 旧版
	add(filepath.Join(home, "Documents", "WeChat Files"))

	// 2) 注册表（微信安装路径 + 文件保存路径）
	for _, reg := range []string{
		`HKEY_CURRENT_USER\Software\Tencent\WeChat`,
		`HKEY_CURRENT_USER\Software\Tencent\Weixin`,
		`HKEY_CURRENT_USER\Software\Tencent\xwechat`,
	} {
		if out, err := runCmd("reg", "query", reg); err == nil {
			for _, line := range strings.Split(out, "\n") {
				line = strings.TrimSpace(line)
				if strings.Contains(line, "FileSavePath") || strings.Contains(line, "InstallPath") || strings.Contains(line, "DataPath") {
					parts := strings.SplitN(line, "REG_SZ", 2)
					if len(parts) == 2 {
						p := strings.TrimSpace(parts[1])
						if strings.Contains(p, "xwechat_files") || strings.Contains(p, "WeChat Files") {
							add(p)
						}
					}
				}
			}
		}
	}

	// 3) 全盘扫描 xwechat_files / WeChat Files 目录
	for _, drive := range listDrives() {
		root := drive + `\`
		add(filepath.Join(root, "xwechat_files"))
		add(filepath.Join(root, "WeChat Files"))
		// 浅层扫描一级子目录（如 D:\数据\xwechat_files）
		if entries, err := os.ReadDir(root); err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				name := strings.ToLower(e.Name())
				if strings.Contains(name, "xwechat") || strings.Contains(name, "wechat") {
					add(filepath.Join(root, e.Name()))
				}
			}
		}
	}

	return dirs
}

func listDrives() []string {
	drives := []string{}
	out, err := runCmd("wmic", "logicaldisk", "get", "name")
	if err == nil {
		for _, line := range strings.Split(out, "\n") {
			line = strings.TrimSpace(line)
			if len(line) == 2 && line[1] == ':' {
				drives = append(drives, line[:1])
			}
		}
	}
	if len(drives) == 0 {
		drives = []string{"C", "D", "E", "F", "G"}
	}
	return drives
}

func runCmd(name string, args ...string) (string, error) {
	cmd := execCommand(name, args...)
	out, err := cmd.Output()
	return string(out), err
}
