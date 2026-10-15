package main

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
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

// WeChatDir 一个微信数据目录候选（结构化，供前端列表选择）
type WeChatDir struct {
	Path         string   `json:"path"`
	Name         string   `json:"name"`
	Format       string   `json:"format"` // "new"=微信4.0 xwechat_files, "old"=旧版 WeChat Files
	Accounts     int      `json:"accounts"`
	AccountsList []string `json:"accounts_list"`
	JSONL        []string `json:"jsonl"` // 目录内（浅层递归）发现的 JSONL 导出
	Updated      string   `json:"updated"`
	Verified     bool     `json:"verified"`
}

// DiscoverWeChatDirs 自动搜索微信聊天记录目录（全盘扫描 + 双重校验）
// 返回: 结构化候选列表 dirs；同时附带已发现的 JSONL 数据源 jsonl（data/ 或 ai-twin/data）
func DiscoverWeChatDirs() map[string]interface{} {
	cands := []WeChatDir{}
	seen := map[string]bool{}

	for _, root := range discoverWeChatRoots() {
		key := strings.ToLower(strings.TrimRight(root, `\`))
		if seen[key] {
			continue
		}
		seen[key] = true
		if d := analyzeWeChatRoot(root); d != nil {
			cands = append(cands, *d)
		}
	}

	// 排序：新版优先、账号多优先、更新时间新优先
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Format != cands[j].Format {
			return cands[i].Format == "new"
		}
		if cands[i].Accounts != cands[j].Accounts {
			return cands[i].Accounts > cands[j].Accounts
		}
		return cands[i].Updated > cands[j].Updated
	})

	return map[string]interface{}{
		"dirs":  cands,
		"jsonl": findJSONL(), // data/ 或 ai-twin/data 下已导出的聊天记录
	}
}

// discoverWeChatRoots 全盘扫描所有盘符，收集微信数据根目录
// 覆盖: ① 用户文档默认路径 ② 注册表保存路径 ③ 所有盘符根目录及一级子目录
func discoverWeChatRoots() []string {
	roots := []string{}
	seen := map[string]bool{}
	add := func(d string) {
		d = strings.TrimSpace(d)
		if d == "" {
			return
		}
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			k := strings.ToLower(strings.TrimRight(d, `\`))
			if !seen[k] {
				seen[k] = true
				roots = append(roots, d)
			}
		}
	}

	// 1) 用户文档默认路径
	home, _ := os.UserHomeDir()
	docs := filepath.Join(home, "Documents")
	add(filepath.Join(docs, "xwechat_files"))
	add(filepath.Join(docs, "WeChat Files"))

	// 2) 注册表（微信安装信息里的数据保存路径）
	for _, reg := range []string{
		`HKEY_CURRENT_USER\Software\Tencent\WeChat`,
		`HKEY_CURRENT_USER\Software\Tencent\Weixin`,
		`HKEY_CURRENT_USER\Software\Tencent\xwechat`,
	} {
		for _, p := range regSavePaths(reg) {
			add(p)
		}
	}

	// 3) 全盘扫描：每个盘符根目录 + 一级子目录
	for _, drive := range listDrives() {
		root := drive + `:\`
		add(filepath.Join(root, "xwechat_files"))
		add(filepath.Join(root, "WeChat Files"))
		if entries, err := os.ReadDir(root); err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				n := strings.ToLower(e.Name())
				// 一级子目录命中（如 D:\数据\xwechat_files）
				if n == "xwechat_files" || n == "wechat files" ||
					(strings.Contains(n, "xwechat") && strings.Contains(n, "files")) ||
					(strings.Contains(n, "wechat") && strings.Contains(n, "files")) {
					add(filepath.Join(root, e.Name()))
				}
			}
		}
	}
	return roots
}

// regSavePaths 从注册表查询微信数据保存路径（FileSavePath / DataPath）
func regSavePaths(reg string) []string {
	out, err := runCmd("reg", "query", reg)
	if err != nil {
		return nil
	}
	var paths []string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "FileSavePath") && !strings.Contains(line, "DataPath") {
			continue
		}
		parts := strings.SplitN(line, "REG_SZ", 2)
		if len(parts) != 2 {
			continue
		}
		p := strings.TrimSpace(parts[1])
		if p != "" {
			paths = append(paths, p)
		}
	}
	return paths
}

// listDrives 枚举所有存在的盘符（多方法兜底，确保不遗漏）
func listDrives() []string {
	drives := []string{}
	seen := map[string]bool{}
	add := func(d string) {
		d = strings.TrimSpace(d)
		if len(d) >= 2 && d[1] == ':' {
			dr := strings.ToUpper(d[:1])
			if dr >= "A" && dr <= "Z" && !seen[dr] {
				seen[dr] = true
				drives = append(drives, dr)
			}
		}
	}
	// 1) wmic（老系统）
	if out, err := runCmd("wmic", "logicaldisk", "get", "name"); err == nil {
		for _, line := range strings.Split(out, "\n") {
			add(line)
		}
	}
	// 2) PowerShell 兜底
	if out, err := runCmd("powershell", "-NoProfile", "-Command", "Get-PSDrive -PSProvider FileSystem | Select-Object -ExpandProperty Name"); err == nil {
		for _, line := range strings.Split(out, "\n") {
			add(strings.TrimSpace(line) + ":")
		}
	}
	// 3) A-Z 直接探测兜底（最后保障）
	for c := 'A'; c <= 'Z'; c++ {
		dr := string(c) + ":"
		if _, err := os.Stat(dr + `\`); err == nil {
			add(dr)
		}
	}
	if len(drives) == 0 {
		drives = []string{"C", "D", "E", "F", "G"}
	}
	sort.Strings(drives)
	return drives
}

// ---------------- 目录校验（目录名 + 内部特征 双重判定） ----------------

// 新版微信 4.0 账号目录：wxid_xxx_序号（如 wxid_e6o9se8d4ob922_1f12）
var newAcctRe = regexp.MustCompile(`(?i)^wxid_[\w-]+_[0-9a-f]{2,}$`)

// 旧版账号目录：wxid_xxx（也可能是中文联系人名）
var oldAcctRe = regexp.MustCompile(`(?i)^wxid_`)

// 明显不是账号目录的通用名称（两版通用）
var nonAccountNames = map[string]bool{
	"all_users": true, "all users": true, "backup": true, "old_backup": true,
	"applet": true, "wmpf": true, "temp": true, "cache": true, "config": true,
	"resource": true, "msg": true, "db_storage": true, "apm_record": true,
	"business": true, "migrate": true, "wechat": true,
}

// analyzeWeChatRoot 分析一个微信根目录，返回结构化候选；无有效账号目录时返回 nil
func analyzeWeChatRoot(root string) *WeChatDir {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	lower := strings.ToLower(root)
	isNew := strings.Contains(lower, "xwechat")
	isOld := strings.Contains(lower, "wechat files") || strings.Contains(lower, "wechat_files")

	valid := []string{}
	latest := time.Time{}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		sub := filepath.Join(root, e.Name())
		if isWeChatAccountDir(sub, e.Name(), isNew) {
			valid = append(valid, e.Name())
			if fi, err := os.Stat(sub); err == nil && fi.ModTime().After(latest) {
				latest = fi.ModTime()
			}
		}
	}
	if len(valid) == 0 {
		return nil // 根目录存在但无有效账号目录 → 不列为候选
	}

	format := "old"
	if isNew {
		format = "new"
	} else if !isOld {
		// 目录名不含特征时按第一个有效账号的内部结构判定
		if hasFeatureDir(filepath.Join(root, valid[0]), "db_storage") {
			format = "new"
		}
	}
	upd := ""
	if !latest.IsZero() {
		upd = latest.Format("2006-01-02 15:04")
	}
	return &WeChatDir{
		Path:         root,
		Name:         filepath.Base(root),
		Format:       format,
		Accounts:     len(valid),
		AccountsList: valid,
		JSONL:        findJSONLInDir(root, 3),
		Updated:      upd,
		Verified:     true,
	}
}

// isWeChatAccountDir 双重校验：目录名 + 内部特征文件
// 新版: wxid_xxx_序号 且含 db_storage；旧版: 名称不限（联系人名），含 Msg/FileStorage/Account
func isWeChatAccountDir(path, name string, isNew bool) bool {
	if isNonAccountName(name) {
		return false
	}
	if isNew {
		if !newAcctRe.MatchString(name) {
			return false
		}
		return hasFeatureDir(path, "db_storage")
	}
	// 旧版或格式未定：优先要求 wxid 名或任意名，但必须含特征目录
	if oldAcctRe.MatchString(name) || true { // 旧版允许联系人中文名，不强制 wxid
		for _, f := range []string{"Msg", "FileStorage", "Account"} {
			if hasFeatureDir(path, f) {
				return true
			}
		}
	}
	return false
}

func isNonAccountName(name string) bool {
	return nonAccountNames[strings.ToLower(name)]
}

func hasFeatureDir(path, name string) bool {
	fi, err := os.Stat(filepath.Join(path, name))
	return err == nil && fi.IsDir()
}

// ---------------- 目录内 JSONL 查找 ----------------

// 跳过的大数据/二进制目录（避免全盘递归拖慢扫描）
var skipJSONLDirs = map[string]bool{
	"db_storage": true, "msg": true, "cache": true, "temp": true,
	"resource": true, "filestorage": true, "emoticon": true,
	"head_image": true, "hardlink": true, "migrate": true, "sns": true,
	"solitaire": true, "third_app_icon": true, "apm_record": true,
	"business": true, "backup": true, "backupfiles": true, "applet": true,
	"wmpf": true, "all_users": true, "image": true, "video": true,
}

// findJSONLInDir 在目录内（≤depth 层）查找 JSONL 导出，跳过大数据目录
func findJSONLInDir(root string, depth int) []string {
	out := []string{}
	var walk func(dir string, d int)
	walk = func(dir string, d int) {
		if d > depth {
			return
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, e := range entries {
			full := filepath.Join(dir, e.Name())
			if e.IsDir() {
				if skipJSONLDirs[strings.ToLower(e.Name())] {
					continue
				}
				walk(full, d+1)
			} else if strings.HasSuffix(strings.ToLower(e.Name()), ".jsonl") {
				out = append(out, full)
			}
		}
	}
	walk(root, 0)
	return out
}

func runCmd(name string, args ...string) (string, error) {
	cmd := execCommand(name, args...)
	out, err := cmd.Output()
	return string(out), err
}
