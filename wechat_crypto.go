package main

// 微信加密聊天库自动解密导入
// 流程: 全盘扫描加密 db → 微信运行时从进程内存提取 SQLCipher 密钥候选 →
// 纯 Go 验证并解密（SQLCipher 3/4）→ 查询消息表 → 导出 JSONL → 合并进索引

import (
	"bufio"
	"bytes"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	_ "modernc.org/sqlite"
	"golang.org/x/sys/windows"
)

// EncryptedDB 一个微信加密数据库
type EncryptedDB struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	Format string `json:"format"` // "new"=微信4.0 xwechat_files / "old"=旧版 WeChat Files
	Acct   string `json:"acct"`   // 账号目录名
	Kind   string `json:"kind"`   // message / contact / biz / media / other
}

// scanEncryptedWeChatDBs 全盘扫描微信加密数据库（排除占位/临时文件）
func scanEncryptedWeChatDBs() []EncryptedDB {
	out := []EncryptedDB{}
	seen := map[string]bool{}
	roots := discoverWeChatRoots() // 复用全盘扫描
	for _, root := range roots {
		lower := strings.ToLower(root)
		isNew := strings.Contains(lower, "xwechat")
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() || isNonAccountName(e.Name()) {
				continue
			}
			acctPath := filepath.Join(root, e.Name())
			var scanDir string
			if isNew {
				scanDir = filepath.Join(acctPath, "db_storage")
			} else {
				scanDir = filepath.Join(acctPath, "Msg")
			}
			filepath.Walk(scanDir, func(p string, fi os.FileInfo, err error) error {
				if err != nil || fi.IsDir() {
					return nil
				}
				n := fi.Name()
				if !strings.HasSuffix(n, ".db") {
					return nil
				}
				// 排除: 非实体/占位/临时/元数据
				if strings.Contains(n, "-shm") || strings.Contains(n, "-wal") ||
					strings.Contains(n, "material") || strings.Contains(n, "-first") ||
					strings.Contains(n, "-incremental") || strings.Contains(n, "-last") ||
					strings.Contains(n, "kvdb") || n == "weclaw.db" {
					return nil
				}
				if fi.Size() < 1024 { // 1 字节占位
					return nil
				}
				if seen[p] {
					return nil
				}
				seen[p] = true
				out = append(out, EncryptedDB{
					Path:   p,
					Size:   fi.Size(),
					Format: map[bool]string{true: "new", false: "old"}[isNew],
					Acct:   e.Name(),
					Kind:   kindOfDB(n, isNew),
				})
				return nil
			})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return kindRank(out[i].Kind) < kindRank(out[j].Kind)
		}
		return out[i].Size > out[j].Size
	})
	return out
}

func kindRank(k string) int {
	switch k {
	case "message":
		return 0
	case "contact":
		return 1
	case "media":
		return 2
	case "biz":
		return 3
	default:
		return 9
	}
}

func kindOfDB(name string, isNew bool) string {
	ln := strings.ToLower(name)
	if strings.HasPrefix(ln, "message") {
		return "message"
	}
	if ln == "contact.db" || strings.HasPrefix(ln, "contact") {
		return "contact"
	}
	if strings.HasPrefix(ln, "media") {
		return "media"
	}
	if strings.HasPrefix(ln, "biz_message") || strings.HasPrefix(ln, "bizchat") {
		return "biz"
	}
	if strings.HasPrefix(ln, "msg") {
		return "message"
	}
	if ln == "micromsg.db" {
		return "contact"
	}
	return "other"
}

// ---------------- 微信进程内存密钥提取 ----------------

var wechatProcNames = []string{"Weixin.exe", "WeChat.exe", "WeChatApp.exe"}

func findWeChatProcess() (uint32, string) {
	snap, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return 0, ""
	}
	defer windows.CloseHandle(snap)
	var pe windows.ProcessEntry32
	pe.Size = uint32(unsafe.Sizeof(pe))
	for {
		if err := windows.Process32First(snap, &pe); err != nil {
			break
		}
		for {
			name := strings.TrimRight(windows.UTF16ToString(pe.ExeFile[:]), "\x00")
			for _, pn := range wechatProcNames {
				if strings.EqualFold(name, pn) {
					return pe.ProcessID, name
				}
			}
			if err := windows.Process32Next(snap, &pe); err != nil {
				return 0, ""
			}
		}
	}
	return 0, ""
}

// extractKeysFromProcess 流式读取进程内存提取密钥候选
// 分块 64MB 扫描（内存峰值 = 块大小 + 重叠区，避免整读数 GB 内存导致死机）
func extractKeysFromProcess(pid uint32, progress func(doneMB int)) ([]string, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION|windows.PROCESS_VM_READ, false, pid)
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(h)

	seen := map[string]bool{}
	var out []string
	add := func(s string) {
		s = strings.ToLower(s)
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	chunk := int64(1024 * 1024)     // 1MB/块（Windows 单次 ReadProcessMemory 大缓冲易失败，小块稳妥）
	overlap := 2 * 1024 * 1024      // 跨块重叠区（保证特征/hex 不被切断）
	var overflow []byte
	var totalMB int
	var lastReported int
	addr := uintptr(0)
	for {
		var mi windows.MemoryBasicInformation
		if err := windows.VirtualQueryEx(h, addr, &mi, uintptr(unsafe.Sizeof(mi))); err != nil {
			break
		}
		prot := mi.Protect & 0xFF
		readable := mi.State == windows.MEM_COMMIT &&
			(prot == windows.PAGE_READWRITE || prot == windows.PAGE_EXECUTE_READWRITE ||
				prot == windows.PAGE_WRITECOPY || prot == windows.PAGE_EXECUTE_WRITECOPY)
		if readable && mi.RegionSize > 0 {
			base := mi.BaseAddress
			remaining := int64(mi.RegionSize)
			for remaining > 0 {
				sz := remaining
				if sz > chunk {
					sz = chunk
				}
				buf := make([]byte, sz)
				var nread uintptr
				if err := windows.ReadProcessMemory(h, base, &buf[0], uintptr(len(buf)), &nread); err == nil && nread > 0 {
					data := buf[:nread]
					// 与上一块尾部拼接，避免跨块切断
					full := data
					if len(overflow) > 0 {
						full = append(append([]byte{}, overflow...), data...)
					}
					scanMemForKeys(full, add)
					// 保留尾部用于下一块
					if int64(len(data)) >= int64(overlap) {
						overflow = append([]byte{}, data[len(data)-overlap:]...)
					} else {
						overflow = append([]byte{}, data...)
					}
					totalMB += int(nread) / 1024 / 1024
					if progress != nil && totalMB-lastReported >= 128 {
						lastReported = totalMB
						progress(totalMB)
					}
				}
				base += uintptr(sz)
				remaining -= sz
			}
		}
		addr = mi.BaseAddress + mi.RegionSize
		if addr == 0 {
			break
		}
	}
	if progress != nil && totalMB-lastReported > 0 {
		progress(totalMB)
	}
	return out, nil
}

// scanMemForKeys 在内存块中提取密钥候选（多形态覆盖微信 3.x/4.0）：
// ① 特征字符串（xwechat_files/db_storage/WeChat Files）邻近窗口：
//    - 32 字节 hex（每窗口限 6，微信 4.0 key 的 hex 变体之一）
//    - 64 字符 hex 全量提取（微信 4.0/3.x key 常见形态）
//    - "i"+56/64 hex 前缀形态（社区报告的微信 4.0 存储变体）
// ② 全块 64 字符 hex 兜底（限 300，3.x 备用）
func scanMemForKeys(mem []byte, add func(string)) {
	const nearby = 1536
	hex64cnt := 0
	hex64Done := false
	hex64Pref := regexp.MustCompile(`(?i)i[0-9a-f]{64}`)
	hex56Pref := regexp.MustCompile(`(?i)i[0-9a-f]{56}`)
	for _, feat := range []string{"xwechat_files", "db_storage", "WeChat Files", "wechat_files"} {
		idx := 0
		for {
			i := bytes.Index(mem[idx:], []byte(feat))
			if i < 0 {
				break
			}
			start := idx + i
			lo := start - nearby
			if lo < 0 {
				lo = 0
			}
			hi := start + len(feat) + nearby
			if hi > len(mem) {
				hi = len(mem)
			}
			window := mem[lo:hi]
			// 32-hex（邻近优先，限量）
			got := 0
			for _, m := range hex32Re.FindAll(window, -1) {
				if len(m) == 32 {
					add(string(m))
					got++
					if got >= 6 {
						break
					}
				}
			}
			// 'i' 前缀 64/56-hex（去前缀后取 64/56 hex）
			for _, m := range hex64Pref.FindAll(window, -1) {
				if len(m) == 65 {
					add(string(m[1:]))
				}
			}
			for _, m := range hex56Pref.FindAll(window, -1) {
				if len(m) == 57 {
					add(string(m[1:]))
				}
			}
			// 64-hex（特征窗口内全量）
			for _, m := range hex64Re.FindAll(window, -1) {
				if len(m) == 64 {
					add(string(m))
					hex64cnt++
					if hex64cnt >= 300 {
						hex64Done = true
						break
					}
				}
			}
			idx = start + len(feat)
		}
	}
	if !hex64Done {
		for _, m := range hex64Re.FindAll(mem, -1) {
			if len(m) == 64 {
				add(string(m))
				hex64cnt++
				if hex64cnt >= 300 {
					break
				}
			}
		}
	}
}

var hex64Re = regexp.MustCompile(`(?i)[0-9a-f]{64}`)
var hex32Re = regexp.MustCompile(`(?i)[0-9a-f]{32}`)

// extractKeyCandidates 从进程内存提取 SQLCipher 密钥候选
// 微信 4.0: 32 字节 raw key，通常与 "xwechat_files"/"db_storage" 路径字符串邻近
// 微信 3.x: 64 字符 hex raw key
func extractKeyCandidates(mem []byte) []string {
	seen := map[string]bool{}
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
		}
	}
	// 1) 特征字符串邻近区域（4.0 主攻）
	for _, feat := range []string{"xwechat_files", "db_storage", "WeChat Files", "wechat_files"} {
		idx := 0
		for {
			i := strings.Index(string(mem[idx:]), feat)
			if i < 0 {
				break
			}
			start := idx + i
			lo, hi := start-4096, start+len(feat)+4096
			if lo < 0 {
				lo = 0
			}
			if hi > len(mem) {
				hi = len(mem)
			}
			window := mem[lo:hi]
			for _, m := range hex32Re.FindAll(window, -1) {
				if len(m) == 32 {
					add(string(m))
				}
			}
			for _, m := range hex64Re.FindAll(window, -1) {
				if len(m) == 64 {
					add(string(m))
				}
			}
			idx = start + len(feat)
		}
	}
	// 2) 全内存 64-hex（3.x 兜底，最多取 300 个）
	cnt := 0
	for _, m := range hex64Re.FindAll(mem, -1) {
		if len(m) == 64 {
			add(string(m))
			cnt++
			if cnt >= 300 {
				break
			}
		}
	}
	// 归一化小写
	out := []string{}
	for k := range seen {
		out = append(out, strings.ToLower(k))
	}
	sort.Strings(out)
	return out
}

// verifyKeyAgainstDBs 用候选密钥解密消息库第一页验证
// 微信同账号所有库使用同一密钥：仅验证第一个消息库即可（避免逐候选遍历 20+ 库的 I/O 与计算开销）
func verifyKeyAgainstDBs(key string, dbs []EncryptedDB) bool {
	for _, db := range dbs {
		if db.Kind == "message" {
			if _, _, ok := tryDecryptFirstPage(db.Path, key); ok {
				return true
			}
			return false
		}
	}
	return false
}

// verifyKeysParallel 并行验证候选密钥（CPU 密集 PBKDF2，并行显著提速；命中即停）
func verifyKeysParallel(cands []string, dbs []EncryptedDB, progress func(done, total int)) string {
	total := len(cands)
	if total == 0 {
		return ""
	}
	// 去重
	seen := map[string]bool{}
	uniq := []string{}
	for _, c := range cands {
		if !seen[c] {
			seen[c] = true
			uniq = append(uniq, c)
		}
	}
	total = len(uniq)
	const workers = 8
	results := make(chan string, 1)
	done := make(chan struct{})
	verified := int64(0)

	go func() {
		idx := 0
		var mu sync.Mutex
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					mu.Lock()
					if idx >= len(uniq) {
						mu.Unlock()
						return
					}
					c := uniq[idx]
					idx++
					mu.Unlock()
					if verifyKeyAgainstDBs(c, dbs) {
						select {
						case results <- c:
						default:
						}
						return
					}
					if progress != nil {
						n := atomic.AddInt64(&verified, 1)
						progress(int(n), total)
					}
				}
			}()
		}
		wg.Wait()
		close(done)
	}()
	select {
	case k := <-results:
		return k
	case <-done:
		return ""
	}
}

// ---------------- 消息提取（解密后明文库） ----------------

// msgTypeName 微信 type 数字 → 中文类型（与现有索引 type 字段对齐）
func msgTypeName(t int) string {
	switch t {
	case 1:
		return "文本"
	case 3:
		return "图片"
	case 34:
		return "语音"
	case 42:
		return "名片"
	case 43:
		return "视频"
	case 47:
		return "动画表情"
	case 48:
		return "位置"
	case 49:
		return "文件/链接"
	case 50:
		return "语音通话"
	case 10000:
		return "系统消息"
	case 10002:
		return "系统消息"
	case 1048625:
		return "拍一拍"
	default:
		return "其他"
	}
}

// extractMsgsFromDecrypted 从已解密明文库提取消息 → JSONL 行
// 兼容微信 4.0（message 表: talkerId/isSend/createTime/type/content）与 3.x（Msg 表: strTalker/isSend/createTime/type/strContent）
func extractMsgsFromDecrypted(dbPath string, contacts map[string]string) (int64, error) {
	conn, err := sql.Open("sqlite", dbPath+"?_pragma=busy_timeout(5000)")
	if err != nil {
		return 0, err
	}
	defer conn.Close()
	// 探测表
	rows, err := conn.Query("SELECT name FROM sqlite_master WHERE type='table'")
	if err != nil {
		return 0, err
	}
	tables := []string{}
	for rows.Next() {
		var t string
		rows.Scan(&t)
		tables = append(tables, t)
	}
	rows.Close()
	// 找消息表：列含 createTime + content
	msgTable, cols := "", []string{}
	for _, t := range tables {
		if !strings.Contains(strings.ToLower(t), "message") && !strings.EqualFold(t, "msg") && t != "Msg" {
			continue
		}
		crows, err := conn.Query("PRAGMA table_info(" + t + ")")
		if err != nil {
			continue
		}
		cs := []string{}
		for crows.Next() {
			var cid int
			var cname, ctype string
			var notnull, pk int
			var dflt interface{}
			if crows.Scan(&cid, &cname, &ctype, &notnull, &dflt, &pk) == nil {
				cs = append(cs, cname)
			}
		}
		crows.Close()
		lc := strings.ToLower(strings.Join(cs, ","))
		if strings.Contains(lc, "createtime") && strings.Contains(lc, "content") {
			msgTable, cols = t, cs
			break
		}
	}
	if msgTable == "" {
		return 0, nil // 无消息表
	}
	colLower := map[string]bool{}
	for _, c := range cols {
		colLower[strings.ToLower(c)] = true
	}
	// 列选择
	nameCol := ""
	if colLower["strtalker"] {
		nameCol = "strTalker"
	} else if colLower["talkerid"] {
		nameCol = "talkerId"
	}
	isSendCol, tsCol, typeCol, contentCol := "", "", "", ""
	if colLower["issend"] {
		isSendCol = "isSend"
	}
	if colLower["createtime"] {
		tsCol = "createTime"
	}
	if colLower["type"] {
		typeCol = "type"
	}
	if colLower["strcontent"] {
		contentCol = "strContent"
	} else if colLower["content"] {
		contentCol = "content"
	}
	if nameCol == "" || tsCol == "" || contentCol == "" {
		return 0, nil
	}
	sel := "SELECT " + nameCol
	if isSendCol != "" {
		sel += "," + isSendCol
	} else {
		sel += ",0"
	}
	sel += "," + tsCol
	if typeCol != "" {
		sel += "," + typeCol
	} else {
		sel += ",1"
	}
	sel += "," + contentCol + " FROM " + msgTable
	if colLower["isdeleted"] {
		sel += " WHERE isDeleted=0"
	}
	q, err := conn.Query(sel)
	if err != nil {
		return 0, err
	}
	defer q.Close()

	out := filepath.Join(dataDir, "wechat_decrypt.jsonl")
	f, err := os.OpenFile(out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1024*1024)
	var n int64
	for q.Next() {
		var talker string
		var isSend int
		var ts int64
		var typ int
		var content string
		if err := q.Scan(&talker, &isSend, &ts, &typ, &content); err != nil {
			continue
		}
		if content == "" && typ != 3 {
			continue
		}
		// ts 单位：毫秒→秒
		if ts > 1e12 {
			ts /= 1000
		}
		nick := contacts[talker]
		if nick == "" {
			nick = talker
		}
		rec := map[string]interface{}{
			"name":    nick,
			"user":    talker,
			"is_self": isSend == 1,
			"ts":      ts,
			"type":    msgTypeName(typ),
			"content": content,
		}
		b, _ := json.Marshal(rec)
		w.Write(b)
		w.WriteByte('\n')
		n++
		if n%20000 == 0 {
			w.Flush()
		}
	}
	w.Flush()
	return n, nil
}

// extractContactsFromDecrypted 从已解密 contact 库提取 {talkerId: 昵称} 映射（4.0 contact 表 / 3.x Contact 表）
func extractContactsFromDecrypted(dbPath string) (map[string]string, error) {
	out := map[string]string{}
	conn, err := sql.Open("sqlite", dbPath+"?_pragma=busy_timeout(5000)")
	if err != nil {
		return out, err
	}
	defer conn.Close()
	// 探测表
	rows, err := conn.Query("SELECT name FROM sqlite_master WHERE type='table'")
	if err != nil {
		return out, err
	}
	tables := []string{}
	for rows.Next() {
		var t string
		rows.Scan(&t)
		tables = append(tables, t)
	}
	rows.Close()
	sel := ""
	for _, t := range tables {
		if !strings.Contains(strings.ToLower(t), "contact") {
			continue
		}
		crows, err := conn.Query("PRAGMA table_info(" + t + ")")
		if err != nil {
			continue
		}
		cs := []string{}
		for crows.Next() {
			var cid int
			var cname, ctype string
			var notnull, pk int
			var dflt interface{}
			if crows.Scan(&cid, &cname, &ctype, &notnull, &dflt, &pk) == nil {
				cs = append(cs, cname)
			}
		}
		crows.Close()
		lc := strings.ToLower(strings.Join(cs, ","))
		if strings.Contains(lc, "talkerid") && (strings.Contains(lc, "nickname") || strings.Contains(lc, "nickname") || strings.Contains(lc, "nick")) {
			sel = "SELECT talkerId,nickname FROM " + t
			break
		}
		// 3.x Contact 表: UserName/NickName
		if strings.Contains(lc, "username") && strings.Contains(lc, "nickname") {
			sel = "SELECT UserName,NickName FROM " + t
			break
		}
	}
	if sel == "" {
		return out, nil
	}
	q, err := conn.Query(sel)
	if err != nil {
		return out, err
	}
	defer q.Close()
	for q.Next() {
		var id, nick string
		if q.Scan(&id, &nick) == nil && id != "" && nick != "" {
			out[id] = nick
		}
	}
	return out, nil
}

// ---------------- 解密导入任务编排 ----------------

type DecryptJob struct {
	mu       sync.Mutex
	Running  bool   `json:"running"`
	Stage    string `json:"stage"`
	Progress int    `json:"progress"` // 0-100
	Message  string `json:"message"`
	Key      string `json:"key_found"`
	DBs      int    `json:"dbs"`
	Added    int64  `json:"added"`
}

var decryptJob = &DecryptJob{}

func decryptJobState() map[string]interface{} {
	decryptJob.mu.Lock()
	defer decryptJob.mu.Unlock()
	return map[string]interface{}{
		"running":  decryptJob.Running,
		"stage":    decryptJob.Stage,
		"progress": decryptJob.Progress,
		"message":  decryptJob.Message,
		"key_found": decryptJob.Key != "",
		"dbs":      decryptJob.DBs,
		"added":    decryptJob.Added,
	}
}

func setDecrypt(stage string, progress int, msg string) {
	decryptJob.mu.Lock()
	decryptJob.Stage = stage
	decryptJob.Progress = progress
	if msg != "" {
		decryptJob.Message = msg
	}
	decryptJob.mu.Unlock()
}

// StartWeChatDecrypt 执行完整解密导入流程（后台 goroutine）
func StartWeChatDecrypt() error {
	decryptJob.mu.Lock()
	if decryptJob.Running {
		decryptJob.mu.Unlock()
		return errors.New("解密任务已在运行")
	}
	decryptJob.Running = true
	decryptJob.Key = ""
	decryptJob.Added = 0
	decryptJob.mu.Unlock()

	go func() {
		defer func() {
			decryptJob.mu.Lock()
			decryptJob.Running = false
			decryptJob.mu.Unlock()
		}()
		runDecryptFlow()
	}()
	return nil
}

func runDecryptFlow() {
	setDecrypt("扫描微信加密数据库", 5, "正在全盘扫描…")
	dbs := scanEncryptedWeChatDBs()
	msgs := []EncryptedDB{}
	contacts := []EncryptedDB{}
	for _, db := range dbs {
		if db.Kind == "message" {
			msgs = append(msgs, db)
		} else if db.Kind == "contact" {
			contacts = append(contacts, db)
		}
	}
	decryptJob.mu.Lock()
	decryptJob.DBs = len(msgs)
	decryptJob.mu.Unlock()
	setDecrypt("扫描完成", 8, fmt.Sprintf("发现 %d 个消息库、%d 个联系人库（微信 4.0 加密格式）", len(msgs), len(contacts)))
	if len(msgs) == 0 {
		setDecrypt("无消息库", 100, "未发现加密消息库。请确认微信已在本机登录并保存过聊天记录")
		return
	}

	// 2) 检查微信进程（解密引擎需要微信运行中提取密钥）
	pid, pname := findWeChatProcess()
	if pid == 0 {
		setDecrypt("未检测到微信进程", 100, "未找到微信进程（Weixin.exe / WeChat.exe）。请先登录微信再重试")
		return
	}

	// 3) 确保 Python 解密环境（无则自动配置：创建 venv + 安装 wechatauto-replica）
	py, err := ensureDecryptEnv(func(stage string, pct int, msg string) {
		setDecrypt(stage, 10+pct/10, msg)
	})
	if err != nil {
		setDecrypt("环境配置失败", 100, err.Error())
		return
	}
	script := filepath.Join(wd, "scripts", "decrypt_wechat.py")
	jsonl := filepath.Join(dataDir, "wechat_decrypt.jsonl")
	os.Remove(jsonl)
	setDecrypt("提取微信密钥", 15, fmt.Sprintf("调用解密引擎（%s PID %d），从进程提取密钥并解密全部消息库…", pname, pid))
	cmd := exec.Command(py, script, "--out", jsonl)
	cmd.Dir = wd
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		setDecrypt("解密引擎启动失败", 100, "无法启动解密引擎："+err.Error())
		return
	}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Start(); err != nil {
		setDecrypt("解密引擎启动失败", 100, "无法启动解密引擎："+err.Error())
		return
	}
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 1024*1024), 4*1024*1024)
	var total int64
	keyFound := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "[PROGRESS]") {
			body := strings.TrimPrefix(line, "[PROGRESS]")
			fields := parseKV(body)
			switch fields["stage"] {
			case "keys":
				if fields["status"] == "ok" {
					keyFound = "yes"
					setDecrypt("密钥提取成功", 20, fmt.Sprintf("账号 %s 全部 %s 个库密钥已就绪", fields["account"], fields["keys"]))
				}
			case "sender_index":
				setDecrypt("读取发送者索引", 25, "正在读取发送者映射（message_resource.db）…")
			case "nickname":
				setDecrypt("读取联系人昵称", 28, "正在读取联系人昵称映射…")
			default:
				if dbName := fields["db"]; dbName != "" {
					setDecrypt("解密消息库", 30+int(total*40/10000000), fmt.Sprintf("正在解密 %s…（已导出 %d 条）", dbName, total))
				}
			}
		} else if strings.HasPrefix(line, "[DONE]") {
			total = parseTotal(line)
			setDecrypt("导出完成", 90, fmt.Sprintf("解密导出完成，共 %d 条消息", total))
		} else if strings.HasPrefix(line, "[ERROR]") {
			setDecrypt("解密失败", 100, strings.TrimSpace(strings.TrimPrefix(line, "[ERROR]")))
			return
		}
	}
	cmd.Wait()
	if keyFound == "" {
		setDecrypt("密钥提取失败", 100, "未能从微信进程提取到数据库密钥。请确认微信已登录、窗口保持打开，且本程序权限不低于微信")
		return
	}
	if total == 0 {
		st, _ := os.Stat(jsonl)
		if st == nil || st.Size() == 0 {
			setDecrypt("未提取到消息", 100, "解密完成但未导出消息记录（请检查微信版本兼容性）")
			return
		}
	}

	// 4) 合并进索引
	setDecrypt("合并去重重建索引", 95, fmt.Sprintf("新增 %d 条解密记录，正在合并进索引…", total))
	if st, err := os.Stat(jsonl); err == nil && st.Size() > 0 {
		paths := []string{}
		for _, s := range SourceInfos() {
			paths = append(paths, s.Path)
		}
		GDB.EnsureIndexMulti(append(paths, jsonl), nil)
	}
	decryptJob.mu.Lock()
	decryptJob.Added = total
	decryptJob.Key = "yes"
	decryptJob.mu.Unlock()
	setDecrypt("完成", 100, fmt.Sprintf("解密导入完成：新增 %d 条聊天记录，已合并进索引用于全部分析", total))
}

// parseKV 解析 "[PROGRESS] k=v k=v" 形式的字段
func parseKV(s string) map[string]string {
	out := map[string]string{}
	for _, f := range strings.Fields(s) {
		if i := strings.Index(f, "="); i > 0 {
			out[f[:i]] = f[i+1:]
		}
	}
	return out
}

func parseTotal(line string) int64 {
	line = strings.TrimPrefix(line, "[DONE]")
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "total=")
	var n int64
	fmt.Sscanf(line, "%d", &n)
	return n
}

// ensureDecryptEnv 确保解密用 Python 环境可用
// 优先级：① 自带 runtime（打包的 Python 3.11，零依赖零网络，推荐）→ ② 已有 venv → ③ 系统 Python 创建 venv + 离线 wheels → ④ 国内镜像
func ensureDecryptEnv(progress func(stage string, pct int, msg string)) (string, error) {
	// 0) 自带嵌入式 Python 运行时（runtime\python.exe，依赖已预装）
	rt := filepath.Join(wd, "runtime", "python.exe")
	if fileExists(rt) && pyHasWechatAuto(rt) {
		progress("解密环境就绪", 90, "使用内置 Python 3.11 运行时（无需安装）")
		return rt, nil
	}
	// 1) 已有 venv 直接用
	if p := os.Getenv("WECHAT_TWIN_PYTHON"); p != "" && fileExists(p) {
		return p, nil
	}
	cands := []string{
		filepath.Join(wd, "..", "ai-twin", "venv", "Scripts", "python.exe"),
		filepath.Join(wd, "..", "ai-twin", "venv", "python.exe"),
		filepath.Join(wd, "venv", "Scripts", "python.exe"),
	}
	for _, c := range cands {
		abs, _ := filepath.Abs(c)
		if fileExists(abs) && pyHasWechatAuto(abs) {
			return abs, nil
		}
	}

	// 2) 自动创建 venv（优先 py -3.11 以匹配离线 wheels，其次 py -3 / python）
	progress("配置 Python 环境", 0, "未找到可用 Python 环境，正在自动创建 venv…")
	venvDir := filepath.Join(wd, "venv")
	venvPy := filepath.Join(venvDir, "Scripts", "python.exe")
	sysPy := ""
	for _, probe := range []string{"py -3.11", "py -3", "python"} {
		parts := strings.Split(probe, " ")
		cmd := exec.Command(parts[0], append(parts[1:], "--version")...)
		if out, err := cmd.CombinedOutput(); err == nil && strings.Contains(strings.ToLower(string(out)), "python") {
			sysPy = probe
			break
		}
	}
	if sysPy == "" {
		return "", errors.New("未检测到系统 Python，且内置运行时缺失。请重新下载包含 runtime\\ 目录的完整安装包，或安装 Python 3.11（勾选 Add to PATH）")
	}
	parts := strings.Split(sysPy, " ")
	cmd := exec.Command(parts[0], append(parts[1:], "-m", "venv", venvDir)...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", errors.New("创建 venv 失败：" + err.Error())
	}
	if !fileExists(venvPy) {
		return "", errors.New("venv 创建后未找到 python.exe（请检查系统 Python 完整性）")
	}

	// 3) 安装/校验 wechatauto-replica（离线 wheels 优先 → 国内镜像 → 官方源）
	if !pyHasWechatAuto(venvPy) {
		progress("安装解密依赖", 50, "正在安装 wechatauto-replica（微信 4.0 解密引擎），优先使用本地离线包…")
		installed := installWechatAutoOffline(venvPy, progress)
		if !installed {
			for _, mirror := range pipMirrors() {
				progress("安装解密依赖", 55, "本地离线包不可用（Python 版本非 3.11？），尝试镜像 "+mirror+" …")
				if pipInstall(venvPy, mirror, "wechatauto-replica", "cryptography") && pyHasWechatAuto(venvPy) {
					installed = true
					break
				}
			}
		}
		if !installed {
			return "", errors.New("自动安装解密依赖失败（网络受限或镜像不可达）。请使用包含 runtime\\ 目录的完整安装包（内置 Python，无需安装依赖），或手动执行：<venv>\\Scripts\\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wechatauto-replica cryptography")
		}
	}
	return venvPy, nil
}

// installWechatAutoOffline 从 wheels/ 离线安装解密依赖（不依赖网络）
// 安装过程写入 data\env_setup.log，方便排查
func installWechatAutoOffline(venvPy string, progress func(stage string, pct int, msg string)) bool {
	wheelsDir := filepath.Join(wd, "wheels")
	if !dirExists(wheelsDir) {
		return false
	}
	logF := filepath.Join(dataDir, "env_setup.log")
	// 先装 wheel（sdist 构建需要 bdist_wheel 命令）
	progress("安装解密依赖", 45, "离线安装 wheel 构建工具…")
	if !pipOffline(venvPy, wheelsDir, logF, "wheel") {
		return false
	}
	// 装解密引擎 + 运行时依赖（--no-deps 显式列出，避免拉取重型可选依赖 opencv/numpy）
	pkgs := []string{"wechatauto_replica", "cryptography", "cffi", "pycparser", "pywin32",
		"winsdk", "uiautomation", "zstandard", "comtypes", "colorama", "psutil", "pyperclip",
		"pymsgbox", "pyautogui", "pyscreeze", "pygetwindow", "pyrect", "pytweening", "mouseinfo", "pillow"}
	args := []string{"-m", "pip", "install", "--quiet", "--disable-pip-version-check",
		"--no-index", "--find-links", wheelsDir, "--no-build-isolation", "--no-deps", "--timeout", "120"}
	args = append(args, pkgs...)
	c := exec.Command(venvPy, args...)
	if f, err := os.OpenFile(logF, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
		c.Stdout = f
		c.Stderr = f
		defer f.Close()
	}
	progress("安装解密依赖", 55, "正在离线安装解密引擎（wheels/ 本地包）…")
	ok := c.Run() == nil && pyHasWechatAuto(venvPy)
	if !ok {
		progress("安装解密依赖", 55, "离线安装失败，已写入日志 data\\env_setup.log，尝试镜像…")
	}
	return ok
}

// pipOffline 离线安装指定包（输出写 log）
func pipOffline(venvPy, wheelsDir, logF, pkg string) bool {
	c := exec.Command(venvPy, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
		"--no-index", "--find-links", wheelsDir, "--timeout", "120", pkg)
	if f, err := os.OpenFile(logF, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
		c.Stdout = f
		c.Stderr = f
		defer f.Close()
	}
	return c.Run() == nil
}

// pipInstall 从指定镜像安装包（-i mirror，空串表示官方源；输出写 log）
func pipInstall(venvPy, mirror string, pkgs ...string) bool {
	args := []string{"-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--timeout", "120"}
	if mirror != "" {
		args = append(args, "-i", mirror)
	}
	args = append(args, pkgs...)
	c := exec.Command(venvPy, args...)
	if f, err := os.OpenFile(filepath.Join(dataDir, "env_setup.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
		c.Stdout = f
		c.Stderr = f
		defer f.Close()
	}
	return c.Run() == nil
}

// pipMirrors 读取 pip-mirrors.txt 镜像列表（读不到时回退内置列表）
func pipMirrors() []string {
	lines, err := os.ReadFile(filepath.Join(wd, "scripts", "pip-mirrors.txt"))
	if err == nil {
		var out []string
		for _, l := range strings.Split(string(lines), "\n") {
			l = strings.TrimSpace(l)
			if l != "" && !strings.HasPrefix(l, "#") {
				out = append(out, l)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return []string{
		"https://pypi.tuna.tsinghua.edu.cn/simple",
		"https://mirrors.aliyun.com/pypi/simple/",
		"https://mirrors.cloud.tencent.com/pypi/simple/",
		"https://mirrors.huaweicloud.com/repository/pypi/simple/",
		"https://pypi.org/simple",
	}
}

// pyHasWechatAuto 检查指定 python 能否 import wechatauto
func pyHasWechatAuto(py string) bool {
	c := exec.Command(py, "-c", "import wechatauto")
	return c.Run() == nil
}

func dirExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}
