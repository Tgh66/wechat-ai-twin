package main

// 微信加密聊天库自动解密导入
// 流程: 全盘扫描加密 db → 微信运行时从进程内存提取 SQLCipher 密钥候选 →
// 纯 Go 验证并解密（SQLCipher 3/4）→ 查询消息表 → 导出 JSONL → 合并进索引

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
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

// readProcMemory 读取进程所有可读内存（分块），返回原始字节
func readProcMemory(pid uint32, progress func(doneMB int)) ([]byte, error) {
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION|windows.PROCESS_VM_READ, false, pid)
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(h)
	var buf []byte
	var addr uintptr
	var totalMB int
	for {
		var mi windows.MemoryBasicInformation
		err := windows.VirtualQueryEx(h, addr, &mi, uintptr(unsafe.Sizeof(mi)))
		if err != nil {
			break
		}
		if mi.State == windows.MEM_COMMIT && mi.Protect == windows.PAGE_READWRITE && mi.RegionSize > 0 {
			chunk := make([]byte, mi.RegionSize)
			var read uintptr
			if windows.ReadProcessMemory(h, mi.BaseAddress, &chunk[0], uintptr(len(chunk)), &read) == nil && read > 0 {
				buf = append(buf, chunk[:read]...)
				totalMB += int(read) / 1024 / 1024
				if progress != nil && totalMB%512 == 0 {
					progress(totalMB)
				}
			}
		}
		addr = mi.BaseAddress + mi.RegionSize
		if addr == 0 {
			break
		}
	}
	return buf, nil
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

// verifyKeyAgainstDBs 用候选密钥解密任一 message 库第一页验证
func verifyKeyAgainstDBs(key string, dbs []EncryptedDB) bool {
	for _, db := range dbs {
		if db.Kind == "message" {
			if _, _, ok := tryDecryptFirstPage(db.Path, key); ok {
				return true
			}
		}
	}
	return false
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

	// 2) 提取密钥（需微信运行）
	setDecrypt("提取微信密钥", 12, "请保持微信运行中（密钥需从微信进程内存提取）…")
	pid, pname := findWeChatProcess()
	if pid == 0 {
		setDecrypt("未检测到微信进程", 100, "未找到微信进程（Weixin.exe / WeChat.exe）。请先登录微信再重试")
		return
	}
	setDecrypt("读取微信进程内存", 15, fmt.Sprintf("进程: %s (PID %d)，正在扫描内存寻找密钥…", pname, pid))
	mem, err := readProcMemory(pid, func(doneMB int) {
		setDecrypt("读取内存", 15+doneMB/100, fmt.Sprintf("已读取 %d MB 内存…", doneMB))
	})
	if err != nil || len(mem) == 0 {
		setDecrypt("内存读取失败", 100, "无法读取微信进程内存："+fmt.Sprintf("%v", err))
		return
	}
	cands := extractKeyCandidates(mem)
	mem = nil // 释放大内存
	setDecrypt("验证密钥", 20, fmt.Sprintf("提取到 %d 个候选密钥，正在验证…", len(cands)))
	var key string
	verified := 0
	for _, c := range cands {
		if verifyKeyAgainstDBs(c, msgs) {
			key = c
			break
		}
		verified++
		if verified%10 == 0 {
			setDecrypt("验证密钥", 20+verified*2, fmt.Sprintf("已尝试 %d/%d 个候选…", verified, len(cands)))
		}
	}
	if key == "" {
		setDecrypt("密钥验证失败", 100, "未能从微信进程提取到有效密钥（微信版本差异或权限不足）。可尝试：以管理员身份运行本程序后重试")
		return
	}
	decryptJob.mu.Lock()
	decryptJob.Key = key
	decryptJob.mu.Unlock()
	setDecrypt("密钥已确认", 25, "密钥提取成功，开始解密…")

	// 3) 解密 contact（昵称映射）
	contactsMap := map[string]string{}
	for i, cd := range contacts {
		plain := filepath.Join(dataDir, "wx_decrypted", safeName(cd.Acct)+"_contact.db")
		os.MkdirAll(filepath.Dir(plain), 0755)
		if _, err := decryptSQLCipherDB(cd.Path, key, plain); err == nil {
			if m, err := extractContactsFromDecrypted(plain); err == nil && len(m) > 0 {
				for k, v := range m {
					contactsMap[k] = v
				}
			}
		}
		setDecrypt("解密联系人库", 30+i*3, fmt.Sprintf("%d/%d 联系人库", i+1, len(contacts)))
	}

	// 4) 解密消息库 → JSONL
	os.Remove(filepath.Join(dataDir, "wechat_decrypt.jsonl"))
	var total int64
	for i, md := range msgs {
		plain := filepath.Join(dataDir, "wx_decrypted", safeName(md.Acct)+"_msg_"+fmt.Sprintf("%d", i)+".db")
		os.MkdirAll(filepath.Dir(plain), 0755)
		setDecrypt("解密消息库", 40+(i*50)/len(msgs), fmt.Sprintf("解密 %s (%.0f MB)…", filepath.Base(md.Path), float64(md.Size)/1e6))
		if _, err := decryptSQLCipherDB(md.Path, key, plain); err != nil {
			continue
		}
		n, err := extractMsgsFromDecrypted(plain, contactsMap)
		if err == nil {
			total += n
			setDecrypt("提取消息", 40+((i+1)*50)/len(msgs), fmt.Sprintf("库 %d/%d 已提取 %d 条（累计 %d）", i+1, len(msgs), n, total))
		}
	}
	// 5) 合并进索引
	if total == 0 {
		setDecrypt("未提取到消息", 100, "解密成功但未提取到消息记录（请检查微信版本兼容性）")
		return
	}
	setDecrypt("合并去重重建索引", 95, fmt.Sprintf("新增 %d 条解密记录，正在合并进索引…", total))
	jsonl := filepath.Join(dataDir, "wechat_decrypt.jsonl")
	if st, err := os.Stat(jsonl); err == nil && st.Size() > 0 {
		paths := []string{}
		for _, s := range SourceInfos() {
			paths = append(paths, s.Path)
		}
		GDB.EnsureIndexMulti(append(paths, jsonl), nil)
	}
	decryptJob.mu.Lock()
	decryptJob.Added = total
	decryptJob.mu.Unlock()
	setDecrypt("完成", 100, fmt.Sprintf("解密导入完成：新增 %d 条聊天记录，已在「数据」页看到并用于全部分析", total))
}
