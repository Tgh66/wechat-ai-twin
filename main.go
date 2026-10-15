// 微信 AI 数字分身 · Go 重构版
// 本地聊天记录分析 + 情感引擎 + 数字分身入口（训练/推理仍由 Python 模块提供）
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------- 全局 ----------------

var (
	dbPath     string // SQLite 索引路径
	dataDir    string // 数据目录
	modelDir   string // 模型目录
	port       = "7860"
	GDB        *DB
	browserPID = 0
)

func init() {
	// 以 exe 所在目录为根（兼容双击运行）；--data/--port 可覆盖（见 main）
	wd = exeDir()
	dataDir = filepath.Join(wd, "data")
	modelDir = filepath.Join(wd, "models")
	_ = os.MkdirAll(dataDir, 0755)
	dbPath = filepath.Join(dataDir, "chats.db")
}

var wd string

func exeDir() string {
	if p, err := os.Executable(); err == nil {
		return filepath.Dir(p)
	}
	if p, err := os.Getwd(); err == nil {
		return p
	}
	return "."
}

// ---------------- 入口 ----------------

func main() {
	noBrowser := flag.Bool("nobrowser", false, "不自动打开浏览器")
	dataFlag := flag.String("data", "", "数据目录（默认 exe 同级 data/，用于多份部署隔离）")
	portFlag := flag.String("port", "", "HTTP 端口（默认 7860）")
	flag.Parse()

	if *dataFlag != "" {
		dataDir = *dataFlag
		dbPath = filepath.Join(dataDir, "chats.db")
		_ = os.MkdirAll(dataDir, 0755)
	}
	if *portFlag != "" {
		port = *portFlag
	}

	log.SetOutput(os.Stdout)
	fmt.Println(strings.Repeat("=", 54))
	fmt.Println("微信 AI 数字分身工作台 (Go 版)")
	fmt.Println("服务地址: http://127.0.0.1:" + port)
	fmt.Println("数据目录:", dataDir)
	fmt.Println("模型目录:", modelDir)
	fmt.Println(strings.Repeat("=", 54))

	// 打开/创建 SQLite 索引
	var err error
	GDB, err = OpenDB(dbPath)
	if err != nil {
		log.Fatalf("打开数据库失败: %v", err)
	}
	defer GDB.Close()

	// 全文索引就绪（FTS5 分批后台回填，不阻塞启动/API）
	go func() {
		for GDB.ensureFTSBatch() {
			time.Sleep(80 * time.Millisecond)
		}
	}()

	// 自动发现数据源：优先 data/*.jsonl，其次自动搜索微信目录
	go autoLoadData()

	// HTTP 服务（端口占用给出明确提示）
	ln, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		log.Fatalf("端口 %s 被占用：可能已有 WeChatTwin 实例在运行（或换 --port 指定其他端口）。错误: %v", port, err)
	}
	mux := http.NewServeMux()
	// 静态页面
	mux.Handle("/", http.FileServer(http.Dir(filepath.Join(wd, "web"))))
	// API
	mux.HandleFunc("/api/", apiHandler)

	server := &http.Server{Handler: mux}

	if !*noBrowser {
		go openBrowserAndWatch("http://127.0.0.1:"+port, func() {
			log.Println("浏览器已关闭，服务停止")
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			_ = server.Shutdown(ctx)
			os.Exit(0)
		})
	}

	log.Println("服务已启动，按 Ctrl+C 停止")
	if err := server.Serve(ln); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

// ---------------- 浏览器管理 ----------------

// openBrowserAndWatch 打开默认浏览器；检测到浏览器新进程退出后调用 onExit
func openBrowserAndWatch(url string, onExit func()) {
	// 启动前快照浏览器进程
	before := browserProcesses()
	launchBrowser(url)
	time.Sleep(3 * time.Second)
	after := browserProcesses()
	newPIDs := []int{}
	for _, p := range after {
		if !contains(before, p) {
			newPIDs = append(newPIDs, p)
		}
	}
	if len(newPIDs) == 0 {
		// 复用已有浏览器进程（无新进程），无法精确跟踪：轮询该浏览器是否整体退出
		for {
			cur := browserProcesses()
			alive := 0
			for _, p := range cur {
				if contains(before, p) {
					alive++
				}
			}
			if alive == 0 {
				onExit()
				return
			}
			time.Sleep(5 * time.Second)
		}
	}
	// 跟踪新打开的浏览器进程
	for {
		alive := false
		for _, pid := range newPIDs {
			if processAlive(pid) {
				alive = true
				break
			}
		}
		if !alive {
			onExit()
			return
		}
		time.Sleep(5 * time.Second)
	}
}

func launchBrowser(url string) {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "start", "", url)
	} else {
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start()
}

var procRe = regexp.MustCompile(`\s+(\d+)\s+chrome\.exe|\s+(\d+)\s+msedge\.exe|\s+(\d+)\s+firefox\.exe|\s+(\d+)\s+360se\.exe|\s+(\d+)\s+QQBrowser\.exe|\s+(\d+)\s+opera\.exe|\s+(\d+)\s+brave\.exe`)

// browserProcesses 返回当前常见浏览器进程 PID
func browserProcesses() []int {
	out, err := exec.Command("tasklist", "/FI", "IMAGENAME eq chrome.exe").Output()
	ids := []int{}
	if err == nil {
		ids = append(ids, parsePIDs(string(out))...)
	}
	for _, name := range []string{"msedge.exe", "firefox.exe", "360se.exe", "QQBrowser.exe", "opera.exe", "brave.exe", "iexplore.exe"} {
		if o, err := exec.Command("tasklist", "/FI", "IMAGENAME eq "+name).Output(); err == nil {
			ids = append(ids, parsePIDs(string(o))...)
		}
	}
	return ids
}

func parsePIDs(out string) []int {
	ids := []int{}
	for _, line := range strings.Split(out, "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			if pid, err := strconv.Atoi(fields[1]); err == nil {
				ids = append(ids, pid)
			}
		}
	}
	return ids
}

func processAlive(pid int) bool {
	out, err := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid)).Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), strconv.Itoa(pid))
}

func contains(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func execCommand(name string, args ...string) *exec.Cmd {
	return exec.Command(name, args...)
}

// ---------------- 数据加载 ----------------

var loadOnce sync.Once

func autoLoadData() {
	loadOnce.Do(func() {
		// 1) 多源合并：data/ 与 ai-twin/data/ 下全部 JSONL 合并去重导入
		srcs := SourceInfos()
		if len(srcs) > 0 {
			files := []string{}
			var gb float64
			for _, s := range srcs {
				files = append(files, s.Path)
				gb += float64(s.Size) / 1e9
			}
			log.Printf("发现 %d 个聊天记录数据源（合计约 %.2f GB），合并去重导入…", len(srcs), gb)
			GDB.EnsureIndexMulti(files, func(done int64, total int64) {
				if total > 0 && done%500000 == 0 {
					log.Printf("导入进度: %d / %d", done, total)
				}
			})
			return
		}
		// 2) 无 JSONL 时自动搜索微信目录（注册表/默认/各盘 xwechat_files）
		log.Println("未发现 JSONL 导出，自动搜索微信目录…")
		res := DiscoverWeChatDirs()
		if dirs, ok := res["dirs"].([]WeChatDir); ok && len(dirs) > 0 {
			log.Printf("发现 %d 个微信数据目录，首选: %s（%s，%d 个账号）。注意：微信 4.0 数据库为加密格式，需导入导出的 JSONL 才能建索引。",
				len(dirs), dirs[0].Path, dirs[0].Format, dirs[0].Accounts)
		}
	})
}

func findJSONL() string {
	// 依次在 exe 同级 data/ 与 上级 ai-twin/data/ 中查找最大的 JSONL 导出
	var roots []string
	roots = append(roots, filepath.Join(dataDir, "*.jsonl"))
	if sibling := filepath.Join(wd, "..", "ai-twin", "data", "*.jsonl"); sibling != filepath.Join(dataDir, "*.jsonl") {
		roots = append(roots, sibling)
	}
	best, bestSize := "", int64(0)
	for _, pat := range roots {
		files, _ := filepath.Glob(pat)
		for _, f := range files {
			if st, err := os.Stat(f); err == nil && st.Size() > bestSize {
				best, bestSize = f, st.Size()
			}
		}
	}
	return best
} // ---------------- API ----------------

func apiHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	path := strings.TrimPrefix(r.URL.Path, "/api/")
	path = strings.TrimSuffix(path, "/")
	q := r.URL.Query()

	writeJSON := func(v interface{}) {
		_ = json.NewEncoder(w).Encode(v)
	}
	writeErr := func(msg string, code int) {
		w.WriteHeader(code)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
	}

	switch {
	case path == "meta":
		writeJSON(map[string]interface{}{
			"version":   "1.2.0",
			"engine":    "go",
			"db_exists": GDB.Exists(),
			"db_count":  GDB.Count(),
			"data_dir":  dataDir,
			"model_dir": modelDir,
		})
	case path == "sources":
		writeJSON(map[string]interface{}{"sources": SourceInfos(), "count": GDB.Count(), "source": GDB.Source()})
	case path == "reindex":
		srcs := SourceInfos()
		files := []string{}
		for _, s := range srcs {
			files = append(files, s.Path)
		}
		if len(files) == 0 {
			writeErr("没有可导入的 JSONL 数据", 400)
			return
		}
		// 清空复用签名强制重建（合并去重）
		GDB.metaSet("source_sig", "")
		go GDB.EnsureIndexMulti(files, nil)
		writeJSON(map[string]interface{}{"ok": true, "msg": "正在合并去重重建索引…", "files": len(files)})
	case path == "discover":
		writeJSON(DiscoverWeChatDirs())
	case path == "wechat_scan":
		pid, pname := findWeChatProcess()
		dbs := scanEncryptedWeChatDBs()
		writeJSON(map[string]interface{}{
			"wechat_running": pid != 0,
			"process":        pname,
			"pid":            pid,
			"dbs":            dbs,
			"total":          len(dbs),
			"total_size_mb":  func() float64 {
				var s int64
				for _, d := range dbs {
					s += d.Size
				}
				return float64(s) / 1e6
			}(),
		})
	case path == "wechat_decrypt":
		if err := StartWeChatDecrypt(); err != nil {
			writeErr(err.Error(), 400)
			return
		}
		writeJSON(map[string]interface{}{"ok": true, "msg": "解密任务已启动，请保持微信运行"})
	case path == "wechat_decrypt_status":
		writeJSON(decryptJobState())
	case path == "import":
		p := q.Get("path")
		if p == "" {
			writeErr("缺少 path 参数", 400)
			return
		}
		go GDB.EnsureIndex(p, nil)
		writeJSON(map[string]interface{}{"ok": true, "msg": "后台导入中…"})
	case path == "status":
		writeJSON(map[string]interface{}{
			"indexing": GDB.Indexing(),
			"count":    GDB.Count(),
			"jsonl":    GDB.Source(),
		})
	case path == "chats":
		// min=最小条数（默认 30），limit=数量上限（0=全部），前端"显示全部"用 min=1&limit=0
		min := atoiOr(q.Get("min"), 30)
		limit := atoiOr(q.Get("limit"), 500)
		chats := GDB.Chats2(min, limit)
		writeJSON(map[string]interface{}{"chats": chats, "count": len(chats)})
	case path == "stats":
		chat := q.Get("chat")
		if chat == "" {
			writeErr("缺少 chat 参数", 400)
			return
		}
		writeJSON(GDB.Stats(chat))
	case path == "sentiment":
		chat := q.Get("chat")
		if chat == "" {
			writeErr("缺少 chat 参数", 400)
			return
		}
		writeJSON(GDB.Sentiment(chat))
	case path == "sentiment_series":
		chat := q.Get("chat")
		writeJSON(GDB.SentimentSeries(chat))
	case path == "sentiment_heatmap":
		chat := q.Get("chat")
		writeJSON(GDB.SentimentHeatmap(chat))
	case path == "sentiment_emotions":
		chat := q.Get("chat")
		writeJSON(GDB.EmotionDistribution(chat))
	case path == "sentiment_msgs":
		chat := q.Get("chat")
		kind := q.Get("kind")
		limit := atoiOr(q.Get("limit"), 100)
		writeJSON(GDB.SentimentMsgs(chat, kind, limit))
	case path == "search":
		// FTS5 全文搜索：q=关键词，page=页码，ps=每页，from/to=时间戳(秒)，text=1 仅文本
		kw := q.Get("q")
		page := atoiOr(q.Get("page"), 1)
		ps := atoiOr(q.Get("ps"), 50)
		from := int64(atoiOr(q.Get("from"), 0))
		to := int64(atoiOr(q.Get("to"), 0))
		onlyText := q.Get("text") == "1"
		writeJSON(GDB.SearchMessages(kw, page, ps, from, to, onlyText))
	case path == "export_train":
		// 从索引导出训练数据 JSONL（P0-5：训练数据源不再依赖外部文件）
		// min=会话最小条数（默认 30），返回导出条数与文件路径
		min := atoiOr(q.Get("min"), 30)
		out := filepath.Join(dataDir, "chat_messages_e.jsonl")
		n, err := GDB.ExportChatMessages(out, min)
		if err != nil {
			writeJSON(map[string]interface{}{"error": err.Error()})
			return
		}
		writeJSON(map[string]interface{}{"ok": true, "count": n, "file": out})
	case path == "export":
		chat := q.Get("chat")
		fmt := q.Get("fmt")
		if fmt == "" {
			fmt = "json"
		}
		writeJSON(GDB.Export(chat, fmt))
	case path == "import_scan":
		p := q.Get("path")
		files := []string{}
		if p != "" {
			matches, _ := filepath.Glob(filepath.Join(p, "*.jsonl"))
			files = append(files, matches...)
		}
		writeJSON(map[string]interface{}{"jsonl": files})
	case path == "twin":
		handleTwin(w, r)
	case path == "gpu":
		writeJSON(GPUStatus())
	case path == "models":
		writeJSON(map[string]interface{}{"local": LocalModels(), "catalog": catalog})
	case path == "train_status":
		writeJSON(trainJob.RunningState())
	case path == "train_log":
		writeJSON(map[string]interface{}{"log": tailFile(trainJob.LogFile, 4000)})
	case path == "train_start":
		chat := q.Get("chat")
		persona := q.Get("persona")
		model := q.Get("model")
		out := q.Get("out")
		epochs := atoiOr(q.Get("epochs"), 2)
		samples := atoiOr(q.Get("samples"), 15000)
		lr := 2e-4
		if v, err := strconv.ParseFloat(q.Get("lr"), 64); err == nil {
			lr = v
		}
		port := atoiOr(q.Get("port"), 5000)
		if chat == "" || model == "" {
			writeErr("缺少 chat/model 参数", 400)
			return
		}
		if out == "" {
			personaName := "me"
			if persona == "other" {
				personaName = "other"
			}
			out = filepath.Join(modelDir, chat+"-"+personaName+"-merged")
		}
		res, err := TrainStart(chat, persona, model, out, epochs, samples, lr, port)
		if err != nil {
			writeJSON(map[string]interface{}{"error": err.Error()})
			return
		}
		writeJSON(res)
	case path == "train_stop":
		TrainStop()
		writeJSON(map[string]interface{}{"ok": true})
	case path == "serve_status":
		writeJSON(map[string]interface{}{"online": ServeStatus(5000), "port": 5000})
	case path == "serve_start":
		model := q.Get("model")
		port := q.Get("port")
		res, err := ServeStart(model, port)
		if err != nil {
			writeJSON(map[string]interface{}{"error": err.Error()})
			return
		}
		writeJSON(res)
	case path == "serve_stop":
		ServeStop()
		writeJSON(map[string]interface{}{"ok": true})
	case path == "serve_log":
		writeJSON(map[string]interface{}{"log": tailFile(filepath.Join(dataDir, "serve_job.log"), 3000)})
	case path == "model_delete":
		p := q.Get("path")
		if p == "" {
			writeErr("缺少 path 参数", 400)
			return
		}
		if err := ModelDelete(p); err != nil {
			writeJSON(map[string]interface{}{"error": err.Error()})
			return
		}
		writeJSON(map[string]interface{}{"ok": true})
	case path == "model_rename":
		p := q.Get("path")
		n := q.Get("name")
		if p == "" || n == "" {
			writeErr("缺少 path/name 参数", 400)
			return
		}
		if err := ModelRename(p, n); err != nil {
			writeJSON(map[string]interface{}{"error": err.Error()})
			return
		}
		writeJSON(map[string]interface{}{"ok": true})
	case path == "model_download":
		mid := q.Get("id")
		if mid == "" {
			writeErr("缺少 id 参数", 400)
			return
		}
		if err := ModelDownload(mid); err != nil {
			writeJSON(map[string]interface{}{"error": err.Error()})
			return
		}
		writeJSON(map[string]interface{}{"ok": true, "msg": "后台下载中…"})
	default:
		writeErr("not found: "+path, 404)
	}
}

func atoiOr(s string, def int) int {
	if v, err := strconv.Atoi(s); err == nil {
		return v
	}
	return def
}

// tailFile 读取文件末尾 n 字节
func tailFile(path string, n int) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	size := st.Size()
	start := size - int64(n)
	if start < 0 {
		start = 0
	}
	buf := make([]byte, size-start)
	if _, err := f.ReadAt(buf, start); err != nil && err != io.EOF {
		return ""
	}
	return string(buf)
}

// handleTwin 转发对话请求到 Python 推理服务（数字分身）
func handleTwin(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if r.Method != http.MethodPost {
		writeJSONMsg(w, map[string]interface{}{"error": "POST only"})
		return
	}
	var req struct {
		Msg  string `json:"msg"`
		Chat string `json:"chat"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONMsg(w, map[string]interface{}{"error": "bad request"})
		return
	}
	if strings.TrimSpace(req.Msg) == "" {
		writeJSONMsg(w, map[string]interface{}{"error": "empty message"})
		return
	}
	// 探测本地 Python 推理服务（训练/推理环境）
	client := &http.Client{Timeout: 180 * time.Second}
	for _, port := range []string{"5000", "8000", "8001", "5001"} {
		url := "http://127.0.0.1:" + port + "/api/twin_chat"
		body, _ := json.Marshal(map[string]string{"msg": req.Msg, "chat": req.Chat})
		resp, err := client.Post(url, "application/json", strings.NewReader(string(body)))
		if err != nil {
			continue
		}
		defer resp.Body.Close()
		var out map[string]interface{}
		if json.NewDecoder(resp.Body).Decode(&out) == nil {
			writeJSONMsg(w, out)
			return
		}
	}
	writeJSONMsg(w, map[string]interface{}{
		"error": "数字分身推理服务未启动。请先运行训练/推理服务（Python 环境），或在训练后重启本工作台。",
	})
}

func writeJSONMsg(w http.ResponseWriter, v interface{}) {
	_ = json.NewEncoder(w).Encode(v)
}
