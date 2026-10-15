// 微信 AI 数字分身 · Go 重构版
// 本地聊天记录分析 + 情感引擎 + 数字分身入口（训练/推理仍由 Python 模块提供）
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
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
	// 以 exe 所在目录为根（兼容双击运行）
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
	flag.Parse()

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

	// 自动发现数据源：优先 data/*.jsonl，其次自动搜索微信目录
	go autoLoadData()

	// HTTP 服务
	ln, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		log.Fatalf("端口被占用: %v", err)
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
		// 1) data/*.jsonl
		jsonl := findJSONL()
		if jsonl != "" {
			log.Println("发现聊天记录:", jsonl)
			GDB.EnsureIndex(jsonl, func(done int64, total int64) {
				if total > 0 && done%500000 == 0 {
					log.Printf("导入进度: %d / %d", done, total)
				}
			})
			return
		}
		// 2) 自动搜索微信目录（注册表/默认/各盘 xwechat_files）
		log.Println("data/ 下未发现导出，自动搜索微信目录…")
		dirs := DiscoverWeChatDirs()
		if len(dirs) > 0 {
			log.Println("发现微信目录:", dirs[0])
		}
	})
}

func findJSONL() string {
	files, _ := filepath.Glob(filepath.Join(dataDir, "*.jsonl"))
	if len(files) == 0 {
		return ""
	}
	// 取最大的
	best, bestSize := "", int64(0)
	for _, f := range files {
		if st, err := os.Stat(f); err == nil && st.Size() > bestSize {
			best, bestSize = f, st.Size()
		}
	}
	return best
}

// ---------------- API ----------------

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
			"version":   "1.1.0",
			"engine":    "go",
			"db_exists": GDB.Exists(),
			"db_count":  GDB.Count(),
			"data_dir":  dataDir,
			"model_dir": modelDir,
		})
	case path == "discover":
		dirs := DiscoverWeChatDirs()
		writeJSON(map[string]interface{}{"dirs": dirs})
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
		chats := GDB.Chats()
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
	case path == "search":
		chat := q.Get("chat")
		kw := q.Get("q")
		writeJSON(GDB.Search(chat, kw))
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
	default:
		writeErr("not found: "+path, 404)
	}
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
	for _, port := range []string{"5000", "8000", "8001", "5001"} {
		url := "http://127.0.0.1:" + port + "/api/twin_chat"
		body, _ := json.Marshal(map[string]string{"msg": req.Msg, "chat": req.Chat})
		resp, err := http.Post(url, "application/json", strings.NewReader(string(body)))
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
