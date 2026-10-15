package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---------------- GPU 监控 ----------------

type GPUInfo struct {
	Available   bool    `json:"available"`
	Name        string  `json:"name"`
	TotalGB     float64 `json:"total_gb"`
	UsedGB      float64 `json:"used_gb"`
	FreeGB      float64 `json:"free_gb"`
	UtilPct     int     `json:"util_pct"`
	TrainerProc bool    `json:"trainer_proc"` // 是否有训练进程占用
}

// GPUInfo 返回当前显卡状态（nvidia-smi）
func GPUStatus() GPUInfo {
	out, err := execCommand("nvidia-smi",
		"--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return GPUInfo{Available: false}
	}
	line := strings.TrimSpace(string(out))
	parts := strings.Split(line, ",")
	if len(parts) < 5 {
		return GPUInfo{Available: false}
	}
	f := func(s string) float64 {
		v, _ := strconv.ParseFloat(strings.TrimSpace(s), 64)
		return v
	}
	util, _ := strconv.Atoi(strings.TrimSpace(parts[4]))
	return GPUInfo{
		Available: true,
		Name:      strings.TrimSpace(parts[0]),
		TotalGB:   f(parts[1]) / 1024,
		UsedGB:    f(parts[2]) / 1024,
		FreeGB:    f(parts[3]) / 1024,
		UtilPct:   util,
	}
}

// ---------------- 模型管理 ----------------

// LocalModels 扫描本地模型目录（含 config.json）
// 主目录: <exe>/models ；次级目录: <exe>/../ai-twin/models
func LocalModels() []map[string]interface{} {
	roots := []string{modelDir}
	if sibling := filepath.Join(wd, "..", "ai-twin", "models"); sibling != modelDir {
		roots = append(roots, sibling)
	}
	seen := map[string]bool{}
	out := []map[string]interface{}{}
	for _, root := range roots {
		if entries, err := os.ReadDir(root); err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				cfg := filepath.Join(root, e.Name(), "config.json")
				if _, err := os.Stat(cfg); err == nil {
					name := e.Name()
					if seen[name] {
						continue
					}
					seen[name] = true
					out = append(out, map[string]interface{}{
						"name":    name,
						"path":    filepath.Join(root, e.Name()),
						"size_gb": round1(dirSize(filepath.Join(root, e.Name())) / 1e9),
					})
				}
			}
		}
	}
	return out
}

func dirSize(path string) float64 {
	var sz float64
	filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			sz += float64(info.Size())
		}
		return nil
	})
	return sz
}

// 镜像模型目录（ModelScope 已验证可下载；Qwen 系列，按显存档位分组）
var catalog = []map[string]interface{}{
	{"id": "Qwen/Qwen3-1.7B", "name": "Qwen3-1.7B", "size_gb": 3.6, "tier": "6GB 显卡推荐", "desc": "快，日常对话够用"},
	{"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B", "size_gb": 8.1, "tier": "6GB 显卡推荐（4bit 微调）", "desc": "质量/速度平衡，当前默认"},
	{"id": "Qwen/Qwen3-8B", "name": "Qwen3-8B", "size_gb": 16.0, "tier": "12GB+ 显卡", "desc": "更强，需大显存"},
	{"id": "Qwen/Qwen3-0.6B", "name": "Qwen3-0.6B", "size_gb": 1.5, "tier": "集显/低配", "desc": "极轻量"},
	{"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "size_gb": 6.1, "tier": "6GB 显卡", "desc": "2.5 系，稳妥"},
	{"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B", "size_gb": 15.0, "tier": "12GB+ 显卡", "desc": "2.5 系大模型"},
}

// ---------------- 训练任务管理 ----------------

type TrainJob struct {
	mu       sync.Mutex
	Running  bool   `json:"running"`
	Stage    string `json:"stage"`
	Step     int    `json:"step"`
	Total    int    `json:"total"`
	Loss     float64 `json:"loss"`
	LogFile  string `json:"log_file"`
	Chat     string `json:"chat"`
	Persona  string `json:"persona"`
	Out      string `json:"out"`
	Started  string `json:"started"`
	proc     *exec.Cmd
	serveURL string
}

var trainJob = &TrainJob{}

var pyExe = func() string {
	candidates := []string{
		filepath.Join(wd, "..", "ai-twin", "venv", "Scripts", "python.exe"),
		filepath.Join(wd, "venv", "Scripts", "python.exe"),
		"python",
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return "python"
}()

var dataJSONL = func() string {
	candidates := []string{
		filepath.Join(wd, "data", "chat_messages_e.jsonl"),
		filepath.Join(wd, "..", "ai-twin", "data", "chat_messages_e.jsonl"),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return filepath.Join(wd, "data", "chat_messages_e.jsonl")
}()

func (j *TrainJob) RunningState() map[string]interface{} {
	j.mu.Lock()
	defer j.mu.Unlock()
	return map[string]interface{}{
		"running":  j.Running,
		"stage":    j.Stage,
		"step":     j.Step,
		"total":    j.Total,
		"loss":     round1(j.Loss),
		"chat":     j.Chat,
		"persona":  j.Persona,
		"out":      j.Out,
		"started":  j.Started,
		"log_file": j.LogFile,
	}
}

// TrainStart 启动训练任务
func TrainStart(chat, persona, modelPath, out string, epochs, samples int, lr float64, port int) (map[string]interface{}, error) {
	j := trainJob
	j.mu.Lock()
	if j.Running {
		j.mu.Unlock()
		return nil, fmtErr("已有训练任务在运行（" + j.Chat + " / " + j.Persona + "）")
	}
	j.mu.Unlock()

	logFile := filepath.Join(dataDir, "train_job.log")
	os.Remove(logFile)

	args := []string{
		filepath.Join(wd, "scripts", "train_cli.py"),
		"--data", dataJSONL,
		"--chat", chat,
		"--persona", persona,
		"--model", modelPath,
		"--epochs", strconv.Itoa(epochs),
		"--samples", strconv.Itoa(samples),
		"--lr", strconv.FormatFloat(lr, 'f', -1, 64),
		"--out", out,
		"--port", strconv.Itoa(port),
		"--log", logFile,
	}
	cmd := exec.Command(pyExe, args...)
	cmd.Dir = wd
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	go func() {
		io.Copy(io.Discard, stdout)
		io.Copy(io.Discard, stderr)
		cmd.Wait()
		j.mu.Lock()
		j.Running = false
		j.mu.Unlock()
	}()

	j.mu.Lock()
	j.Running = true
	j.Stage = "启动中"
	j.Step, j.Total, j.Loss = 0, 0, 0
	j.Chat, j.Persona = chat, persona
	j.Out = out
	j.Started = time.Now().Format("2006-01-02 15:04:05")
	j.LogFile = logFile
	j.proc = cmd
	j.mu.Unlock()

	go j.watchLog(logFile)
	return map[string]interface{}{"ok": true, "log": logFile}, nil
}

// watchLog 轮询训练日志，解析进度
func (j *TrainJob) watchLog(path string) {
	for {
		time.Sleep(2 * time.Second)
		j.mu.Lock()
		running := j.Running
		j.mu.Unlock()
		if !running {
			return
		}
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		// 读最后 50 行
		lines := []string{}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1024*64), 1024*64)
		for sc.Scan() {
			lines = append(lines, sc.Text())
			if len(lines) > 50 {
				lines = lines[1:]
			}
		}
		f.Close()
		for _, ln := range lines {
			if strings.HasPrefix(ln, "[STAGE]") {
				fields := strings.Fields(ln)
				if len(fields) >= 2 {
					j.mu.Lock()
					j.Stage = fields[1]
					j.mu.Unlock()
				}
			}
			if strings.HasPrefix(ln, "[TRAIN] step=") {
				parts := strings.Fields(ln)
				for _, p := range parts {
					if strings.HasPrefix(p, "step=") {
						j.mu.Lock()
						j.Step, _ = strconv.Atoi(strings.TrimPrefix(p, "step="))
						j.mu.Unlock()
					}
					if strings.HasPrefix(p, "total=") {
						j.mu.Lock()
						j.Total, _ = strconv.Atoi(strings.TrimPrefix(p, "total="))
						j.mu.Unlock()
					}
					if strings.HasPrefix(p, "loss=") {
						v, _ := strconv.ParseFloat(strings.TrimPrefix(p, "loss="), 64)
						j.mu.Lock()
						j.Loss = v
						j.mu.Unlock()
					}
				}
			}
		}
	}
}

// TrainStop 停止训练
func TrainStop() error {
	j := trainJob
	j.mu.Lock()
	if j.proc != nil && j.Running {
		// 杀进程树（Windows：taskkill /T）
		execCommand("taskkill", "/PID", strconv.Itoa(j.proc.Process.Pid), "/T", "/F").Run()
		j.Running = false
		j.Stage = "已停止"
	}
	j.mu.Unlock()
	return nil
}

// ServeStatus 推理服务健康检查
func ServeStatus(port int) bool {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://127.0.0.1:" + strconv.Itoa(port) + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var out map[string]interface{}
	return json.NewDecoder(resp.Body).Decode(&out) == nil
}

// ModelDownload 从 ModelScope 下载模型（后台）
func ModelDownload(modelID string) error {
	j := trainJob
	j.mu.Lock()
	if j.Running {
		j.mu.Unlock()
		return fmtErr("训练/下载任务进行中，请稍后再试")
	}
	j.mu.Unlock()

	logFile := filepath.Join(dataDir, "model_dl.log")
	os.Remove(logFile)
	dl := exec.Command(pyExe, "-c",
		"import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace');"+
			"from modelscope import snapshot_download; import os,time;"+
			"T=r'"+modelDir+"';"+
			"mid='"+modelID+"'; name=mid.split('/')[-1]; target=os.path.join(T,name); os.makedirs(target, exist_ok=True);"+
			"print('[DL] start '+mid, flush=True);"+
			"snapshot_download(mid, local_dir=target);"+
			"print('[DL] done', flush=True)")
	dl.Dir = wd
	if err := dl.Start(); err != nil {
		return err
	}
	j.mu.Lock()
	j.Running = true
	j.Stage = "下载模型 " + modelID
	j.Chat = modelID
	j.Persona = "download"
	j.LogFile = logFile
	j.Started = time.Now().Format("2006-01-02 15:04:05")
	j.proc = dl
	j.mu.Unlock()
	go func() {
		dl.Wait()
		j.mu.Lock()
		j.Running = false
		j.Stage = "下载完成"
		j.mu.Unlock()
	}()
	go j.watchLog(logFile)
	return nil
}

func fmtErr(s string) error {
	return &trainErr{s}
}

type trainErr struct{ s string }

func (e *trainErr) Error() string { return e.s }
