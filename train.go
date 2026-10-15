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

// pyHasModule 检查指定 python 能否 import 某模块
func pyHasModule(py, mod string) bool {
	c := exec.Command(py, "-c", "import "+mod)
	return c.Run() == nil
}

// 镜像模型目录（ModelScope 可下载；Qwen/DeepSeek/GLM/MiniCPM 系列，按显存分组）
var catalog = []map[string]interface{}{
	// —— 核显 / CPU 可跑（int4 量化后 <2GB，无独显也能推理）——
	{"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B", "size_gb": 1.0, "tier": "核显/CPU 可跑（int4）", "desc": "极轻量，核显首选"},
	{"id": "Qwen/Qwen3-0.6B", "name": "Qwen3-0.6B", "size_gb": 1.5, "tier": "核显/CPU 可跑（int4）", "desc": "轻量，支持思考模式"},
	{"id": "Qwen/Qwen1.5-0.5B-Chat", "name": "Qwen1.5-0.5B", "size_gb": 1.0, "tier": "核显/CPU 可跑（int4）", "desc": "经典小模型"},
	{"id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen2.5-1.5B", "size_gb": 3.1, "tier": "核显/CPU 可跑（int4）", "desc": "小模型，速度质量均衡"},
	{"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "name": "Qwen2.5-Coder-1.5B", "size_gb": 3.1, "tier": "核显/CPU 可跑（int4）", "desc": "编程小模型"},
	{"id": "Qwen/Qwen2.5-Math-1.5B-Instruct", "name": "Qwen2.5-Math-1.5B", "size_gb": 3.1, "tier": "核显/CPU 可跑（int4）", "desc": "数学小模型"},
	{"id": "Qwen/Qwen1.5-1.8B-Chat", "name": "Qwen1.5-1.8B", "size_gb": 3.6, "tier": "核显/CPU 可跑（int4）", "desc": "轻量对话"},
	{"id": "Qwen/Qwen2-1.5B-Instruct", "name": "Qwen2-1.5B", "size_gb": 3.1, "tier": "核显/CPU 可跑（int4）", "desc": "Qwen2 系小模型"},
	{"id": "Qwen/Qwen2.5-0.5B", "name": "Qwen2.5-0.5B-Base", "size_gb": 1.0, "tier": "核显/CPU 可跑（int4）", "desc": "基座版（需自行微调）"},
	{"id": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "name": "Qwen2.5-Coder-0.5B", "size_gb": 1.0, "tier": "核显/CPU 可跑（int4）", "desc": "轻量编程模型"},
	// —— 6GB 显卡推荐（4bit 微调 / 8bit 推理）——
	{"id": "Qwen/Qwen3-1.7B", "name": "Qwen3-1.7B", "size_gb": 3.6, "tier": "6GB 显卡推荐", "desc": "快，日常对话够用"},
	{"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B", "size_gb": 8.1, "tier": "6GB 显卡推荐（4bit 微调）", "desc": "质量/速度平衡，当前默认"},
	{"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen2.5-3B", "size_gb": 6.1, "tier": "6GB 显卡", "desc": "2.5 系，稳妥"},
	{"id": "Qwen/Qwen2.5-Coder-3B-Instruct", "name": "Qwen2.5-Coder-3B", "size_gb": 6.2, "tier": "6GB 显卡", "desc": "编程 3B"},
	{"id": "Qwen/Qwen2-VL-2B-Instruct", "name": "Qwen2-VL-2B", "size_gb": 5.0, "tier": "6GB 显卡", "desc": "视觉语言 2B"},
	{"id": "Qwen/Qwen2.5-VL-3B-Instruct", "name": "Qwen2.5-VL-3B", "size_gb": 7.0, "tier": "6GB 显卡", "desc": "视觉语言 3B"},
	{"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "name": "DeepSeek-R1-1.5B", "size_gb": 3.2, "tier": "6GB 显卡", "desc": "推理蒸馏小模型"},
	{"id": "OpenBMB/MiniCPM3-4B", "name": "MiniCPM3-4B", "size_gb": 8.0, "tier": "6GB 显卡", "desc": "国产 4B 强模型"},
	// —— 12GB 显卡推荐 ——
	{"id": "Qwen/Qwen3-8B", "name": "Qwen3-8B", "size_gb": 16.0, "tier": "12GB+ 显卡", "desc": "更强，需大显存"},
	{"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen2.5-7B", "size_gb": 15.0, "tier": "12GB+ 显卡", "desc": "2.5 系大模型"},
	{"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "name": "Qwen2.5-Coder-7B", "size_gb": 15.3, "tier": "12GB+ 显卡", "desc": "编程 7B"},
	{"id": "Qwen/Qwen2.5-Math-7B-Instruct", "name": "Qwen2.5-Math-7B", "size_gb": 15.0, "tier": "12GB+ 显卡", "desc": "数学 7B"},
	{"id": "Qwen/Qwen2-VL-7B-Instruct", "name": "Qwen2-VL-7B", "size_gb": 18.0, "tier": "12GB+ 显卡", "desc": "视觉语言 7B"},
	{"id": "Qwen/Qwen2.5-VL-7B-Instruct", "name": "Qwen2.5-VL-7B", "size_gb": 19.0, "tier": "12GB+ 显卡", "desc": "视觉语言 7B 新版"},
	{"id": "Qwen/Qwen2-7B-Instruct", "name": "Qwen2-7B", "size_gb": 15.0, "tier": "12GB+ 显卡", "desc": "Qwen2 7B"},
	{"id": "Qwen/Qwen1.5-7B-Chat", "name": "Qwen1.5-7B", "size_gb": 15.0, "tier": "12GB+ 显卡", "desc": "Qwen1.5 7B"},
	{"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek-R1-7B", "size_gb": 15.0, "tier": "12GB+ 显卡", "desc": "推理蒸馏 7B"},
	{"id": "ZhipuAI/chatglm3-6b", "name": "ChatGLM3-6B", "size_gb": 12.5, "tier": "12GB+ 显卡", "desc": "智谱 GLM3"},
	{"id": "ZhipuAI/glm-4-9b-chat", "name": "GLM-4-9B", "size_gb": 18.0, "tier": "12GB+ 显卡", "desc": "智谱 GLM4"},
	{"id": "OpenBMB/MiniCPM-V-2_6", "name": "MiniCPM-V-2.6", "size_gb": 16.0, "tier": "12GB+ 显卡", "desc": "多模态 8B"},
	// —— 16GB+ 显卡 ——
	{"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen2.5-14B", "size_gb": 29.0, "tier": "16GB+ 显卡", "desc": "2.5 系 14B"},
	{"id": "Qwen/Qwen3-14B", "name": "Qwen3-14B", "size_gb": 28.0, "tier": "16GB+ 显卡", "desc": "Qwen3 14B"},
	{"id": "Qwen/Qwen2.5-Coder-14B-Instruct", "name": "Qwen2.5-Coder-14B", "size_gb": 29.0, "tier": "16GB+ 显卡", "desc": "编程 14B"},
	{"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "name": "DeepSeek-R1-14B", "size_gb": 29.0, "tier": "16GB+ 显卡", "desc": "推理蒸馏 14B"},
	// —— 24GB+ 显卡 ——
	{"id": "Qwen/Qwen3-30B-A3B", "name": "Qwen3-30B-A3B", "size_gb": 18.5, "tier": "24GB+ 显卡", "desc": "MoE 激活 3B，速度快"},
	{"id": "Qwen/Qwen3-32B", "name": "Qwen3-32B", "size_gb": 65.0, "tier": "24GB+ 显卡", "desc": "Qwen3 32B 满血"},
	{"id": "Qwen/Qwen2.5-32B-Instruct", "name": "Qwen2.5-32B", "size_gb": 65.0, "tier": "24GB+ 显卡", "desc": "2.5 系 32B"},
	{"id": "Qwen/QwQ-32B", "name": "QwQ-32B", "size_gb": 65.0, "tier": "24GB+ 显卡", "desc": "推理专用 32B"},
	{"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen2.5-72B", "size_gb": 145.0, "tier": "24GB+ 显卡（量化）", "desc": "2.5 系 72B"},
	// —— 服务器级 ——
	{"id": "Qwen/Qwen3-235B-A22B", "name": "Qwen3-235B-A22B", "size_gb": 140.0, "tier": "服务器级", "desc": "MoE 旗舰"},
}

// ---------------- 训练任务管理 ----------------

type TrainJob struct {
	mu       sync.Mutex
	Running  bool    `json:"running"`
	Stage    string  `json:"stage"`
	Step     int     `json:"step"`
	Total    int     `json:"total"`
	Loss     float64 `json:"loss"`
	LogFile  string  `json:"log_file"`
	Chat     string  `json:"chat"`
	Persona  string  `json:"persona"`
	Out      string  `json:"out"`
	Started  string  `json:"started"`
	DLSize   float64 `json:"dl_size"`  // 下载已用 MB
	DLSpeed  float64 `json:"dl_speed"` // 下载速度 MB/s
	DLTotal  float64 `json:"dl_total"` // 下载总大小 MB（0=未知）
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

// trainDepsOK 检测训练/推理依赖（torch/transformers/peft/accelerate）是否可用（P0-6）
func trainDepsOK() bool {
	probe := "import importlib.util as u; mods=['torch','transformers','peft','accelerate']; print(all(u.find_spec(m) is not None for m in mods))"
	c := exec.Command(pyExe, "-c", probe)
	out, err := c.Output()
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(out)) == "True"
}

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
		"dl_size":  round1(j.DLSize),
		"dl_speed": round1(j.DLSpeed),
		"dl_total": round1(j.DLTotal),
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

	// P0-5：训练数据源不再依赖外部文件——每次训练前从 chats.db 索引导出（对齐 train_cli.py 解析字段）
	trainData := filepath.Join(dataDir, "chat_messages_e.jsonl")
	if GDB != nil && GDB.Count() > 0 {
		n, err := GDB.ExportChatMessages(trainData, 1)
		if err != nil {
			return nil, fmtErr("导出训练数据失败: " + err.Error())
		}
		if n == 0 {
			return nil, fmtErr("索引中没有可训练的消息（请先导入聊天记录/解密微信数据库）")
		}
		j.mu.Lock()
		j.Stage = "导出训练数据"
		j.mu.Unlock()
	} else {
		return nil, fmtErr("索引为空：请先在数据页导入聊天记录或解密微信数据库，再开始训练")
	}

	// P0-6：训练依赖检测（torch/transformers/peft/accelerate）——缺失时给出明确安装命令
	if !trainDepsOK() {
		return nil, fmtErr("训练依赖未安装。请先执行（无 GPU 机器用 CPU 版）：\n" +
			"  <venv>\\Scripts\\python.exe -m pip install -r scripts\\requirements-train-cpu.txt\n" +
			"  有 NVIDIA GPU 再加：-r scripts\\requirements-train-gpu.txt\n" +
			"  国内网络可追加 -i https://pypi.tuna.tsinghua.edu.cn/simple")
	}

	logFile := filepath.Join(dataDir, "train_job.log")
	os.Remove(logFile)

	args := []string{
		filepath.Join(wd, "scripts", "train_cli.py"),
		"--data", trainData,
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
			if strings.HasPrefix(ln, "[DLTOTAL]") {
				fields := strings.Fields(ln)
				if len(fields) >= 2 {
					if v, err := strconv.ParseFloat(fields[1], 64); err == nil {
						j.mu.Lock()
						j.DLTotal = v / 1e6 // 字节 -> MB
						j.mu.Unlock()
					}
				}
			}
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
	j.Running = true
	j.Stage = "检查下载环境…"
	j.DLSize = 0
	j.DLSpeed = 0
	j.DLTotal = 0
	j.mu.Unlock()

	// 确保 modelscope 可用（缺则自动安装，国内镜像列表来自 pip-mirrors.txt）
	if !pyHasModule(pyExe, "modelscope") {
		j.mu.Lock()
		j.Stage = "安装 modelscope（下载工具）…"
		j.mu.Unlock()
		installed := false
		for _, mirror := range pipMirrors() {
			args := []string{"-m", "pip", "install", "--quiet", "--disable-pip-version-check"}
			if mirror != "" {
				args = append(args, "-i", mirror)
			}
			args = append(args, "modelscope")
			c := exec.Command(pyExe, args...)
			if c.Run() == nil && pyHasModule(pyExe, "modelscope") {
				installed = true
				break
			}
		}
		if !installed {
			j.mu.Lock()
			j.Running = false
			j.Stage = "安装 modelscope 失败（网络受限？）"
			j.mu.Unlock()
			return fmtErr("自动安装 modelscope 失败（网络受限？）。可手动执行：" + pyExe + " -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope")
		}
	}

	logFile := filepath.Join(dataDir, "model_dl.log")
	os.Remove(logFile)
	f, err := os.Create(logFile)
	if err != nil {
		j.mu.Lock()
		j.Running = false
		j.mu.Unlock()
		return err
	}
	name := modelID
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	target := filepath.Join(modelDir, name)
	dl := exec.Command(pyExe, filepath.Join(wd, "scripts", "model_download.py"), modelID, target)
	dl.Dir = wd
	dl.Stdout = f
	dl.Stderr = f
	if err := dl.Start(); err != nil {
		j.mu.Lock()
		j.Running = false
		j.mu.Unlock()
		return err
	}
	j.mu.Lock()
	j.Running = true
	j.Stage = "下载模型 " + modelID
	j.Chat = modelID
	j.Persona = "download"
	j.LogFile = logFile
	j.Started = time.Now().Format("2006-01-02 15:04:05")
	j.DLSize, j.DLSpeed, j.DLTotal = 0, 0, 0
	j.proc = dl
	j.mu.Unlock()
	go func() {
		err := dl.Wait()
		f.Close()
		j.mu.Lock()
		j.Running = false
		if err != nil || dirSize(target) < 1e6 {
			j.Stage = "下载失败（详见日志/模型目录）"
		} else {
			j.Stage = "下载完成"
		}
		j.DLSpeed = 0
		j.mu.Unlock()
	}()
	go j.watchLog(logFile)
	// 下载速度轮询：统计目标目录大小增量
	go func() {
		prev := 0.0
		t0 := time.Now()
		for {
			time.Sleep(2 * time.Second)
			j.mu.Lock()
			if !j.Running || j.Persona != "download" {
				j.mu.Unlock()
				return
			}
			cur := dirSize(target) / 1e6
			dt := time.Since(t0).Seconds()
			if dt >= 2 {
				j.DLSpeed = (cur - prev) / dt
				if j.DLSpeed < 0 {
					j.DLSpeed = 0
				}
				prev = cur
				t0 = time.Now()
			}
			j.DLSize = cur
			j.mu.Unlock()
		}
	}()
	return nil
}

// ---------------- 推理服务手动启停 ----------------

var serveMu sync.Mutex
var serveProc *exec.Cmd

// ServeStart 手动启动数字分身推理服务（serve_twin.py）
func ServeStart(model, portStr string) (map[string]interface{}, error) {
	serveMu.Lock()
	defer serveMu.Unlock()
	if serveProc != nil && processAlive(serveProc.Process.Pid) {
		return nil, fmtErr("推理服务已在运行，可先停止再重启")
	}
	port, _ := strconv.Atoi(portStr)
	if port == 0 {
		port = 5000
	}
	if model == "" {
		model = defaultServeModel()
	}
	if model == "" {
		return nil, fmtErr("没有可用模型。请先在「训练」页下载或选择本地模型")
	}
	if _, err := os.Stat(filepath.Join(model, "config.json")); err != nil {
		return nil, fmtErr("模型目录不存在或缺少 config.json: " + model)
	}
	logFile := filepath.Join(dataDir, "serve_job.log")
	os.Remove(logFile)
	f, err := os.Create(logFile)
	if err != nil {
		return nil, err
	}
	args := []string{
		filepath.Join(wd, "scripts", "serve_twin.py"),
		"--model", model,
		"--port", strconv.Itoa(port),
		"--bits", "8",
	}
	cmd := exec.Command(pyExe, args...)
	cmd.Dir = wd
	cmd.Stdout = f
	cmd.Stderr = f
	if err := cmd.Start(); err != nil {
		f.Close()
		return nil, err
	}
	serveProc = cmd
	go func() {
		cmd.Wait()
		f.Close()
		serveMu.Lock()
		serveProc = nil
		serveMu.Unlock()
	}()
	return map[string]interface{}{"ok": true, "msg": "推理服务启动中…", "port": port, "model": model, "log": logFile}, nil
}

// ServeStop 停止推理服务
func ServeStop() {
	serveMu.Lock()
	defer serveMu.Unlock()
	if serveProc != nil {
		execCommand("taskkill", "/PID", strconv.Itoa(serveProc.Process.Pid), "/T", "/F").Run()
		serveProc = nil
	}
}

// defaultServeModel 选择默认推理模型：优先 *-merged，其次本地最大模型
func defaultServeModel() string {
	ms := LocalModels()
	if len(ms) == 0 {
		return ""
	}
	for _, m := range ms {
		if n, ok := m["name"].(string); ok && strings.Contains(n, "merged") {
			if p, ok := m["path"].(string); ok {
				return p
			}
		}
	}
	if p, ok := ms[0]["path"].(string); ok {
		return p
	}
	return ""
}

// ---------------- 模型管理（删除/重命名） ----------------

// modelRoots 返回允许管理的模型根目录（exe/models 与 ../ai-twin/models）
func modelRoots() []string {
	roots := []string{modelDir}
	if s := filepath.Join(wd, "..", "ai-twin", "models"); s != modelDir {
		roots = append(roots, s)
	}
	return roots
}

// inModelRoot 校验 path 是否在某个模型根目录的一级子目录内（防路径穿越）
func inModelRoot(path string) (string, bool) {
	clean := filepath.Clean(path)
	for _, r := range modelRoots() {
		rr := filepath.Clean(r)
		if clean == rr {
			return "", false // 根目录本身不可删
		}
		if strings.HasPrefix(clean, rr+string(os.PathSeparator)) {
			rel, _ := filepath.Rel(rr, clean)
			if !strings.Contains(rel, string(os.PathSeparator)) {
				return rr, true // 仅允许一级子目录（模型目录本身）
			}
		}
	}
	return "", false
}

// ModelDelete 删除本地模型目录（不可逆，前端需二次确认）
func ModelDelete(path string) error {
	if _, ok := inModelRoot(path); !ok {
		return fmtErr("只允许删除 models/ 下的模型目录")
	}
	if strings.HasSuffix(strings.ToLower(path), ".exe") || strings.HasSuffix(strings.ToLower(path), ".py") {
		return fmtErr("不是有效的模型目录")
	}
	return os.RemoveAll(path)
}

// ModelRename 重命名本地模型目录
func ModelRename(path, newName string) error {
	root, ok := inModelRoot(path)
	if !ok {
		return fmtErr("只允许重命名 models/ 下的模型目录")
	}
	newName = strings.TrimSpace(newName)
	if newName == "" || strings.ContainsAny(newName, `\/:*?"<>|`) {
		return fmtErr("模型名不合法（不能包含 \\ / : * ? \" < > |）")
	}
	np := filepath.Join(root, newName)
	if _, err := os.Stat(np); err == nil {
		return fmtErr("已存在同名目录")
	}
	return os.Rename(path, np)
}

func fmtErr(s string) error {
	return &trainErr{s}
}

type trainErr struct{ s string }

func (e *trainErr) Error() string { return e.s }
