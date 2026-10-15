package main

import (
	"bufio"
	"crypto/sha1"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// DB SQLite 索引封装
type DB struct {
	conn    *sql.DB
	mu      sync.RWMutex
	indexMu sync.Mutex
	source  string
	count   int64
}

func OpenDB(path string) (*DB, error) {
	conn, err := sql.Open("sqlite", path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(10000)")
	if err != nil {
		return nil, err
	}
	conn.SetMaxOpenConns(1) // 单写
	d := &DB{conn: conn}
	d.mu.RLock()
	row := d.conn.QueryRow("SELECT COUNT(*) FROM msgs")
	if err := row.Scan(&d.count); err != nil {
		d.count = 0
	}
	d.mu.RUnlock()
	return d, nil
}

func (d *DB) Close() { d.conn.Close() }
func (d *DB) Exists() bool {
	return d.count > 0
}
func (d *DB) Count() int64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.count
}
func (d *DB) Source() string {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.source
}
func (d *DB) Indexing() bool {
	return indexing
}

var indexing bool

// ---------------- 多源合并去重导入 ----------------

// JSONLSource 一个聊天记录数据源
type JSONLSource struct {
	Path    string `json:"path"`
	Size    int64  `json:"size"`
	Est     int64  `json:"est_lines"` // 估算消息数（采样）
	ModTime string `json:"mtime"`
}

// SourceInfos 返回全部已知 JSONL 数据源（含大小/估算条数）
func SourceInfos() []JSONLSource {
	outs := []JSONLSource{}
	seen := map[string]bool{}
	roots := []string{filepath.Join(dataDir, "*.jsonl")}
	if sibling := filepath.Join(wd, "..", "ai-twin", "data", "*.jsonl"); sibling != filepath.Join(dataDir, "*.jsonl") {
		roots = append(roots, sibling)
	}
	for _, pat := range roots {
		files, _ := filepath.Glob(pat)
		for _, f := range files {
			if seen[f] {
				continue
			}
			seen[f] = true
			st, err := os.Stat(f)
			if err != nil {
				continue
			}
			outs = append(outs, JSONLSource{
				Path:    f,
				Size:    st.Size(),
				Est:     estJSONLLines(f),
				ModTime: st.ModTime().Format("2006-01-02 15:04"),
			})
		}
	}
	sort.Slice(outs, func(i, j int) bool { return outs[i].Size > outs[j].Size })
	return outs
}

// estJSONLLines 采样估算 JSONL 行数（读前 20 万行算平均行字节）
func estJSONLLines(path string) int64 {
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	var n, bytes int64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		bytes += int64(len(sc.Bytes()))
		n++
		if n >= 200000 {
			break
		}
	}
	if n == 0 {
		return 0
	}
	if st, err := os.Stat(path); err == nil {
		return int64(float64(st.Size()) * float64(n) / float64(bytes))
	}
	return n
}

// EnsureIndex 单文件导入（兼容旧调用），内部走多源合并
func (d *DB) EnsureIndex(jsonl string, progress func(done, total int64)) {
	d.EnsureIndexMulti([]string{jsonl}, progress)
}

// flexBool 兼容 JSON 中 is_self 的 true/false 与 1/0 两种写法
type flexBool bool

func (b *flexBool) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case "true":
		*b = true
		return nil
	case "false":
		*b = false
		return nil
	}
	var x int
	if err := json.Unmarshal(data, &x); err != nil {
		return err
	}
	*b = x != 0
	return nil
}

// JSONLRow 聊天记录导出行（兼容字段缺失）
type JSONLRow struct {
	Name    string   `json:"name"`
	User    string   `json:"user"`
	IsSelf  flexBool `json:"is_self"`
	TS      int64    `json:"ts"`
	Type    string   `json:"type"`
	Content string   `json:"content"`
}

// parseRow 解析一行；失败时打印采样（前 5 个）便于排查数据格式问题
func parseRow(line []byte, errs *int) (JSONLRow, bool) {
	var r JSONLRow
	if json.Unmarshal(line, &r) != nil {
		if errs != nil && *errs < 5 {
			*errs++
			fmt.Printf("JSONL 行解析失败(前%d字节): %s\n", 80, string(line[:min(80, len(line))]))
		}
		return r, false
	}
	return r, true
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// EnsureIndexMulti 从多个 JSONL 合并建立索引，按内容+时间+会话去重
func (d *DB) EnsureIndexMulti(files []string, progress func(done, total int64)) {
	d.indexMu.Lock()
	if indexing {
		d.indexMu.Unlock()
		return
	}
	indexing = true
	d.indexMu.Unlock()

	defer func() {
		d.indexMu.Lock()
		indexing = false
		d.indexMu.Unlock()
	}()

	// 过滤存在的文件，按大小排序（大文件在前）
	exist := []string{}
	for _, f := range files {
		if st, err := os.Stat(f); err == nil && st.Size() > 0 {
			exist = append(exist, f)
		}
	}
	if len(exist) == 0 {
		fmt.Println("没有可导入的 JSONL 数据")
		return
	}

	// 复用检查：DB 有数据 且 已导入的文件集合(含 mtime)与当前一致
	sig := sourceSig(exist)
	d.mu.RLock()
	has := d.count > 0
	prev := d.metaGet("source_sig")
	d.mu.RUnlock()
	if has && prev == sig {
		d.mu.Lock()
		d.source = strings.Join(exist, "|")
		d.mu.Unlock()
		fmt.Printf("复用已有索引（%d 条），数据源未变化\n", d.count)
		return
	}

	fmt.Println("构建聊天索引（多源合并去重）:", exist)
	t0 := time.Now()

	// 重建表（sig 唯一键实现去重）
	d.conn.Exec("DROP TABLE IF EXISTS msgs")
	d.conn.Exec("CREATE TABLE msgs(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, is_self INTEGER, ts INTEGER, type TEXT, content TEXT, sig TEXT UNIQUE)")
	d.conn.Exec("CREATE INDEX IF NOT EXISTS idx_name ON msgs(name)")
	d.conn.Exec("CREATE INDEX IF NOT EXISTS idx_ts ON msgs(ts)")

	var total, done int64
	for _, f := range exist {
		if st, err := os.Stat(f); err == nil {
			total += estJSONLLines(f)
			_ = st
		}
	}
	if total == 0 {
		total = 1
	}

	ins := func(path string, p func(done, total int64)) int64 {
		f, err := os.Open(path)
		if err != nil {
			return 0
		}
		defer f.Close()
		var n int64
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1024*1024), 1024*1024)
		tx, err := d.conn.Begin()
		if err != nil {
			fmt.Printf("Begin 失败: %v\n", err)
			return 0
		}
		stmt, err := tx.Prepare("INSERT OR IGNORE INTO msgs(name,is_self,ts,type,content,sig) VALUES(?,?,?,?,?,?)")
		if err != nil {
			fmt.Printf("Prepare 失败: %v\n", err)
			return 0
		}
		batch := 0
		var execErrs, parseErrs int
		for sc.Scan() {
			line := sc.Bytes()
			if len(line) < 2 {
				continue
			}
			r, ok := parseRow(line, &parseErrs)
			if !ok {
				continue
			}
			name := r.Name
			if name == "" {
				name = r.User
			}
			if name == "" {
				name = "?"
			}
			ts := r.TS
			if ts == 0 {
				ts = time.Now().Unix()
			}
			self := 0
			if r.IsSelf {
				self = 1
			}
			sig := msgSig(name, self, ts, r.Type, r.Content)
			if res, err := stmt.Exec(name, self, ts, r.Type, r.Content, sig); err == nil {
				if aff, _ := res.RowsAffected(); aff > 0 {
					n++
				}
			} else if execErrs < 5 {
				execErrs++
				fmt.Printf("Exec 错误: %v\n", err)
			}
			batch++
			if batch >= 20000 {
				tx.Commit()
				tx, err = d.conn.Begin()
				if err != nil {
					fmt.Printf("重新 Begin 失败: %v\n", err)
					break
				}
				stmt, err = tx.Prepare("INSERT OR IGNORE INTO msgs(name,is_self,ts,type,content,sig) VALUES(?,?,?,?,?,?)")
				if err != nil {
					fmt.Printf("重新 Prepare 失败: %v\n", err)
					break
				}
				batch = 0
				if p != nil {
					p(n, total)
				}
			}
		}
		if err := sc.Err(); err != nil {
			fmt.Printf("Scanner 错误: %v\n", err)
		}
		tx.Commit()
		return n
	}

	for _, f := range exist {
		fmt.Println("导入:", f)
		n := ins(f, func(done, total int64) {
			if progress != nil {
				progress(done, total)
			}
		})
		done += n
		fmt.Printf("  本文件 %d 条（累计去重后 %d 条）\n", n, done)
	}

	// 写 meta：source_sig（复用依据）
	d.mu.Lock()
	d.count = done
	d.source = strings.Join(exist, "|")
	d.metaSet("source_sig", sig)
	d.mu.Unlock()
	fmt.Printf("索引完成: 去重后 %d 条，耗时 %s\n", done, time.Since(t0).Round(time.Second))
}

func msgSig(name string, self int, ts int64, typ, content string) string {
	h := sha1.Sum([]byte(fmt.Sprintf("%s|%d|%d|%s|%s", name, self, ts, typ, content)))
	return hex.EncodeToString(h[:])
}

func sourceSig(files []string) string {
	var b strings.Builder
	for _, f := range files {
		st, err := os.Stat(f)
		if err != nil {
			continue
		}
		b.WriteString(f)
		b.WriteString(":")
		b.WriteString(fmt.Sprintf("%d", st.Size()))
		b.WriteString(":")
		b.WriteString(fmt.Sprintf("%d", st.ModTime().Unix()))
		b.WriteString(";")
	}
	h := sha1.Sum([]byte(b.String()))
	return hex.EncodeToString(h[:])
}

// ---------------- meta 键值 ----------------

func (d *DB) metaGet(k string) string {
	var v string
	err := d.conn.QueryRow("SELECT v FROM meta WHERE k=?", k).Scan(&v)
	if err != nil {
		return ""
	}
	return v
}

func (d *DB) metaSet(k, v string) {
	d.conn.Exec("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
	d.conn.Exec("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", k, v)
}

// ---------------- 查询 ----------------

type ChatInfo struct {
	Name  string `json:"name"`
	Total int64  `json:"total"`
	Self  int64  `json:"self"`
	Other int64  `json:"other"`
}

// Chats 会话列表（毫秒级）
func (d *DB) Chats() []ChatInfo {
	rows, err := d.conn.Query("SELECT name, COUNT(*), SUM(is_self) FROM msgs GROUP BY name HAVING COUNT(*) >= 30 ORDER BY COUNT(*) DESC LIMIT 500")
	if err != nil {
		return []ChatInfo{}
	}
	defer rows.Close()
	chats := []ChatInfo{}
	for rows.Next() {
		var c ChatInfo
		var s sql.NullInt64
		if rows.Scan(&c.Name, &c.Total, &s) == nil {
			c.Self = s.Int64
			c.Other = c.Total - c.Self
			chats = append(chats, c)
		}
	}
	return chats
}

// Export 导出会话
func (d *DB) Export(chat, format string) map[string]interface{} {
	rows, err := d.conn.Query("SELECT is_self, ts, type, content FROM msgs WHERE name=? ORDER BY id", chat)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()
	type msg struct {
		Self    bool   `json:"self"`
		TS      int64  `json:"ts"`
		Type    string `json:"type"`
		Content string `json:"content"`
	}
	msgs := []msg{}
	for rows.Next() {
		var m msg
		var self int
		rows.Scan(&self, &m.TS, &m.Type, &m.Content)
		m.Self = self == 1
		msgs = append(msgs, m)
	}
	// 写出文件
	out := filepath.Join(dataDir, fmt.Sprintf("export_%s.%s", safeName(chat), format))
	data, _ := json.MarshalIndent(msgs, "", "  ")
	if format == "json" {
		os.WriteFile(out, data, 0644)
	}
	return map[string]interface{}{"count": len(msgs), "file": out}
}

func safeName(s string) string {
	out := make([]rune, 0, len(s))
	for _, r := range s {
		if r == '/' || r == '\\' || r == ':' || r == '*' || r == '?' || r == '"' || r == '<' || r == '>' || r == '|' {
			out = append(out, '_')
		} else {
			out = append(out, r)
		}
	}
	return string(out)
}
