package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

// EnsureIndex 从 JSONL 建立/更新索引（已存在且未过期则跳过）
func (d *DB) EnsureIndex(jsonl string, progress func(done, total int64)) {
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

	st, err := os.Stat(jsonl)
	if err != nil {
		fmt.Println("数据文件不存在:", jsonl)
		return
	}

	// 复用检查：DB 有数据且不比 jsonl 旧
	d.mu.RLock()
	has := d.count > 0
	cur := d.source
	d.mu.RUnlock()
	if has && cur == jsonl {
		fmt.Printf("索引已就绪（%d 条），跳过导入\n", d.count)
		return
	}
	if has {
		if dbst, err := os.Stat(dbPath); err == nil && dbst.ModTime().After(st.ModTime()) {
			d.mu.Lock()
			d.source = jsonl
			d.mu.Unlock()
			fmt.Printf("复用已有索引（%d 条）\n", d.count)
			return
		}
	}

	fmt.Println("构建聊天索引:", jsonl)
	t0 := time.Now()
	// 重建
	d.conn.Exec("DROP TABLE IF EXISTS msgs")
	d.conn.Exec("CREATE TABLE msgs(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, is_self INTEGER, ts INTEGER, type TEXT, content TEXT)")
	d.conn.Exec("CREATE INDEX IF NOT EXISTS idx_name ON msgs(name)")
	d.conn.Exec("CREATE INDEX IF NOT EXISTS idx_ts ON msgs(ts)")

	f, err := os.Open(jsonl)
	if err != nil {
		fmt.Println("打开失败:", err)
		return
	}
	defer f.Close()

	total := st.Size()
	var done int64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	tx, _ := d.conn.Begin()
	stmt, _ := tx.Prepare("INSERT INTO msgs(name,is_self,ts,type,content) VALUES(?,?,?,?,?)")
	batch := 0
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) < 2 {
			continue
		}
		var r struct {
			Name    string `json:"name"`
			User    string `json:"user"`
			IsSelf  bool   `json:"is_self"`
			TS      int64  `json:"ts"`
			Type    string `json:"type"`
			Content string `json:"content"`
		}
		if json.Unmarshal(line, &r) != nil {
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
		if _, err := stmt.Exec(name, self, ts, r.Type, r.Content); err != nil {
			continue
		}
		done++
		batch++
		if batch >= 20000 {
			tx.Commit()
			tx, _ = d.conn.Begin()
			stmt, _ = tx.Prepare("INSERT INTO msgs(name,is_self,ts,type,content) VALUES(?,?,?,?,?)")
			batch = 0
			if progress != nil {
				progress(done, total)
			}
		}
	}
	tx.Commit()

	d.mu.Lock()
	d.count = done
	d.source = jsonl
	d.mu.Unlock()
	fmt.Printf("索引完成: %d 条，耗时 %s\n", done, time.Since(t0).Round(time.Second))
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
