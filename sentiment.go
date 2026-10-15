package main

import (
	"strings"
)

// ---------------- 词频 ----------------

var stopWords = map[string]bool{
	"的": true, "了": true, "是": true, "我": true, "你": true, "他": true, "她": true, "它": true, "们": true, "在": true,
	"有": true, "就": true, "都": true, "也": true, "很": true, "说": true, "会": true, "能": true, "要": true, "想": true,
	"看": true, "去": true, "来": true, "和": true, "跟": true, "与": true, "这": true, "那": true, "吧": true, "啊": true,
	"吗": true, "呢": true, "哈": true, "哦": true, "嗯": true, "呀": true, "嘛": true, "啦": true, "不": true, "没": true,
	"么": true, "个": true, "人": true, "上": true, "下": true, "中": true, "大": true, "小": true, "好": true, "还": true,
	"又": true, "再": true, "可": true, "以": true, "过": true, "给": true, "对": true, "把": true, "被": true, "让": true,
	"从": true, "向": true, "于": true, "为": true, "而": true, "则": true, "或": true, "及": true, "但": true, "因为": true,
	"所以": true, "如果": true, "虽然": true, "但是": true, "只是": true, "不过": true, "然后": true, "现在": true, "什么": true, "怎么": true,
	"为什么": true, "时候": true, "自己": true, "知道": true, "觉得": true, "感觉": true, "这样": true, "那样": true, "这个": true, "那个": true,
	"一个": true, "一下": true, "一点": true, "有点": true, "真的": true, "没有": true, "不是": true, "是吗": true, "好的": true, "可以": true,
	"行吧": true, "哈哈": true, "哈哈哈": true, "哈哈哈哈": true, "ok": true, "OK": true, "嗯嗯": true,
}

var noiseRe = []string{"http", "www", "com", "cn", "net", "org", "链接", "图片", "语音", "视频"}

// topWords 简单中文分词（双字 bigram + 停用词 + 噪声过滤）
// 为控制耗时，只统计最近 30000 条文本消息（词频近似，足够代表性）
func topWords(d *DB, chat string, isSelf int, n int) [][2]interface{} {
	rows, err := d.conn.Query("SELECT content FROM msgs WHERE name=? AND is_self=? AND type='文本' AND LENGTH(content) BETWEEN 2 AND 120 ORDER BY id DESC LIMIT 30000", chat, isSelf)
	if err != nil {
		return [][2]interface{}{}
	}
	defer rows.Close()
	freq := map[string]int64{}
	for rows.Next() {
		var c string
		rows.Scan(&c)
		if strings.ContainsAny(c, "http<>{}[]\\/") {
			continue
		}
		runes := []rune(c)
		for i := 0; i+1 < len(runes); i++ {
			w := string(runes[i : i+2])
			if stopWords[w] {
				continue
			}
			freq[w]++
		}
	}
	return topKV(freq, n)
}

// ---------------- 情感引擎 ----------------

// 正负情感词典
var posWords = []string{"哈哈", "开心", "开心", "高兴", "快乐", "棒", "牛", "厉害", "优秀", "爽", "顺利",
	"恭喜", "谢谢", "感谢", "晚安", "么么", "嘻嘻", "嘿嘿", "耶", "喜欢", "爱", "期待", "好", "可以",
	"行", "没问题", "完美", "漂亮", "加油", "真好", "太好了", "幸福", "温暖", "感动", "好笑", "笑死", "笑疯",
	"太强", "太牛", "神仙", "宝藏", "绝了", "nice", "good", "cool", "帅", "美", "可爱", "贴心", "靠谱"}
var negWords = []string{"烦", "累", "难受", "气", "烦死", "无语", "崩溃", "哭", "难", "亏", "糟", "讨厌",
	"伤心", "难过", "焦虑", "压力", "坑", "生气", "不行", "完了", "叹气", "烦人", "恶心", "讨厌", "失望",
	"绝望", "痛苦", "郁闷", "烦躁", "愤怒", "生气", "可恶", "糟糕", "失败", "倒霉", "太差", "差评",
	"伤心", "心碎", "难过", "想哭", "哭死", "烦死了", "无语死了", "气死", "气炸", "愁", "慌", "怕", "担心"}

// 大连理工七类情绪（简化词表）
var emotionDict = map[string][]string{
	"乐": {"哈哈", "开心", "高兴", "快乐", "笑", "乐", "嘻嘻", "嘿嘿", "耶", "爽", "喜", "欢喜", "愉快"},
	"好": {"好", "棒", "牛", "赞", "优秀", "完美", "漂亮", "喜欢", "爱", "可爱", "靠谱", "贴心", "温柔", "善良", "帅", "美"},
	"怒": {"气", "生气", "愤怒", "烦", "可恶", "火", "气死", "气炸", "怒", "恼", "恨", "讨厌"},
	"哀": {"哭", "难过", "伤心", "心碎", "痛苦", "悲哀", "悲伤", "想哭", "哭死", "绝望", "失落", "郁闷", "哀"},
	"惧": {"怕", "担心", "慌", "害怕", "恐惧", "焦虑", "紧张", "心虚", "吓", "惊"},
	"恶": {"恶心", "讨厌", "烦", "嫌弃", "厌恶", "反感", "差评", "垃圾", "拉胯", "菜", "弱", "烂"},
	"惊": {"哇", "天", "惊", "震惊", "离谱", "意外", "居然", "竟然", "卧槽", "我去", "woc", "服", "绝"},
}

// Sentiment 情感总览（多指标）
func (d *DB) Sentiment(chat string) map[string]interface{} {
	rows, err := d.conn.Query("SELECT is_self, ts, content FROM msgs WHERE name=? AND type='文本' AND LENGTH(content) BETWEEN 1 AND 150", chat)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()

	type rec struct {
		self bool
		ts   int64
		cont string
	}
	recs := []rec{}
	for rows.Next() {
		var self int
		var ts int64
		var c string
		rows.Scan(&self, &ts, &c)
		recs = append(recs, rec{self == 1, ts, c})
	}

	totalScore, n := 0, 0
	emotionCnt := map[string]int64{}
	posCnt, negCnt := 0, 0
	strongCnt := 0
	selfScore, selfN, otherScore, otherN := 0, 0, 0, 0
	emotionWords := map[string]int64{}
	monthScore := map[string]int64{}
	monthN := map[string]int64{}
	hourScore := make([]int64, 24)
	hourN := make([]int64, 24)

	for _, r := range recs {
		s := 0
		for _, w := range posWords {
			if strings.Contains(r.cont, w) {
				s++
			}
		}
		for _, w := range negWords {
			if strings.Contains(r.cont, w) {
				s--
			}
		}
		// 情绪归类
		for emo, words := range emotionDict {
			for _, w := range words {
				if strings.Contains(r.cont, w) {
					emotionCnt[emo]++
					if len(w) >= 2 && !stopWords[w] {
						emotionWords[w]++
					}
					break
				}
			}
		}
		if s > 0 {
			posCnt++
		} else if s < 0 {
			negCnt++
		}
		if s >= 2 || s <= -2 {
			strongCnt++
		}
		totalScore += s
		n++
		if r.self {
			selfScore += s
			selfN++
		} else {
			otherScore += s
			otherN++
		}
		t := r.ts
		month := tsMonth(t)
		monthScore[month] += int64(s)
		monthN[month]++
		h := tsHour(t)
		if h >= 0 && h < 24 {
			hourScore[h] += int64(s)
			hourN[h]++
		}
	}

	avg := 0.0
	if n > 0 {
		avg = float64(totalScore) / float64(n) * 40 // 映射 -100~100
		if avg > 100 {
			avg = 100
		}
		if avg < -100 {
			avg = -100
		}
	}

	// 情绪波动指数（std）
	variance := 0.0
	if n > 1 {
		mean := float64(totalScore) / float64(n)
		var sq float64
		for _, r := range recs {
			s := scoreText(r.cont)
			diff := float64(s) - mean
			sq += diff * diff
		}
		variance = sq / float64(n)
	}

	// 亲密度：自己消息占比 + 积极率 + 互发均衡度
	intimacy := 50.0
	if n > 0 {
		selfRatio := float64(selfN) / float64(n)
		posRatio := float64(posCnt) / float64(n)
		intimacy = 50 + (selfRatio-0.5)*60 + (posRatio-0.5)*80
		if intimacy < 0 {
			intimacy = 0
		}
		if intimacy > 100 {
			intimacy = 100
		}
	}

	// 情感突变点：相邻消息得分差
	type point struct {
		TS    int64 `json:"ts"`
		Score int   `json:"score"`
	}
	series := []point{}
	for _, r := range recs {
		series = append(series, point{r.ts, scoreText(r.cont)})
	}

	return map[string]interface{}{
		"chat":          chat,
		"sentiment":     round1(avg),
		"n":             n,
		"pos":           posCnt,
		"neg":           negCnt,
		"neutral":       n - posCnt - negCnt,
		"strong":        strongCnt,
		"self_score":    round1(avgOf(selfScore, selfN)),
		"other_score":   round1(avgOf(otherScore, otherN)),
		"self_n":        selfN,
		"other_n":       otherN,
		"emotions":      emotionCnt,
		"emotion_words": topKV(emotionWords, 25),
		"months":        zipped(monthScore, monthN),
		"hours":         avgByHour(hourScore, hourN),
		"volatility":    round1(variance),
		"intimacy":      round1(intimacy),
		"series":        series,
	}
}

// SentimentSeries 月度情感序列
func (d *DB) SentimentSeries(chat string) map[string]interface{} {
	s := d.Sentiment(chat)
	months := s["months"]
	return map[string]interface{}{"months": months, "chat": chat}
}

// SentimentHeatmap 日×小时情感热力图（近 30 天）
func (d *DB) SentimentHeatmap(chat string) map[string]interface{} {
	rows, err := d.conn.Query("SELECT ts, content FROM msgs WHERE name=? AND type='文本' AND LENGTH(content) BETWEEN 1 AND 150 AND ts >= ?",
		chat, now()-30*86400)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()
	type cell struct {
		Date string `json:"date"`
		Hour int    `json:"hour"`
		Sum  int64  `json:"sum"`
		N    int64  `json:"n"`
	}
	grid := map[string]*cell{}
	order := []string{}
	for rows.Next() {
		var ts int64
		var c string
		rows.Scan(&ts, &c)
		key := tsDate(ts) + "|" + itoa(tsHour(ts))
		c2, ok := grid[key]
		if !ok {
			c2 = &cell{Date: tsDate(ts), Hour: tsHour(ts)}
			grid[key] = c2
			order = append(order, key)
		}
		c2.Sum += int64(scoreText(c))
		c2.N++
	}
	out := []cell{}
	for _, k := range order {
		c := grid[k]
		val := 0.0
		if c.N > 0 {
			val = float64(c.Sum) / float64(c.N) * 40
		}
		out = append(out, cell{c.Date, c.Hour, int64(val), c.N})
	}
	return map[string]interface{}{"cells": out, "chat": chat}
}

// EmotionDistribution 七类情绪分布
func (d *DB) EmotionDistribution(chat string) map[string]interface{} {
	s := d.Sentiment(chat)
	return map[string]interface{}{"emotions": s["emotions"], "chat": chat}
}

// Search 关键词搜索
func (d *DB) Search(chat, kw string) map[string]interface{} {
	if kw == "" {
		return map[string]interface{}{"error": "empty query"}
	}
	rows, err := d.conn.Query("SELECT is_self, ts, content FROM msgs WHERE name=? AND type='文本' AND content LIKE ? ORDER BY id DESC LIMIT 100",
		chat, "%"+kw+"%")
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()
	type hit struct {
		Self    bool   `json:"self"`
		TS      int64  `json:"ts"`
		Content string `json:"content"`
	}
	hits := []hit{}
	for rows.Next() {
		var self int
		var ts int64
		var c string
		rows.Scan(&self, &ts, &c)
		hits = append(hits, hit{self == 1, ts, c})
	}
	return map[string]interface{}{"hits": hits, "count": len(hits)}
}

// SentimentMsgs 返回某会话积极/消极消息列表（最近消息优先，含内容/时间/说话方）
// kind: "pos" 积极（score>0）| "neg" 消极（score<0）
func (d *DB) SentimentMsgs(chat, kind string, limit int) map[string]interface{} {
	if kind != "pos" && kind != "neg" {
		kind = "pos"
	}
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := d.conn.Query("SELECT is_self, ts, content FROM msgs WHERE name=? AND type='文本' AND LENGTH(content) BETWEEN 1 AND 150 ORDER BY ts DESC LIMIT 8000", chat)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()
	type msg struct {
		Self    bool   `json:"self"`
		TS      int64  `json:"ts"`
		Content string `json:"content"`
		Score   int    `json:"score"`
	}
	out := []msg{}
	for rows.Next() {
		var self int
		var ts int64
		var c string
		rows.Scan(&self, &ts, &c)
		s := scoreText(c)
		if kind == "pos" && s > 0 {
			out = append(out, msg{self == 1, ts, c, s})
		} else if kind == "neg" && s < 0 {
			out = append(out, msg{self == 1, ts, c, s})
		}
		if len(out) >= limit {
			break
		}
	}
	return map[string]interface{}{"msgs": out, "kind": kind, "count": len(out), "chat": chat}
}

// ---------------- 工具 ----------------

func scoreText(c string) int {
	s := 0
	for _, w := range posWords {
		if strings.Contains(c, w) {
			s++
		}
	}
	for _, w := range negWords {
		if strings.Contains(c, w) {
			s--
		}
	}
	return s
}

func avgOf(sum, n int) float64 {
	if n == 0 {
		return 0
	}
	return float64(sum) / float64(n) * 40
}

func round1(f float64) float64 {
	return float64(int(f*10+0.5)) / 10
}

func tsMonth(ts int64) string {
	t := timeUnix(ts)
	return t.Format("2006-01")
}
func tsHour(ts int64) int {
	return timeUnix(ts).Hour()
}
func tsDate(ts int64) string {
	return timeUnix(ts).Format("2006-01-02")
}
func now() int64 { return timeUnixNow() }

func zipped(sum, n map[string]int64) map[string]interface{} {
	out := map[string]interface{}{}
	for k, v := range n {
		if v > 0 {
			out[k] = float64(sum[k]) / float64(v) * 40
		}
	}
	return out
}

func avgByHour(sum []int64, n []int64) []float64 {
	out := make([]float64, 24)
	for i := 0; i < 24; i++ {
		if n[i] > 0 {
			out[i] = float64(sum[i]) / float64(n[i]) * 40
		}
	}
	return out
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}
