package main

import (
	"strings"
	"time"
)

// Stats 单会话统计（单次流式扫描 + 内存聚合，避免多次 GROUP BY 全表扫描）
func (d *DB) Stats(chat string) map[string]interface{} {
	rows, err := d.conn.Query("SELECT is_self, ts, type, content FROM msgs WHERE name=? ORDER BY id", chat)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer rows.Close()

	var total, self, other int64
	var first, last int64
	typeCnt := map[string]int64{}
	hours := make([]int64, 24)
	week := make([]int64, 7)
	monthCnt := map[string]int64{}
	var sessions int64
	var lastTS int64
	var textCnt int64
	var lenSum int64
	emoCnt := map[string]int64{}
	selfFreq := map[string]int64{}
	otherFreq := map[string]int64{}
	selfTexts, otherTexts := 0, 0
	var bestDay string
	var bestDayCnt int64
	dayCnt := map[string]int64{}
	var latestNight string
	latestNightTS := int64(0)

	for rows.Next() {
		var isSelf int
		var ts int64
		var typ, content string
		rows.Scan(&isSelf, &ts, &typ, &content)
		total++
		if isSelf == 1 {
			self++
		} else {
			other++
		}
		if first == 0 || ts < first {
			first = ts
		}
		if ts > last {
			last = ts
		}
		typeCnt[typ]++

		t := timeUnix(ts)
		h := t.Hour()
		hours[h]++
		week[(int(t.Weekday())+6)%7]++
		m := t.Format("2006-01")
		monthCnt[m]++

		day := t.Format("2006-01-02")
		dayCnt[day]++
		if dayCnt[day] > bestDayCnt {
			bestDayCnt = dayCnt[day]
			bestDay = day
		}
		if (h >= 22 || h < 5) && ts > latestNightTS {
			latestNightTS = ts
			latestNight = t.Format("15:04")
		}

		if lastTS == 0 || ts-lastTS >= 1800 {
			sessions++
		}
		lastTS = ts

		if typ == "文本" {
			textCnt++
			lenSum += int64(len([]rune(content)))
			// 表情
			if strings.HasPrefix(content, "[") {
				if i := strings.IndexByte(content, ']'); i > 0 && i <= 12 {
					emoCnt[content[:i+1]]++
				}
			}
			// 词频（自己/对方各取最近 30000 条）
			if isSelf == 1 && selfTexts < 30000 {
				selfTexts++
				addBigrams(selfFreq, content)
			} else if isSelf == 0 && otherTexts < 30000 {
				otherTexts++
				addBigrams(otherFreq, content)
			}
		}
	}

	days := int64(0)
	if first > 0 && last > 0 {
		days = (last-first)/86400 + 1
		if days < 1 {
			days = 1
		}
	}
	avgLen := 0.0
	if textCnt > 0 {
		avgLen = float64(lenSum) / float64(textCnt)
	}
	dailyAvg := 0.0
	if days > 0 {
		dailyAvg = float64(total) / float64(days)
	}

	// 月度序列（近 24 个月）
	type MonthPoint struct {
		Month string `json:"month"`
		Count int64  `json:"count"`
	}
	months := []MonthPoint{}
	start := time.Now().AddDate(0, -24, 0).Format("2006-01")
	startAdded := false
	for i := 0; i < 25; i++ {
		mm := time.Now().AddDate(0, -24+i, 0).Format("2006-01")
		if mm == start {
			startAdded = true
		}
		if !startAdded {
			continue
		}
		months = append(months, MonthPoint{mm, monthCnt[mm]})
	}

	return map[string]interface{}{
		"chat":         chat,
		"total":        total,
		"self":         self,
		"other":        other,
		"first":        first,
		"last":         last,
		"active_days":  days,
		"daily_avg":    round1(dailyAvg),
		"types":        typeCnt,
		"hours":        hours,
		"week":         week,
		"months":       months,
		"sessions":     sessions,
		"reply_median": 0,
		"best_day":     bestDay,
		"best_day_cnt": bestDayCnt,
		"latest_night": latestNight,
		"avg_len":      round1(avgLen),
		"text_cnt":     textCnt,
		"self_words":   topKV(selfFreq, 40),
		"other_words":  topKV(otherFreq, 40),
		"emojis":       topKV(emoCnt, 20),
	}
}

func addBigrams(freq map[string]int64, s string) {
	if strings.ContainsAny(s, "http<>{}[]\\/") {
		return
	}
	runes := []rune(s)
	if len(runes) > 80 {
		runes = runes[:80]
	}
	for i := 0; i+1 < len(runes); i++ {
		w := string(runes[i : i+2])
		if stopWords[w] {
			continue
		}
		freq[w]++
	}
}

func atoi(s string) int {
	n := 0
	for _, c := range s {
		if c >= '0' && c <= '9' {
			n = n*10 + int(c-'0')
		}
	}
	return n
}

func median(a []int64) float64 {
	if len(a) == 0 {
		return 0
	}
	var s int64
	for _, v := range a {
		s += v
	}
	return float64(s) / float64(len(a))
}

func topKV(m map[string]int64, n int) [][2]interface{} {
	type kv struct {
		k string
		v int64
	}
	all := []kv{}
	for k, v := range m {
		all = append(all, kv{k, v})
	}
	// 简单排序（插入排序，n 通常<200）
	for i := 1; i < len(all); i++ {
		for j := i; j > 0 && all[j].v > all[j-1].v; j-- {
			all[j], all[j-1] = all[j-1], all[j]
		}
	}
	out := [][2]interface{}{}
	for i := 0; i < len(all) && i < n; i++ {
		out = append(out, [2]interface{}{all[i].k, all[i].v})
	}
	return out
}
