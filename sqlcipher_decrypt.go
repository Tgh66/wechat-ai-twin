package main

// 纯 Go SQLCipher 3/4 数据库解密器（用于微信聊天记录库）
// 不依赖任何 C 库：PBKDF2 派生 + AES-256-CBC 逐页解密 → 输出明文 SQLite 文件，
// 再用内置 modernc sqlite 查询提取消息。

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha512"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"os"
	"strings"
)

const sqliteHeader = "SQLite format 3\x00"

// looksLikeSQLitePage0 判断解密出的页 0 数据区是否为有效 SQLite 结构
// SQLCipher 库页 0 数据区 = 原明文 SQLite 文件 offset 16 起（前 16 字节 header 被 salt 替换）：
//   数据区 offset 0-1 = page size（大端，原 header 16-17）
//   数据区 offset 2-3 = format version（01 01，原 header 18-19）
//   数据区 offset 4   = reserved（原 header 20）
//   数据区 offset 84 起 = btree 页 1 头（原文件 offset 100：0x0D=叶子表页 / 0x05=内部表页 等）
// 组合校验使随机密钥误判概率可忽略
func looksLikeSQLitePage0(pl []byte) bool {
	if len(pl) < 120 {
		return false
	}
	ps := uint16(pl[0])<<8 | uint16(pl[1])
	// SQLite 页大小: 512-65536；65536 特殊编码为 1（0x0001）
	if ps != 512 && ps != 1024 && ps != 2048 && ps != 4096 && ps != 8192 && ps != 16384 && ps != 32768 && ps != 1 {
		return false
	}
	if pl[2] != 0x01 || pl[3] != 0x01 {
		return false
	}
	if pl[4] != 0x00 && pl[4] != 0x01 && pl[4] != 0x20 {
		return false // reserved 区通常 0
	}
	t := pl[84]
	return t == 0x0d || t == 0x05 || t == 0x02 || t == 0x0a
}

// pbkdf2 标准 PBKDF2（Go 标准库未内置，手写）
func pbkdf2Key(password, salt []byte, iter, keyLen int, h func() hash.Hash) []byte {
	prf := hmac.New(h, password)
	hashLen := prf.Size()
	numBlocks := (keyLen + hashLen - 1) / hashLen
	out := make([]byte, 0, numBlocks*hashLen)
	var u []byte
	for block := 1; len(out) < keyLen; block++ {
		prf.Reset()
		prf.Write(salt)
		buf := make([]byte, 4)
		buf[0] = byte(block >> 24)
		buf[1] = byte(block >> 16)
		buf[2] = byte(block >> 8)
		buf[3] = byte(block)
		prf.Write(buf)
		u = prf.Sum(nil)
		t := append([]byte(nil), u...)
		for i := 1; i < iter; i++ {
			prf.Reset()
			prf.Write(u)
			u = prf.Sum(nil)
			for j := range t {
				t[j] ^= u[j]
			}
		}
		out = append(out, t...)
	}
	return out[:keyLen]
}

// tryDecryptFirstPage 尝试用 keyHex 解密 db 第一页，返回明文前 16 字节与参数组合名
// 覆盖 SQLCipher 4（SHA512/256000/48）与 3（SHA1/64000/32、SHA1/64000/36）常用参数
func tryDecryptFirstPage(dbPath, keyHex string) ([]byte, string, bool) {
	raw, err := os.ReadFile(dbPath)
	if err != nil || len(raw) < 64 {
		return nil, "", false
	}
	keyBytes, err := hex.DecodeString(keyHex)
	if err != nil || len(keyBytes) == 0 {
		return nil, "", false
	}
	// salt = 文件前 16 字节
	salt := raw[:16]
	combos := []struct {
		name string
		iter int
		res  int
		hfn  func() hash.Hash
	}{
		{"v4-sha512-256k", 256000, 48, sha512.New},
		{"v4-sha512-64k", 64000, 48, sha512.New},
		{"v3-sha1-64k-r32", 64000, 32, sha1.New},
		{"v3-sha1-1k-r32", 1000, 32, sha1.New},
	}
	for _, c := range combos {
		dk := pbkdf2Key(keyBytes, salt, c.iter, 32, c.hfn)
		if pl := decryptPage(raw, 0, dk, c.res); pl != nil && looksLikeSQLitePage0(pl) {
			return pl[:120], c.name, true
		}
	}
	return nil, "", false
}

// decryptPage 解密第 idx 页（idx=0 为第一页）
// SQLite 页 1 从文件 offset 0 开始（前 100 字节为数据库头）；SQLCipher 把页 1 前 16 字节替换为 salt：
//   页 0: 数据区 = file[16 : 16+4096-16-res]，reserve 紧随其后
//   页 i>0: 数据区 = file[idx*4096 : idx*4096+4096-res]
// res 为每页预留字节（SQLCipher 3/4 的 IV+HMAC）
func decryptPage(file []byte, idx int, key []byte, res int) []byte {
	pageSize := 4096
	var off int
	if idx == 0 {
		off = 16
	} else {
		off = idx * pageSize
	}
	if off < 0 || off+pageSize > len(file) {
		return nil
	}
	dataLen := pageSize - res
	if idx == 0 {
		dataLen = pageSize - 16 - res
	}
	if dataLen%16 != 0 || dataLen < 16 {
		return nil
	}
	ciphertext := file[off : off+dataLen]
	// reserve 布局: [IV(16)][HMAC(res-16)]，IV 在数据区之后的前 16 字节
	iv := file[off+dataLen : off+dataLen+16]
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil
	}
	plain := make([]byte, len(ciphertext))
	cipher.NewCBCDecrypter(block, iv).CryptBlocks(plain, ciphertext)
	return plain
}

// decryptSQLCipherDB 完整解密 db 为明文 SQLite 文件
// 自动探测 SQLCipher 4/3 参数（reserve 与 KDF）；成功写 outPath 并返回参数名
func decryptSQLCipherDB(dbPath, keyHex, outPath string) (string, error) {
	raw, err := os.ReadFile(dbPath)
	if err != nil {
		return "", err
	}
	if len(raw) < 64 {
		return "", errors.New("db 文件过小，可能不是有效数据库")
	}
	keyBytes, err := hex.DecodeString(keyHex)
	if err != nil || len(keyBytes) == 0 {
		return "", errors.New("无效密钥")
	}
	salt := raw[:16]

	// 尝试参数组合（先 4 后 3）
	combos := []struct {
		name string
		iter int
		res  int
		hfn  func() hash.Hash
	}{
		{"SQLCipher4", 256000, 48, sha512.New},
		{"SQLCipher4-64k", 64000, 48, sha512.New},
		{"SQLCipher3", 64000, 32, sha1.New},
		{"SQLCipher3-1k", 1000, 32, sha1.New},
	}
	lastErr := errors.New("所有参数组合均无法解密（密钥可能不匹配）")
	for _, c := range combos {
		dk := pbkdf2Key(keyBytes, salt, c.iter, 32, c.hfn)
		// 先验证第一页
		if pl := decryptPage(raw, 0, dk, c.res); pl == nil || !looksLikeSQLitePage0(pl) {
			continue
		}
		// 全部页解密（SQLite 文件 = 页数 × 4096，页 1 的 salt 由解密后补回的魔法串占位）
		pageSize := 4096
		nPages := len(raw) / pageSize
		if len(raw)%pageSize != 0 {
			nPages++
		}
		var buf bytes.Buffer
		buf.Grow(nPages * pageSize)
		buf.WriteString(sqliteHeader) // 补回被 salt 替换的数据库头前 16 字节
		for i := 0; i < nPages; i++ {
			pl := decryptPage(raw, i, dk, c.res)
			if pl == nil {
				return "", errors.New("解密页 " + fmt.Sprintf("%d", i) + " 失败")
			}
			buf.Write(pl)
			buf.Write(make([]byte, c.res)) // reserve 区置零
		}
		if err := os.WriteFile(outPath, buf.Bytes(), 0644); err != nil {
			return "", err
		}
		return c.name, nil
	}
	return "", lastErr
}

// guessKeyHexCandidates 从十六进制候选串中过滤出 64 字符 hex（32 字节 raw key）
func guessKeyHexCandidates(cands []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, c := range cands {
		c = strings.TrimSpace(c)
		if len(c) == 64 {
			if _, err := hex.DecodeString(c); err == nil {
				if !seen[c] {
					seen[c] = true
					out = append(out, c)
				}
			}
		}
	}
	return out
}
