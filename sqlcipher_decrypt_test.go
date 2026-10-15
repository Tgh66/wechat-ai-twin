package main

// SQLCipher 解密器自测：构造 SQLCipher 4 加密库 → 解密往返 → 内容一致
// 仅测试用（_test.go 不进入 build）

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha512"
	"database/sql"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

// encryptSQLCipher4 反向实现（仅自测）：明文 sqlite → SQLCipher 4 加密文件
func encryptSQLCipher4(plain []byte, keyHex string, salt []byte) []byte {
	keyBytes, _ := hex.DecodeString(keyHex)
	dk := pbkdf2Key(keyBytes, salt, 256000, 32, sha512.New)
	pageSize, res := 4096, 48
	var out bytes.Buffer
	out.Write(salt)
	n := len(plain) / pageSize
	if len(plain)%pageSize != 0 {
		n++
	}
	for i := 0; i < n; i++ {
		off := i * pageSize
		dataLen := pageSize - res
		if i == 0 {
			off = 16
			dataLen = pageSize - 16 - res // 页 1 跳过 salt 后的 16 字节
		}
		data := plain[off : off+dataLen]
		iv := []byte{byte(i*13 + 1), byte(i*29 + 2), byte(i*37 + 3), byte(i*41 + 5),
			byte(i*17 + 7), byte(i*23 + 11), byte(i*31 + 13), byte(i*43 + 17),
			byte(i*3 + 19), byte(i*5 + 23), byte(i*7 + 29), byte(i*11 + 31),
			byte(i*19 + 37), byte(i*53 + 41), byte(i*59 + 43), byte(i*61 + 47)}
		block, _ := aes.NewCipher(dk)
		ct := make([]byte, len(data))
		cipher.NewCBCEncrypter(block, iv).CryptBlocks(ct, data)
		out.Write(ct)
		out.Write(iv)
		hm := hmac.New(sha512.New, dk)
		hm.Write(iv)
		hm.Write(ct)
		out.Write(hm.Sum(nil)[:res-16])
	}
	return out.Bytes()
}

func TestSQLCipher4RoundTrip(t *testing.T) {
	// 1) 创建明文 SQLite 库
	dir := t.TempDir()
	plainDB := filepath.Join(dir, "plain.db")
	conn, err := sql.Open("sqlite", plainDB)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conn.Exec("CREATE TABLE t(a TEXT, b INTEGER)"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		conn.Exec("INSERT INTO t VALUES(?,?)", "hello"+string(rune('0'+i)), i)
	}
	conn.Close()
	plain, err := os.ReadFile(plainDB)
	if err != nil {
		t.Fatal(err)
	}
	if len(plain) < 16 || !bytes.Equal(plain[:16], []byte(sqliteHeader)) {
		t.Fatalf("明文库头不对: %x", plain[:16])
	}
	// 2) 加密
	keyHex := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	salt := []byte("0123456789abcdef")
	enc := encryptSQLCipher4(plain, keyHex, salt)
	encDB := filepath.Join(dir, "enc.db")
	os.WriteFile(encDB, enc, 0644)

	// 3) 正确密钥 → 解密成功
	decDB := filepath.Join(dir, "dec.db")
	mode, err := decryptSQLCipherDB(encDB, keyHex, decDB)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}
	if mode != "SQLCipher4" {
		t.Fatalf("参数识别不对: %s", mode)
	}
	dec, _ := os.ReadFile(decDB)
	if !bytes.Equal(dec[:16], []byte(sqliteHeader)) {
		t.Fatalf("解密后头不对: %x", dec[:16])
	}
	// 数据区往返对比（跳过每页 reserve 48 字节；普通库 reserve=0，加密时每页尾 48 视为 reserve 丢弃）
	// 真实 SQLCipher 库每页为 usable 4048 + reserve 48，本测试验证解密器按该布局正确解析
	pageSize := 4096
	res := 48
	nPages := len(plain) / pageSize
	// 页 1: 数据区 16..4048；后续页: idx*4096 .. idx*4096+4048
	check := func(idx int) bool {
		var off, dataLen int
		if idx == 0 {
			off, dataLen = 16, pageSize-16-res
		} else {
			off, dataLen = idx*pageSize, pageSize-res
		}
		return bytes.Equal(plain[off:off+dataLen], dec[off:off+dataLen])
	}
	for i := 0; i < nPages; i++ {
		if !check(i) {
			t.Fatalf("页 %d 数据区不一致", i)
		}
	}
	t.Logf("往返成功: mode=%s pages=%d size=%d（数据区逐页一致）", mode, nPages, len(dec))

	// 4) 错误密钥 → 拒绝
	badDB := filepath.Join(dir, "bad.db")
	if _, err := decryptSQLCipherDB(encDB, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", badDB); err == nil {
		t.Fatal("错误密钥不应解密成功")
	}
	// 5) tryDecryptFirstPage 判断
	if _, _, ok := tryDecryptFirstPage(encDB, keyHex); !ok {
		t.Fatal("正确密钥应通过首页验证")
	}
	if _, _, ok := tryDecryptFirstPage(encDB, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); ok {
		t.Fatal("错误密钥不应通过首页验证")
	}
	t.Log("密钥验证逻辑正确")
}
