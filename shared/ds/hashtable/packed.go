package hashtable

import (
	"bytes"
	"encoding/binary"
)

// PackedHash: Lưu trữ siêu gọn trong 1 mảng byte (CPU Cache Friendly)
type PackedHash struct {
	data []byte
}

func NewPacked() *PackedHash {
	// Pre-alloc header (2 byte cho Count)
	d := make([]byte, 2, 64)
	return &PackedHash{data: d}
}

func (p *PackedHash) IsPacked() bool { return true }

func (p *PackedHash) count() int {
	return int(binary.LittleEndian.Uint16(p.data[0:2]))
}

func (p *PackedHash) incCount(delta int) {
	c := p.count() + delta
	binary.LittleEndian.PutUint16(p.data[0:2], uint16(c))
}

func (p *PackedHash) Len() int {
	return p.count()
}

func (p *PackedHash) SizeInBytes() int {
	return len(p.data)
}

// find scan tuyến tính (Hiệu năng cực cao với L1 Cache)
func (p *PackedHash) find(field string) (int, int, int, bool) {
	fieldBytes := []byte(field)
	pos := 2
	n := len(p.data)
	count := p.count()

	for i := 0; i < count; i++ {
		if pos >= n {
			break
		}
		kLen := int(p.data[pos])
		pos++

		match := false
		if kLen == len(field) {
			if bytes.Equal(p.data[pos:pos+kLen], fieldBytes) {
				match = true
			}
		}
		keyStart := pos - 1
		pos += kLen

		if pos+4 > n {
			break
		}
		vLen := int(binary.LittleEndian.Uint32(p.data[pos : pos+4]))
		pos += 4

		valStart := pos
		valLen := vLen
		pos += vLen

		if match {
			return keyStart, valStart, valLen, true
		}
	}
	return -1, -1, -1, false
}

func (p *PackedHash) Get(field string) ([]byte, bool) {
	if len(field) > 255 {
		return nil, false
	}
	_, vStart, vLen, found := p.find(field)
	if !found {
		return nil, false
	}
	out := make([]byte, vLen)
	copy(out, p.data[vStart:vStart+vLen])
	return out, true
}

func (p *PackedHash) Set(field string, value []byte) (HashObject, bool) {
	// [UPGRADE LOGIC]
	// Nếu vượt quá giới hạn, chuyển sang PPHT
	if len(field) > 255 || len(p.data)+len(field)+len(value) > MaxPackedSize || p.count() >= MaxPackedCount {
		ppht := p.ToPPHT() // Convert sang cấu trúc xịn
		ppht.Set(field, value)
		return ppht, true // Báo hiệu đã upgrade
	}

	// [FIX] Thay keyStart bằng _ vì không sử dụng
	_, _, _, found := p.find(field)

	if found {
		p.Delete(field)
	}

	newSize := len(p.data) + 1 + len(field) + 4 + len(value)
	if cap(p.data) < newSize {
		newData := make([]byte, len(p.data), newSize*2)
		copy(newData, p.data)
		p.data = newData
	}

	p.data = append(p.data, byte(len(field)))
	p.data = append(p.data, field...)

	lenBuf := make([]byte, 4)
	binary.LittleEndian.PutUint32(lenBuf, uint32(len(value)))
	p.data = append(p.data, lenBuf...)
	p.data = append(p.data, value...)

	if !found {
		p.incCount(1)
	}

	return p, false
}

func (p *PackedHash) Delete(field string) (bool, int) {
	if len(field) > 255 {
		return false, p.count()
	}
	keyStartHeader, _, vLen, found := p.find(field)
	if !found {
		return false, p.count()
	}

	totalLen := 1 + len(field) + 4 + vLen
	endOfEntry := keyStartHeader + totalLen

	copy(p.data[keyStartHeader:], p.data[endOfEntry:])
	p.data = p.data[:len(p.data)-totalLen]

	p.incCount(-1)
	return true, p.count()
}

func (p *PackedHash) GetAll() map[string][]byte {
	m := make(map[string][]byte)
	pos := 2
	n := len(p.data)
	count := p.count()

	for i := 0; i < count; i++ {
		if pos >= n {
			break
		}
		kLen := int(p.data[pos])
		pos++
		key := string(p.data[pos : pos+kLen])
		pos += kLen

		vLen := int(binary.LittleEndian.Uint32(p.data[pos : pos+4]))
		pos += 4
		val := make([]byte, vLen)
		copy(val, p.data[pos:pos+vLen])
		pos += vLen

		m[key] = val
	}
	return m
}

// ToPPHT chuyển đổi từ PackedHash sang PPHT (Upgrade)
func (p *PackedHash) ToPPHT() *PPHT {
	// Khởi tạo PPHT với size dự kiến gấp đôi hiện tại để tránh rehash sớm
	// 4 shards là đủ cho object vừa mới upgrade
	ht := New(p.Len()*2, 4)

	// Scan và insert vào PPHT
	all := p.GetAll()
	for k, v := range all {
		ht.Set(k, v)
	}
	return ht
}
