package bitpack

import (
	"encoding/binary"
	"sync"
)

const (
	// Cấu hình hạt lựu
	WordSize = 8 // 64-bit word = 8 bytes

	// Ngưỡng nén
	RatioBase       = 7 // 8 bytes -> 7 bytes (Standard ASCII)
	RatioAggressive = 6 // 8 bytes -> 6 bytes (Base64-like / Dense integers)
)

// PPE Interface giả lập (Bạn sẽ hook AI thật vào đây)
type Predictor interface {
	IsHot(key string) bool
}

type DefaultPredictor struct{}

func (d DefaultPredictor) IsHot(key string) bool { return true }

// PPBP: Pomai Pomegranate BitPacker
type PPBP struct {
	predictor Predictor
}

func New() *PPBP {
	return &PPBP{predictor: DefaultPredictor{}}
}

// -----------------------------------------------------------------------------
// Core Packing Logic (SWAR - SIMD Within A Register)
// -----------------------------------------------------------------------------

// Pack nén dữ liệu đầu vào.
// Nếu hot -> Nén nhẹ (7-bit) hoặc giữ nguyên (nếu không phải ASCII).
// Nếu cold -> Nén mạnh (6-bit).
// Trả về: (packed_data, original_len, is_compressed)
func (b *PPBP) Pack(key string, data []byte) ([]byte, int, bool) {
	n := len(data)
	if n == 0 {
		return nil, 0, false
	}

	// 1. Quyết định chiến thuật dựa trên PPE và Entropy
	isHot := b.predictor.IsHot(key)

	// Nếu dữ liệu quá ngắn, overhead nén không đáng -> Skip
	if n < 16 {
		return data, n, false
	}

	// Kiểm tra sơ bộ xem có nén được không (Nếu byte > 127 thì không dùng 7-bit được)
	// Ta dùng hàm check nhanh 64 byte đầu
	if !canCompressASCII(data) {
		return data, n, false // Fallback to Raw
	}

	ratio := RatioBase
	if !isHot {
		// Nếu Cold data và dữ liệu phù hợp (vd: chỉ gồm chữ cái/số), dùng 6-bit
		// Ở đây để an toàn ta mặc định dùng 7-bit cho mọi case ASCII,
		// logic 6-bit phức tạp hơn cần bảng map riêng.
		ratio = RatioBase
	}

	// 2. Tính kích thước đích
	// Công thức: ceil(n * 7 / 8)
	outLen := (n*ratio + 7) / 8
	out := make([]byte, outLen)

	// 3. Parallel Packing (Mô phỏng SIMD bằng Goroutines cho chunks lớn)
	// Với data nhỏ (< 64KB), chạy tuần tự nhanh hơn do overhead của Go Scheduler.
	if n > 65536 {
		packParallel(data, out, n)
	} else {
		packSerial(data, out, n)
	}

	return out, n, true
}

// Unpack giải nén về dạng gốc
func (b *PPBP) Unpack(packed []byte, originalLen int) []byte {
	if originalLen == 0 {
		return nil
	}
	out := make([]byte, originalLen)

	if len(packed) >= 8192 {
		unpackParallel(packed, out, originalLen)
	} else {
		unpackSerial(packed, out, originalLen)
	}
	return out
}

// -----------------------------------------------------------------------------
// Low-Level Optimization (Zero-Allocation & Unrolling)
// -----------------------------------------------------------------------------

// canCompressASCII kiểm tra nhanh xem data có bit cao (MSB) nào được set không
func canCompressASCII(data []byte) bool {
	// Check 8 bytes at a time
	limit := len(data) - (len(data) % 8)
	for i := 0; i < limit; i += 8 {
		val := binary.LittleEndian.Uint64(data[i:])
		if (val & 0x8080808080808080) != 0 {
			return false
		}
	}
	// Check tail
	for i := limit; i < len(data); i++ {
		if data[i] > 127 {
			return false
		}
	}
	return true
}

// packSerial: Nén 8 bytes -> 7 bytes sử dụng thanh ghi 64-bit
func packSerial(src []byte, dst []byte, n int) {
	srcIdx := 0
	dstIdx := 0

	// Main loop: Xử lý từng khối 8 byte
	limit := n - (n % 8)
	for srcIdx < limit {
		// Load 8 bytes vào 1 thanh ghi uint64
		// Giả sử input: 0aaaaaaa 0bbbbbbb 0ccccccc ...
		val := binary.LittleEndian.Uint64(src[srcIdx:])

		// Bit Manipulation Magic (Pack 8x8 -> 7x8 in register)
		// Bước này đẩy các bit có nghĩa lại gần nhau, loại bỏ bit 0 ở đầu mỗi byte
		// Note: Logic này cần mapping chính xác little endian.
		// Để đơn giản và chính xác trong Go thuần, ta dùng cách thủ công unrolled sẽ nhanh hơn bit magic phức tạp.

		// Byte 0: lấy 7 bit
		dst[dstIdx] = byte(val) | byte(val>>8)<<7
		// Byte 1: lấy 6 bit từ byte 1, 2 bit từ byte 2
		dst[dstIdx+1] = byte(val>>9) | byte(val>>16)<<6
		// Byte 2: lấy 5 bit byte 2, 3 bit byte 3
		dst[dstIdx+2] = byte(val>>18) | byte(val>>24)<<5
		// Byte 3: lấy 4 bit byte 3, 4 bit byte 4
		dst[dstIdx+3] = byte(val>>27) | byte(val>>32)<<4
		// Byte 4: lấy 3 bit byte 4, 5 bit byte 5
		dst[dstIdx+4] = byte(val>>36) | byte(val>>40)<<3
		// Byte 5: lấy 2 bit byte 5, 6 bit byte 6
		dst[dstIdx+5] = byte(val>>45) | byte(val>>48)<<2
		// Byte 6: lấy 1 bit byte 6, 7 bit byte 7
		dst[dstIdx+6] = byte(val>>54) | byte(val>>56)<<1

		srcIdx += 8
		dstIdx += 7
	}

	// Handle tail (những byte lẻ cuối cùng)
	// Đơn giản hóa: Copy phần dư hoặc dùng logic nén bit lẻ (phức tạp)
	// Ở đây ta dùng logic simple pack cho phần đuôi
	packTail(src[srcIdx:], dst[dstIdx:])
}

func unpackSerial(src []byte, dst []byte, n int) {
	srcIdx := 0
	dstIdx := 0
	limit := n - (n % 8)

	// Mỗi lần loop tạo ra 8 bytes output từ 7 bytes input
	for dstIdx < limit {
		// Đọc 8 bytes (thực ra chỉ cần 7 bytes nhưng đọc 8 cho an toàn alignment nếu buffer đủ)
		// Cần đảm bảo src còn đủ 8 byte hoặc handle edge case.
		// Ở đây giả định padding an toàn.

		// Reconstruct
		// Logic ngược lại của Pack
		b0 := src[srcIdx]
		b1 := src[srcIdx+1]
		b2 := src[srcIdx+2]
		b3 := src[srcIdx+3]
		b4 := src[srcIdx+4]
		b5 := src[srcIdx+5]
		b6 := src[srcIdx+6]

		dst[dstIdx] = b0 & 0x7F
		dst[dstIdx+1] = (b0 >> 7) | ((b1 << 1) & 0x7F)
		dst[dstIdx+2] = (b1 >> 6) | ((b2 << 2) & 0x7F)
		dst[dstIdx+3] = (b2 >> 5) | ((b3 << 3) & 0x7F)
		dst[dstIdx+4] = (b3 >> 4) | ((b4 << 4) & 0x7F)
		dst[dstIdx+5] = (b4 >> 3) | ((b5 << 5) & 0x7F)
		dst[dstIdx+6] = (b5 >> 2) | ((b6 << 6) & 0x7F)
		dst[dstIdx+7] = (b6 >> 1)

		srcIdx += 7
		dstIdx += 8
	}

	unpackTail(src[srcIdx:], dst[dstIdx:])
}

// Helpers cho tail (Simple implementation)
func packTail(src, dst []byte) {
	// Implementation simplified: just copy for tail or proper bit shifting loop
	// Để đảm bảo data integrity, ta có thể copy nguyên byte cho phần dư
	// Hoặc implement bit writer.
	// POMAI Fast Path: Copy tail bytes (chấp nhận lãng phí vài bit ở đuôi)
	copy(dst, src)
}

func unpackTail(src, dst []byte) {
	copy(dst, src)
}

// Parallel Helpers
func packParallel(src, dst []byte, n int) {
	var wg sync.WaitGroup
	chunks := 4
	chunkSize := n / chunks
	// Align to 8
	chunkSize = (chunkSize / 8) * 8

	for i := 0; i < chunks; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize
		if i == chunks-1 {
			end = n
		}

		// Tính offset đích: (start / 8) * 7
		dstStart := (start / 8) * 7

		go func(s, e, dStart int) {
			defer wg.Done()
			// Cần slice chính xác để tránh race condition?
			// Go slices truy cập index khác nhau là safe.
			packSerial(src[s:e], dst[dStart:], e-s)
		}(start, end, dstStart)
	}
	wg.Wait()
}

func unpackParallel(src, dst []byte, n int) {
	var wg sync.WaitGroup
	chunks := 4
	chunkSize := n / chunks // Output bytes
	chunkSize = (chunkSize / 8) * 8

	for i := 0; i < chunks; i++ {
		wg.Add(1)
		dstStart := i * chunkSize
		dstEnd := dstStart + chunkSize
		if i == chunks-1 {
			dstEnd = n
		}

		srcStart := (dstStart / 8) * 7

		go func(sStart, dStart, dEnd int) {
			defer wg.Done()
			unpackSerial(src[sStart:], dst[dStart:], dEnd-dStart)
		}(srcStart, dstStart, dstEnd)
	}
	wg.Wait()
}
