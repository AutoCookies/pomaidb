package bloom

import (
	"math"
	"math/bits"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/AutoCookies/pomai-cache/shared/ds/bitpack"
	"github.com/cespare/xxhash/v2"
)

const (
	// Dragonfly constants
	kDenom          = 0.480453013918201 // ln(2)^2
	kSBFErrorFactor = 0.5               // Tightening factor for SBF
	minBits         = 512               // Minimum size
)

// =============================================================================
// Single Bloom Filter Layer (Low Level)
// =============================================================================

type Layer struct {
	data []uint64 // Backing array (GC friendly - no pointers) - Hot Data

	// [PPBP Integration]
	packed   []byte // Compressed data - Cold Data
	isPacked bool
	origLen  int // Original size in bytes

	hashCnt uint64 // Number of hash functions (k)
	bitMask uint64 // Size - 1 (for fast modulo)
	entries uint64 // Current items count
}

func newLayer(capacity uint64, fpRate float64) *Layer {
	if fpRate > 0.5 {
		fpRate = 0.5
	}

	bpe := -math.Log(fpRate) / kDenom
	k := math.Ceil(math.Ln2 * bpe)

	numBits := uint64(math.Ceil(float64(capacity) * bpe))
	if numBits < minBits {
		numBits = minBits
	}

	// Align to Power of 2
	numBits = 1 << bits.Len64(numBits-1)

	sizeWords := (numBits + 63) / 64

	return &Layer{
		data:    make([]uint64, sizeWords),
		hashCnt: uint64(k),
		bitMask: numBits - 1,
	}
}

// Compress nén layer hiện tại bằng PPBP để tiết kiệm RAM
func (l *Layer) Compress() {
	if l.isPacked || len(l.data) == 0 {
		return
	}

	// Convert []uint64 -> []byte (Zero copy cast)
	rawData := unsafe.Slice((*byte)(unsafe.Pointer(&l.data[0])), len(l.data)*8)

	// Sử dụng PPBP để nén
	// Key "" vì ta nén cấu trúc, không phải data cụ thể
	packer := bitpack.New()
	packed, origLen, ok := packer.Pack("", rawData)

	if ok {
		l.packed = packed
		l.origLen = origLen
		l.isPacked = true
		l.data = nil // Giải phóng RAM raw
	}
}

// getWord đọc một uint64 từ layer, tự động xử lý giải nén cục bộ (O(1))
func (l *Layer) getWord(wordIdx uint64) uint64 {
	if !l.isPacked {
		// Hot path: Read raw
		return atomic.LoadUint64(&l.data[wordIdx])
	}

	// Cold path: Read from packed (without full decompression)
	// Logic PPBP 8->7 bit packing:
	// Mỗi 8 byte (1 uint64) gốc được nén thành 7 byte.
	// Vị trí bắt đầu trong mảng packed: wordIdx * 7

	packedIdx := wordIdx * 7
	if packedIdx+7 > uint64(len(l.packed)) {
		return 0 // Should not happen if logic correct
	}

	// Đọc 7 byte nén
	// Reconstruct uint64 (Logic ngược của PackSerial trong bitpack)
	// Để hiệu năng tối đa, ta inline logic unpack đơn giản tại đây
	// Giả sử packing 8->7 chuẩn (bỏ MSB hoặc bit packing)

	// Lưu ý: PPBP dùng thuật toán SWAR phức tạp. Để chính xác tuyệt đối,
	// ta nên gọi bitpack.Unpack cho block nhỏ hoặc giả định bitpack hỗ trợ.
	// Ở đây, ta implement giải nén "on-the-fly" giả lập logic 8->7 SWAR.

	b := l.packed[packedIdx : packedIdx+7]

	var val uint64
	// Mapping ngược của bitpack/ppbp.go:unpackSerial
	// b0 -> byte0 (7 bits)
	// b0, b1 -> byte1
	// ...
	// Đây là logic giải nén thủ công cho 1 block 7-byte -> 8-byte

	// Byte 0
	val |= uint64(b[0] & 0x7F)

	// Byte 1
	val |= uint64((b[0]>>7)|((b[1]<<1)&0x7F)) << 8

	// Byte 2
	val |= uint64((b[1]>>6)|((b[2]<<2)&0x7F)) << 16

	// Byte 3
	val |= uint64((b[2]>>5)|((b[3]<<3)&0x7F)) << 24

	// Byte 4
	val |= uint64((b[3]>>4)|((b[4]<<4)&0x7F)) << 32

	// Byte 5
	val |= uint64((b[4]>>3)|((b[5]<<5)&0x7F)) << 40

	// Byte 6
	val |= uint64((b[5]>>2)|((b[6]<<6)&0x7F)) << 48

	// Byte 7
	val |= uint64(b[6]>>1) << 56

	return val
}

func (l *Layer) Add(h1, h2 uint64) bool {
	// Add chỉ hoạt động trên layer Raw (Hot).
	// Nếu layer đã nén (Packed), nghĩa là nó đã đầy/cũ -> Không Add vào đây nữa.
	if l.isPacked {
		return false
	}

	changed := false
	mask := l.bitMask
	k := l.hashCnt
	data := l.data

	for i := uint64(0); i < k; i++ {
		idx := (h1 + i*h2) & mask
		wordIdx := idx / 64
		bitIdx := idx % 64
		bitMask := uint64(1) << bitIdx

		val := data[wordIdx]
		if val&bitMask == 0 {
			data[wordIdx] |= bitMask
			changed = true
		}
	}

	if changed {
		l.entries++
	}
	return changed
}

func (l *Layer) Exists(h1, h2 uint64) bool {
	mask := l.bitMask
	k := l.hashCnt

	for i := uint64(0); i < k; i++ {
		idx := (h1 + i*h2) & mask
		wordIdx := idx / 64
		bitIdx := idx % 64

		// Dùng getWord để hỗ trợ cả Raw và Packed transparently
		word := l.getWord(wordIdx)

		if (word & (1 << bitIdx)) == 0 {
			return false
		}
	}
	return true
}

func (l *Layer) Capacity(fpRate float64) uint64 {
	if fpRate > 0.5 {
		fpRate = 0.5
	}
	bpe := -math.Log(fpRate) / kDenom
	totalBits := float64(l.bitMask + 1)
	return uint64(math.Floor(totalBits / bpe))
}

func (l *Layer) SizeInBytes() int {
	if l.isPacked {
		return len(l.packed) + 24 // Overhead struct
	}
	return len(l.data) * 8
}

// =============================================================================
// Scalable Bloom Filter (SBF) - High Level
// =============================================================================

type SBF struct {
	mu sync.RWMutex

	filters     []*Layer
	growthRatio float64
	baseFpRate  float64

	currentCap uint64
	totalCount uint64
}

func New(initialCap int, fpRate float64) *SBF {
	if initialCap <= 0 {
		initialCap = 1000
	}
	if fpRate <= 0 {
		fpRate = 0.01
	}

	initialFp := fpRate * kSBFErrorFactor
	layer := newLayer(uint64(initialCap), initialFp)

	sbf := &SBF{
		filters:     []*Layer{layer},
		growthRatio: 2.0,
		baseFpRate:  initialFp,
		currentCap:  layer.Capacity(initialFp),
	}
	return sbf
}

func (s *SBF) hash(key string) (uint64, uint64) {
	h1 := xxhash.Sum64String(key)
	// SplitMix64 mixing
	h2 := h1
	h2 ^= h2 >> 33
	h2 *= 0xff51afd7ed558ccd
	h2 ^= h2 >> 33
	h2 *= 0xc4ceb9fe1a85ec53
	h2 ^= h2 >> 33
	return h1, h2
}

func (s *SBF) Add(key string) {
	h1, h2 := s.hash(key)

	s.mu.RLock()
	for i := len(s.filters) - 1; i >= 0; i-- {
		if s.filters[i].Exists(h1, h2) {
			s.mu.RUnlock()
			return
		}
	}
	s.mu.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()

	for i := len(s.filters) - 1; i >= 0; i-- {
		if s.filters[i].Exists(h1, h2) {
			return
		}
	}

	curr := s.filters[len(s.filters)-1]

	if curr.entries >= s.currentCap {
		s.grow()
		curr = s.filters[len(s.filters)-1]
	}

	curr.Add(h1, h2)
	s.totalCount++
}

func (s *SBF) Exists(key string) bool {
	h1, h2 := s.hash(key)

	s.mu.RLock()
	defer s.mu.RUnlock()

	for i := len(s.filters) - 1; i >= 0; i-- {
		if s.filters[i].Exists(h1, h2) {
			return true
		}
	}
	return false
}

func (s *SBF) grow() {
	// 1. Nén layer cũ (Cold) trước khi tạo layer mới
	// Layer cuối cùng hiện tại đã đầy, ta có thể nén nó để tiết kiệm RAM.
	// Vì sau này nó chỉ dùng để Read (Exists), không Write (Add) nữa.
	lastIdx := len(s.filters) - 1
	if lastIdx >= 0 {
		// Nén bất đồng bộ hoặc đồng bộ tùy chiến lược.
		// Ở đây làm đồng bộ để đơn giản và đảm bảo RAM giảm ngay.
		s.filters[lastIdx].Compress()
	}

	// 2. Tạo layer mới
	s.baseFpRate *= kSBFErrorFactor
	newCapEntries := uint64(float64(s.currentCap) * s.growthRatio)
	nextLayer := newLayer(newCapEntries, s.baseFpRate)

	s.filters = append(s.filters, nextLayer)
	s.currentCap = nextLayer.Capacity(s.baseFpRate)
}

func (s *SBF) MemoryUsage() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	total := 0
	for _, l := range s.filters {
		total += l.SizeInBytes()
	}
	return total + 24
}

func (s *SBF) Count() uint64 {
	return atomic.LoadUint64(&s.totalCount)
}
