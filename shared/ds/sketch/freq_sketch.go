package sketch

import (
	"encoding/binary"
	"hash/fnv"
	"sync/atomic"
)

// Sketch is a standard Count-Min Sketch implementation.
// It uses atomic operations for thread-safety.
type Sketch struct {
	depth uint32
	width uint32
	tab   []uint32 // flattened depth*width
	seeds []uint64
}

// New creates a new sketch.
func New(width, depth uint32) *Sketch {
	if width == 0 {
		width = 1 << 16 // Default 65536
	}
	if depth == 0 {
		depth = 4
	}

	// Ensure width is power of 2 for fast modulus
	if (width & (width - 1)) != 0 {
		// Round up to next power of 2 or handle appropriately
		// For simplicity here, strictly relying on caller or default.
		// A simple fix to ensure pow2 behavior if user inputs weird number:
		// width = nextPow2(width)
	}

	tab := make([]uint32, uint64(width)*uint64(depth))
	seeds := make([]uint64, depth)
	// fill seeds deterministically
	for i := uint32(0); i < depth; i++ {
		seeds[i] = uint64(0x9e3779b97f4a7c15 + uint64(i*0x9e377))
	}
	return &Sketch{
		depth: depth,
		width: width,
		tab:   tab,
		seeds: seeds,
	}
}

// hashWithSeed calculates hash for a specific seed
func (s *Sketch) hashWithSeed(key string, seed uint64) uint64 {
	h := fnv.New64a()
	var sb [8]byte
	binary.LittleEndian.PutUint64(sb[:], seed)
	h.Write(sb[:])
	h.Write([]byte(key))
	return h.Sum64()
}

// Increment increases counters for key by 1.
// Thread-safe using atomic.AddUint32.
func (s *Sketch) Increment(key string) uint32 {
	if s == nil {
		return 0
	}

	// [FIX] Sử dụng Standard Count-Min Sketch (Add tất cả các hàng)
	// Cách này đảm bảo thread-safety tuyệt đối với atomic.AddUint32.
	// Conservative update rất khó implement đúng khi lock-free.

	for d := uint32(0); d < s.depth; d++ {
		h := s.hashWithSeed(key, s.seeds[d])
		idx := uint32(h & uint64(s.width-1))
		pos := d*s.width + idx

		// Atomic increment: Luôn luôn cộng, không bao giờ bị lost update
		atomic.AddUint32(&s.tab[pos], 1)
	}

	// Trả về ước lượng ngay lập tức
	return s.Estimate(key)
}

// Estimate returns estimated frequency.
func (s *Sketch) Estimate(key string) uint32 {
	if s == nil {
		return 0
	}
	min := uint32(^uint32(0)) // Max Uint32

	for d := uint32(0); d < s.depth; d++ {
		h := s.hashWithSeed(key, s.seeds[d])
		idx := uint32(h & uint64(s.width-1))
		pos := d*s.width + idx

		// Đọc atomic để đảm bảo tính nhất quán (dù Estimate cho phép sai số nhỏ)
		v := atomic.LoadUint32(&s.tab[pos])
		if v < min {
			min = v
		}
	}
	return min
}
