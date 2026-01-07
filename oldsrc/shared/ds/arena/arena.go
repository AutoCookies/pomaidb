package arena

import (
	"sync"
	"sync/atomic"
	"time"
)

const (
	// Cấu hình hạt lựu
	SlabSize     = 64 * 1024 // 64KB per slab (nhỏ gọn, vừa L2 cache)
	MaxSmallSize = 4096      // Object lớn hơn mức này sẽ dùng Go Alloc thường
)

// PPE Interface giả lập
type Predictor interface {
	IsCold(slabID int64) bool
}

type DefaultPredictor struct{}

func (d DefaultPredictor) IsCold(id int64) bool { return false }

// -----------------------------------------------------------------------------
// Size Classes (Phân loại hạt)
// -----------------------------------------------------------------------------
// Mapping size -> Bin Index (0: 8B, 1: 16B, 2: 32B ...)
var sizeClasses = []int{8, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 384, 512, 1024, 2048, 4096}

func getSizeClass(size int) (int, int) {
	for i, s := range sizeClasses {
		if size <= s {
			return i, s
		}
	}
	return -1, -1 // Too big
}

// -----------------------------------------------------------------------------
// Slab (Trái Lựu) - 64KB Block
// -----------------------------------------------------------------------------
type Slab struct {
	data       []byte   // Vùng nhớ thực tế
	freeStack  []uint16 // Danh sách offset đã giải phóng (LIFO)
	bumpOffset uint32   // Con trỏ cấp phát tuần tự (Bump Pointer)

	activeCount int32 // Số lượng object đang sống
	lastAccess  int64 // Timestamp cho PPE Predict decay
	id          int64 // Unique ID
}

func newSlab(id int64) *Slab {
	return &Slab{
		data:       make([]byte, SlabSize),
		freeStack:  make([]uint16, 0, 64),
		bumpOffset: 0,
		lastAccess: time.Now().UnixNano(),
		id:         id,
	}
}

// -----------------------------------------------------------------------------
// Bin (Màng Lựu) - Quản lý list các Slab cùng size
// -----------------------------------------------------------------------------
type Bin struct {
	mu       sync.Mutex
	slotSize int

	fullSlabs    map[int64]*Slab // Các trái đã đầy
	partialSlabs map[int64]*Slab // Các trái còn chỗ (ưu tiên alloc)

	current *Slab // Slab hiện tại đang bump pointer
}

func newBin(size int) *Bin {
	return &Bin{
		slotSize:     size,
		fullSlabs:    make(map[int64]*Slab),
		partialSlabs: make(map[int64]*Slab),
	}
}

// -----------------------------------------------------------------------------
// LuuArena (Pomai Lựu Arena) - Trái tim quản lý bộ nhớ
// -----------------------------------------------------------------------------
type LuuArena struct {
	bins      []*Bin
	slabIDGen atomic.Int64
	predictor Predictor

	// Stats
	AllocBytes atomic.Int64
	FreeBytes  atomic.Int64
}

func NewLuuArena() *LuuArena {
	la := &LuuArena{
		bins:      make([]*Bin, len(sizeClasses)),
		predictor: DefaultPredictor{},
	}
	for i, s := range sizeClasses {
		la.bins[i] = newBin(s)
	}
	// Chạy thread Decay ngầm (như arena_decay của Redis)
	go la.decayLoop()
	return la
}

// Alloc cấp phát bộ nhớ.
// Trả về pointer an toàn (nhưng trỏ vào bên trong arena).
// Lưu ý: User phải copy data vào đây.
func (a *LuuArena) Alloc(size int) ([]byte, *Slab) {
	// 1. Nếu size quá lớn, bypass arena (để Go tự lo)
	if size > MaxSmallSize {
		a.AllocBytes.Add(int64(size))
		return make([]byte, size), nil
	}

	// 2. Tìm Bin phù hợp
	binIdx, slotSize := getSizeClass(size)
	bin := a.bins[binIdx]

	bin.mu.Lock()
	defer bin.mu.Unlock()

	// 3. Alloc từ Current Slab (Nhanh nhất - Bump Pointer)
	if bin.current != nil {
		ptr, ok := a.allocFromSlab(bin.current, slotSize)
		if ok {
			return ptr, bin.current
		}
		// Current đầy -> Move sang Full
		bin.fullSlabs[bin.current.id] = bin.current
		bin.current = nil
	}

	// 4. Tìm trong Partial Slabs (Slab cũ có lỗ trống do Free)
	for id, s := range bin.partialSlabs {
		ptr, ok := a.allocFromSlab(s, slotSize)
		if ok {
			if s.activeCount*int32(slotSize) >= int32(len(s.data)) {
				// Nếu đầy thì chuyển sang Full
				delete(bin.partialSlabs, id)
				bin.fullSlabs[id] = s
			} else {
				// Set làm current để ưu tiên lấp đầy
				bin.current = s
				delete(bin.partialSlabs, id)
			}
			return ptr, s
		}
	}

	// 5. Tạo Slab mới
	newS := newSlab(a.slabIDGen.Add(1))
	bin.current = newS
	ptr, _ := a.allocFromSlab(newS, slotSize) // Chắc chắn thành công

	a.AllocBytes.Add(int64(slotSize))
	return ptr, newS
}

func (a *LuuArena) allocFromSlab(s *Slab, size int) ([]byte, bool) {
	// A. Ưu tiên lấy từ FreeStack (Lỗ trống)
	if len(s.freeStack) > 0 {
		idx := len(s.freeStack) - 1
		offset := s.freeStack[idx]
		s.freeStack = s.freeStack[:idx]
		s.activeCount++
		s.lastAccess = time.Now().UnixNano()
		return s.data[offset : int(offset)+size], true
	}

	// B. Bump Pointer (Lấn dần vùng chưa dùng)
	if int(s.bumpOffset)+size <= len(s.data) {
		start := s.bumpOffset
		s.bumpOffset += uint32(size)
		s.activeCount++
		s.lastAccess = time.Now().UnixNano()
		return s.data[start : int(start)+size], true
	}

	return nil, false // Hết chỗ
}

// Free giải phóng vùng nhớ (Logic dalloc)
// Cần pointer gốc và Slab tham chiếu
func (a *LuuArena) Free(ptr []byte, s *Slab) {
	if s == nil {
		// Large object (Go managed), chỉ update stats
		a.FreeBytes.Add(int64(len(ptr)))
		return
	}

	// Tính toán offset từ pointer
	// Trong Go, so sánh địa chỉ slice hơi tricky.
	// Cách chuẩn: user phải giữ handle offset.
	// Ở đây giả định ta tính được offset nhờ pointer arithmetic (unsafe)
	// Hoặc đơn giản: Store sẽ lưu (SlabID, Offset) thay vì slice []byte

	// Code mô phỏng logic Decay:
	atomic.AddInt32(&s.activeCount, -1)

	// Update stat
	a.FreeBytes.Add(int64(cap(ptr)))

	// Nếu Slab trống rỗng -> Có thể giải phóng hoàn toàn về OS (cho GC hốt)
}

// decayLoop (Giống arena_decay_dirty)
func (a *LuuArena) decayLoop() {
	ticker := time.NewTicker(5 * time.Second)
	for range ticker.C {
		a.Purge()
	}
}

// Purge quét các Slab lạnh (Cold) hoặc bẩn (Dirty) để dọn dẹp
func (a *LuuArena) Purge() {
	now := time.Now().UnixNano()

	for _, bin := range a.bins {
		bin.mu.Lock()

		// 1. Quét Partial Slabs
		for id, s := range bin.partialSlabs {
			// PPE Logic: Nếu Slab này lạnh (ít truy cập) VÀ tỉ lệ sống thấp (< 25%)
			isCold := a.predictor.IsCold(s.id) || (now-s.lastAccess > int64(30*time.Second))
			isSparse := float64(s.activeCount)/float64(cap(s.data)/bin.slotSize) < 0.25

			if isCold && isSparse {
				// "Bóc lựu hỏng":
				// Chiến thuật: Copy các object còn sống sang Slab mới (Compact),
				// sau đó delete Slab cũ để GC hốt trọn ổ.
				// (Logic compact phức tạp, ở đây ta đơn giản là clear references nếu active=0)

				if s.activeCount == 0 {
					delete(bin.partialSlabs, id)
					// s sẽ bị GC thu gom
				}
			}
		}

		bin.mu.Unlock()
	}
}
