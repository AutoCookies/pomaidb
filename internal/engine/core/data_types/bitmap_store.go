package data_types

import (
	"fmt"
	"math/bits" // [NEW] Import để đếm bit nhanh
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/AutoCookies/pomai-cache/shared/ds/bitpack"
)

const (
	BitmapPageBits = 1 << 16             // 64K bits per page
	BitmapPageSize = BitmapPageBits / 64 // uint64 words count
)

type BitmapStore struct {
	bitmaps sync.Map
	packer  *bitpack.PPBP
}

type Page struct {
	mu           sync.RWMutex
	Raw          []uint64
	Packed       []byte
	IsCompressed bool
	OriginalSize int
}

type Bitmap struct {
	mu         sync.RWMutex
	pages      map[uint32]*Page
	totalOnes  atomic.Int64
	lastOffset atomic.Uint64
}

func NewBitmapStore() *BitmapStore {
	return &BitmapStore{
		packer: bitpack.New(),
	}
}

func (bs *BitmapStore) getOrCreate(key string) *Bitmap {
	v, ok := bs.bitmaps.Load(key)
	if ok {
		return v.(*Bitmap)
	}
	b := &Bitmap{
		pages: make(map[uint32]*Page),
	}
	actual, _ := bs.bitmaps.LoadOrStore(key, b)
	return actual.(*Bitmap)
}

func (bs *BitmapStore) SetBit(key string, offset uint64, value int) (int, error) {
	if value != 0 && value != 1 {
		return 0, fmt.Errorf("bit value must be 0 or 1")
	}

	b := bs.getOrCreate(key)

	pageIdx := uint32(offset / BitmapPageBits)
	bitIdx := offset % BitmapPageBits
	wordIdx := bitIdx / 64
	bitMask := uint64(1 << (bitIdx % 64))

	b.mu.Lock()
	page, exists := b.pages[pageIdx]
	if !exists {
		page = &Page{
			Raw: make([]uint64, BitmapPageSize),
		}
		b.pages[pageIdx] = page
	}
	b.mu.Unlock()

	page.mu.Lock()
	defer page.mu.Unlock()

	if page.IsCompressed {
		bs.decompressPage(page)
	}

	word := page.Raw[wordIdx]
	original := 0
	if word&bitMask != 0 {
		original = 1
	}

	if value == 1 {
		if original == 0 {
			page.Raw[wordIdx] |= bitMask
			b.totalOnes.Add(1)
		}
	} else {
		if original == 1 {
			page.Raw[wordIdx] &^= bitMask
			b.totalOnes.Add(-1)
		}
	}

	currentMax := b.lastOffset.Load()
	if offset > currentMax {
		b.lastOffset.Store(offset)
	}

	return original, nil
}

func (bs *BitmapStore) GetBit(key string, offset uint64) (int, error) {
	v, ok := bs.bitmaps.Load(key)
	if !ok {
		return 0, nil
	}
	b := v.(*Bitmap)

	pageIdx := uint32(offset / BitmapPageBits)
	bitIdx := offset % BitmapPageBits
	wordIdx := bitIdx / 64
	bitMask := uint64(1 << (bitIdx % 64))

	b.mu.RLock()
	page, exists := b.pages[pageIdx]
	b.mu.RUnlock()

	if !exists {
		return 0, nil
	}

	page.mu.RLock()
	defer page.mu.RUnlock()

	if page.IsCompressed {
		// Upgrade lock to write
		page.mu.RUnlock()
		page.mu.Lock()
		if page.IsCompressed {
			bs.decompressPage(page)
		}
		page.mu.Unlock()
		page.mu.RLock()
	}

	if page.Raw[wordIdx]&bitMask != 0 {
		return 1, nil
	}
	return 0, nil
}

// [NEW] BitCount đếm số bit set trong khoảng byte [start, end]
func (bs *BitmapStore) BitCount(key string, start, end int64) (int64, error) {
	v, ok := bs.bitmaps.Load(key)
	if !ok {
		return 0, nil
	}
	b := v.(*Bitmap)

	b.mu.RLock()
	defer b.mu.RUnlock()

	// Xử lý Redis-style negative indices
	lastByte := int64(b.lastOffset.Load()) / 8
	totalBytes := lastByte + 1

	if start < 0 {
		start += totalBytes
	}
	if end < 0 {
		end += totalBytes
	}
	if start < 0 {
		start = 0
	}
	if end < 0 {
		end = 0
	}
	if start > end {
		return 0, nil
	}

	// Chuyển đổi byte range -> bit range
	startBit := uint64(start) * 8
	endBit := uint64(end+1) * 8 // Exclusive limit

	var total int64 = 0

	// Duyệt qua các page có trong map
	for pageIdx, page := range b.pages {
		pageStartBit := uint64(pageIdx) * BitmapPageBits
		pageEndBit := pageStartBit + BitmapPageBits

		// Tìm giao điểm giữa [startBit, endBit) và Page
		interStart := startBit
		if pageStartBit > interStart {
			interStart = pageStartBit
		}
		interEnd := endBit
		if pageEndBit < interEnd {
			interEnd = pageEndBit
		}

		if interStart >= interEnd {
			continue // Page nằm ngoài range
		}

		// Tính offset tương đối trong page
		relStart := interStart - pageStartBit
		relEnd := interEnd - pageStartBit

		total += bs.countBitsInPage(page, relStart, relEnd)
	}

	return total, nil
}

// Helper đếm bit trong Page (xử lý cả compressed)
func (bs *BitmapStore) countBitsInPage(p *Page, start, end uint64) int64 {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Nếu đang nén -> Giải nén (Coi như truy cập -> Hot)
	if p.IsCompressed {
		bs.decompressPage(p)
	}

	// Đếm bit trong range [start, end) của mảng Raw []uint64
	var count int64
	startWord := start / 64
	endWord := (end + 63) / 64

	for i := startWord; i < endWord; i++ {
		word := p.Raw[i]

		// Masking cho word đầu và cuối
		// Logic:
		// 1. Nếu là word đầu tiên, mask bỏ các bit trước start
		// 2. Nếu là word cuối cùng, mask bỏ các bit sau end

		wordStartBit := i * 64
		wordEndBit := wordStartBit + 64

		// Mask đầu
		if wordStartBit < start {
			mask := ^uint64(0) << (start % 64)
			word &= mask
		}

		// Mask cuối
		if wordEndBit > end {
			diff := wordEndBit - end
			mask := ^uint64(0) >> diff
			word &= mask
		}

		count += int64(bits.OnesCount64(word))
	}
	return count
}

func (bs *BitmapStore) CompressPage(key string, pageIdx uint32) bool {
	v, ok := bs.bitmaps.Load(key)
	if !ok {
		return false
	}
	b := v.(*Bitmap)

	b.mu.RLock()
	page, exists := b.pages[pageIdx]
	b.mu.RUnlock()
	if !exists {
		return false
	}

	page.mu.Lock()
	defer page.mu.Unlock()

	if page.IsCompressed {
		return false
	}

	byteView := u64SliceToByteSlice(page.Raw)
	packed, origLen, compressed := bs.packer.Pack(key, byteView)

	if compressed {
		page.Packed = packed
		page.Raw = nil
		page.OriginalSize = origLen
		page.IsCompressed = true
		return true
	}
	return false
}

func (bs *BitmapStore) decompressPage(p *Page) {
	if !p.IsCompressed {
		return
	}
	bytes := bs.packer.Unpack(p.Packed, p.OriginalSize)
	p.Raw = byteSliceToU64Slice(bytes)
	p.Packed = nil
	p.IsCompressed = false
}

func u64SliceToByteSlice(u64 []uint64) []byte {
	const sizeOfUint64 = 8
	lenBytes := len(u64) * sizeOfUint64
	return unsafe.Slice((*byte)(unsafe.Pointer(&u64[0])), lenBytes)
}

func byteSliceToU64Slice(b []byte) []uint64 {
	const sizeOfUint64 = 8
	lenU64 := len(b) / sizeOfUint64
	newU64 := make([]uint64, lenU64)
	copy(u64SliceToByteSlice(newU64), b)
	return newU64
}
