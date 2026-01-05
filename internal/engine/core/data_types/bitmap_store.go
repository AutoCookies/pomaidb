package data_types

import (
	"fmt"
	"math/bits"
	"sync"
	"sync/atomic"
)

const (
	BitmapPageBits = 1 << 16
	BitmapPageSize = BitmapPageBits / 64
)

type BitmapStore struct {
	bitmaps sync.Map
}

type Bitmap struct {
	mu         sync.RWMutex
	pages      map[uint32][]uint64
	totalOnes  atomic.Int64
	lastOffset atomic.Uint64
}

func NewBitmapStore() *BitmapStore {
	return &BitmapStore{}
}

func (bs *BitmapStore) getOrCreate(key string) *Bitmap {
	v, ok := bs.bitmaps.Load(key)
	if ok {
		return v.(*Bitmap)
	}
	b := &Bitmap{
		pages: make(map[uint32][]uint64),
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
	defer b.mu.Unlock()

	page, exists := b.pages[pageIdx]
	if !exists {
		if value == 0 {
			return 0, nil
		}
		page = make([]uint64, BitmapPageSize)
		b.pages[pageIdx] = page
	}

	word := page[wordIdx]
	original := 0
	if word&bitMask != 0 {
		original = 1
	}

	if value == 1 {
		if original == 0 {
			page[wordIdx] |= bitMask
			b.totalOnes.Add(1)
		}
	} else {
		if original == 1 {
			page[wordIdx] &^= bitMask
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
	defer b.mu.RUnlock()

	page, exists := b.pages[pageIdx]
	if !exists {
		return 0, nil
	}

	if page[wordIdx]&bitMask != 0 {
		return 1, nil
	}
	return 0, nil
}

func (bs *BitmapStore) BitCount(key string, start, end int64) (int64, error) {
	v, ok := bs.bitmaps.Load(key)
	if !ok {
		return 0, nil
	}
	b := v.(*Bitmap)

	if start == 0 && (end == -1 || end == 0) {
		return b.totalOnes.Load(), nil
	}

	b.mu.RLock()
	defer b.mu.RUnlock()

	var count int64
	for _, page := range b.pages {
		for _, word := range page {
			count += int64(bits.OnesCount64(word))
		}
	}
	return count, nil
}

func (bs *BitmapStore) DropBitmap(key string) {
	bs.bitmaps.Delete(key)
}
