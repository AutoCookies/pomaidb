package sharding

import (
	"container/list"
	"sync"
	"sync/atomic"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
)

const (
	fnvOffset32 = 2166136261
	fnvPrime32  = 16777619

	defaultStripeCount = 256
	minCap             = 4096
)

type LockFreeShard struct {
	stripes    []shardStripe
	stripeMask uint32
}

type shardStripe struct {
	mu    sync.RWMutex
	items map[string]*list.Element
	ll    *list.List
	bytes atomic.Int64
	pad   [56]byte
}

func (s *LockFreeShard) AtomicMutate(key string, mutator func(old *common.Entry) (*common.Entry, error)) error {
	stripe := s.getStripe(key)
	stripe.mu.Lock()
	defer stripe.mu.Unlock()

	var oldEntry *common.Entry
	elem, exists := stripe.items[key]

	if exists {
		oldEntry = elem.Value.(*common.Entry)
	}

	newEntry, err := mutator(oldEntry)
	if err != nil {
		return err
	}

	if newEntry == nil {
		return nil
	}

	newSize := int64(newEntry.Size())
	var deltaBytes int64

	if exists {
		deltaBytes = newSize - int64(oldEntry.Size())
		elem.Value = newEntry
		stripe.ll.MoveToFront(elem)
	} else {
		elem := stripe.ll.PushFront(newEntry)
		stripe.items[key] = elem
		deltaBytes = newSize
	}

	stripe.bytes.Add(deltaBytes)
	return nil
}

func NewLockFreeShard(stripeCount int) *LockFreeShard {
	if stripeCount <= 0 {
		stripeCount = defaultStripeCount
	}

	powerOf2 := 1
	for powerOf2 < stripeCount {
		powerOf2 <<= 1
	}
	stripeCount = powerOf2

	s := &LockFreeShard{
		stripes:    make([]shardStripe, stripeCount),
		stripeMask: uint32(stripeCount - 1),
	}

	for i := 0; i < stripeCount; i++ {
		s.stripes[i].items = make(map[string]*list.Element, minCap/stripeCount)
		s.stripes[i].ll = list.New()
	}
	return s
}

func (s *LockFreeShard) MoveToFront(key string) {
	stripe := s.getStripe(key)
	stripe.mu.Lock()
	defer stripe.mu.Unlock()

	elem, ok := stripe.items[key]
	if ok {
		stripe.ll.MoveToFront(elem)
	}
}

func (s *LockFreeShard) getStripe(key string) *shardStripe {
	hash := uint32(fnvOffset32)
	for i := 0; i < len(key); i++ {
		hash ^= uint32(key[i])
		hash *= fnvPrime32
	}
	return &s.stripes[hash&s.stripeMask]
}

func (s *LockFreeShard) Get(key string) (*common.Entry, bool) {
	stripe := s.getStripe(key)

	stripe.mu.RLock()
	elem := stripe.items[key]
	stripe.mu.RUnlock()

	if elem == nil {
		return nil, false
	}

	entry := elem.Value.(*common.Entry)
	if entry.IsExpired() {
		return nil, false
	}

	entry.Touch()

	stripe.mu.Lock()
	if elem2, ok := stripe.items[key]; ok && elem2 == elem {
		stripe.ll.MoveToFront(elem)
	}
	stripe.mu.Unlock()

	return entry, true
}

func (s *LockFreeShard) Set(entry *common.Entry) (*common.Entry, int64) {
	key := entry.Key()
	newSize := int64(entry.Size())

	stripe := s.getStripe(key)

	stripe.mu.Lock()

	var oldEntry *common.Entry
	var deltaBytes int64

	if elem := stripe.items[key]; elem != nil {
		oldEntry = elem.Value.(*common.Entry)
		deltaBytes = newSize - int64(oldEntry.Size())
		elem.Value = entry
		stripe.ll.MoveToFront(elem)
	} else {
		elem := stripe.ll.PushFront(entry)
		stripe.items[key] = elem
		deltaBytes = newSize
	}

	stripe.mu.Unlock()
	stripe.bytes.Add(deltaBytes)

	entry.Touch()

	return oldEntry, deltaBytes
}

func (s *LockFreeShard) Delete(key string) (*common.Entry, bool) {
	stripe := s.getStripe(key)

	stripe.mu.Lock()

	elem := stripe.items[key]
	if elem == nil {
		stripe.mu.Unlock()
		return nil, false
	}

	entry := elem.Value.(*common.Entry)
	entrySize := int64(entry.Size())

	delete(stripe.items, key)
	stripe.ll.Remove(elem)

	stripe.mu.Unlock()
	stripe.bytes.Add(-entrySize)

	return entry, true
}

func (s *LockFreeShard) Len() int {
	total := 0
	for i := range s.stripes {
		stripe := &s.stripes[i]
		stripe.mu.RLock()
		total += len(stripe.items)
		stripe.mu.RUnlock()
	}
	return total
}

func (s *LockFreeShard) Bytes() int64 {
	total := int64(0)
	for i := range s.stripes {
		total += s.stripes[i].bytes.Load()
	}
	return total
}

func (s *LockFreeShard) EvictExpired() []*common.Entry {
	expired := make([]*common.Entry, 0, 64)

	for i := range s.stripes {
		stripe := &s.stripes[i]

		stripe.mu.Lock()

		toDelete := make([]*list.Element, 0, 8)
		var totalSize int64

		count := 0
		for elem := stripe.ll.Back(); elem != nil && count < 10; elem = elem.Prev() {
			entry := elem.Value.(*common.Entry)
			if entry.IsExpired() {
				expired = append(expired, entry)
				toDelete = append(toDelete, elem)
				totalSize += int64(entry.Size())
			}
			count++
		}

		for _, elem := range toDelete {
			entry := elem.Value.(*common.Entry)
			delete(stripe.items, entry.Key())
			stripe.ll.Remove(elem)
		}

		stripe.mu.Unlock()

		if totalSize > 0 {
			stripe.bytes.Add(-totalSize)
		}
	}

	return expired
}

func (s *LockFreeShard) Clear() {
	for i := range s.stripes {
		stripe := &s.stripes[i]
		stripe.mu.Lock()
		stripe.items = make(map[string]*list.Element, minCap/len(s.stripes))
		stripe.ll = list.New()
		stripe.mu.Unlock()
		stripe.bytes.Store(0)
	}
}

func (s *LockFreeShard) GetAndTouch(key string) (*common.Entry, bool) {
	return s.Get(key)
}

func (s *LockFreeShard) GetFast(key string) (*common.Entry, bool) {
	return s.Get(key)
}

func (s *LockFreeShard) SetFast(entry *common.Entry) (*common.Entry, int64) {
	return s.Set(entry)
}

func (s *LockFreeShard) Lock()    {}
func (s *LockFreeShard) Unlock()  {}
func (s *LockFreeShard) RLock()   {}
func (s *LockFreeShard) RUnlock() {}

func (s *LockFreeShard) GetItems() map[string]interface{} {
	items := make(map[string]interface{})
	for i := range s.stripes {
		stripe := &s.stripes[i]
		stripe.mu.RLock()
		for k, v := range stripe.items {
			items[k] = v
		}
		stripe.mu.RUnlock()
	}
	return items
}

func (s *LockFreeShard) GetLRUBack() interface{} {
	return nil
}

func (s *LockFreeShard) DeleteItem(key string) (size int, ok bool) {
	entry, ok := s.Delete(key)
	if !ok {
		return 0, false
	}
	return entry.Size(), true
}

func (s *LockFreeShard) GetBytes() int64 {
	return s.Bytes()
}

func (s *LockFreeShard) AddBytes(delta int64) {
	stripe := &s.stripes[0]
	stripe.bytes.Add(delta)
}

func (s *LockFreeShard) GetItemCount() int {
	return s.Len()
}

func (s *LockFreeShard) Compact() int {
	return len(s.EvictExpired())
}
