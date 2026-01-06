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
	pad   [56]byte // Padding tránh False Sharing cache line
}

func NewLockFreeShard() *LockFreeShard {
	s := &LockFreeShard{
		stripes:    make([]shardStripe, defaultStripeCount),
		stripeMask: defaultStripeCount - 1,
	}

	for i := range s.stripes {
		s.stripes[i].items = make(map[string]*list.Element, minCap/defaultStripeCount)
		s.stripes[i].ll = list.New()
	}

	return s
}

func (s *LockFreeShard) getStripe(key string) *shardStripe {
	h := fnv32(key)
	return &s.stripes[h&s.stripeMask]
}

func fnv32(key string) uint32 {
	hash := uint32(fnvOffset32)
	for i := 0; i < len(key); i++ {
		hash ^= uint32(key[i])
		hash *= fnvPrime32
	}
	return hash
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
		if exists {
			s.deleteElement(stripe, elem)
		}
		return nil
	}

	if exists {
		elem.Value = newEntry
		stripe.ll.MoveToFront(elem)
		delta := int64(newEntry.Size()) - int64(oldEntry.Size())
		stripe.bytes.Add(delta)
	} else {
		elem := stripe.ll.PushFront(newEntry)
		stripe.items[key] = elem
		stripe.bytes.Add(int64(newEntry.Size()))
	}

	return nil
}

func (s *LockFreeShard) Get(key string) (*common.Entry, bool) {
	stripe := s.getStripe(key)
	stripe.mu.RLock()
	defer stripe.mu.RUnlock()

	if elem, ok := stripe.items[key]; ok {
		return elem.Value.(*common.Entry), true
	}
	return nil, false
}

func (s *LockFreeShard) GetAndTouch(key string) (*common.Entry, bool) {
	stripe := s.getStripe(key)
	stripe.mu.Lock()
	defer stripe.mu.Unlock()

	if elem, ok := stripe.items[key]; ok {
		stripe.ll.MoveToFront(elem)
		return elem.Value.(*common.Entry), true
	}
	return nil, false
}

// Set thêm/cập nhật và trả về oldEntry để Store giải phóng bộ nhớ Arena
func (s *LockFreeShard) Set(entry *common.Entry) (*common.Entry, int64) {
	stripe := s.getStripe(entry.Key())
	stripe.mu.Lock()
	defer stripe.mu.Unlock()

	var oldEntry *common.Entry
	var delta int64

	if elem, ok := stripe.items[entry.Key()]; ok {
		oldEntry = elem.Value.(*common.Entry)
		elem.Value = entry
		stripe.ll.MoveToFront(elem)
		delta = int64(entry.Size()) - int64(oldEntry.Size())
	} else {
		elem := stripe.ll.PushFront(entry)
		stripe.items[entry.Key()] = elem
		delta = int64(entry.Size())
	}

	stripe.bytes.Add(delta)
	return oldEntry, delta
}

// DeleteItem xóa và trả về Entry để giải phóng Arena
func (s *LockFreeShard) DeleteItem(key string) (*common.Entry, bool) {
	stripe := s.getStripe(key)
	stripe.mu.Lock()
	defer stripe.mu.Unlock()

	elem, ok := stripe.items[key]
	if !ok {
		return nil, false
	}

	entry := s.deleteElement(stripe, elem)
	return entry, true
}

func (s *LockFreeShard) deleteElement(stripe *shardStripe, elem *list.Element) *common.Entry {
	stripe.ll.Remove(elem)
	entry := elem.Value.(*common.Entry)
	delete(stripe.items, entry.Key())
	stripe.bytes.Add(-int64(entry.Size()))
	return entry
}

func (s *LockFreeShard) Clear() {
	for i := range s.stripes {
		stripe := &s.stripes[i]
		stripe.mu.Lock()
		stripe.items = make(map[string]*list.Element, minCap/len(s.stripes))
		stripe.ll = list.New()
		stripe.bytes.Store(0)
		stripe.mu.Unlock()
	}
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

func (s *LockFreeShard) Len() int {
	count := 0
	for i := range s.stripes {
		stripe := &s.stripes[i]
		stripe.mu.RLock()
		count += len(stripe.items)
		stripe.mu.RUnlock()
	}
	return count
}

func (s *LockFreeShard) Bytes() int64 {
	var total int64
	for i := range s.stripes {
		total += s.stripes[i].bytes.Load()
	}
	return total
}

func (s *LockFreeShard) EvictExpired() []*common.Entry {
	var expiredEntries []*common.Entry

	for i := range s.stripes {
		stripe := &s.stripes[i]
		stripe.mu.Lock()
		for _, elem := range stripe.items {
			entry := elem.Value.(*common.Entry)
			if entry.IsExpired() {
				s.deleteElement(stripe, elem)
				expiredEntries = append(expiredEntries, entry)
			}
		}
		stripe.mu.Unlock()
	}

	return expiredEntries
}
