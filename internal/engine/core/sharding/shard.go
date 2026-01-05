package sharding

import (
	"container/list"
	"sync"
	"sync/atomic"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
)

const (
	minCapacity = 4096
)

type Shard struct {
	mu    sync.RWMutex
	items map[string]*list.Element
	ll    *list.List
	bytes atomic.Int64

	lockfree    *LockFreeShard
	useLockfree bool

	syncmap    *SyncMapShard
	useSyncMap bool
}

func (s *Shard) AtomicMutate(key string, mutator func(old *common.Entry) (*common.Entry, error)) error {
	if s.useLockfree {
		return s.lockfree.AtomicMutate(key, mutator)
	}

	if s.useSyncMap {
		return s.syncmap.AtomicMutate(key, mutator)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var oldEntry *common.Entry
	elem, exists := s.items[key]

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

	var deltaBytes int64
	newSize := int64(newEntry.Size())

	if exists {
		deltaBytes = newSize - int64(oldEntry.Size())
		elem.Value = newEntry
		s.ll.MoveToFront(elem)
	} else {
		elem := s.ll.PushFront(newEntry)
		s.items[key] = elem
		deltaBytes = newSize
	}

	s.bytes.Add(deltaBytes)
	return nil
}

func NewShard() *Shard {
	return NewShardWithCapacity(minCapacity)
}

func NewShardWithCapacity(capacity int) *Shard {
	if capacity < minCapacity {
		capacity = minCapacity
	}

	return &Shard{
		items:       make(map[string]*list.Element, capacity),
		ll:          list.New(),
		useLockfree: false,
		useSyncMap:  false,
	}
}

func NewLockFreeShardAdapter() *Shard {
	lfs := NewLockFreeShard(0)

	return &Shard{
		items:       make(map[string]*list.Element, 1),
		ll:          list.New(),
		lockfree:    lfs,
		useLockfree: true,
		useSyncMap:  false,
	}
}

func NewSyncMapShardAdapter() *Shard {
	sms := NewSyncMapShard()

	return &Shard{
		items:       make(map[string]*list.Element, 1),
		ll:          list.New(),
		syncmap:     sms,
		useSyncMap:  true,
		useLockfree: false,
	}
}

func (s *Shard) Get(key string) (*common.Entry, bool) {
	if s.useSyncMap {
		entry, ok := s.syncmap.Get(key)
		if ok {
			entry.Touch()
		}
		return entry, ok
	}

	if s.useLockfree {
		entry, ok := s.lockfree.Get(key)
		return entry, ok
	}

	s.mu.RLock()
	elem := s.items[key]
	s.mu.RUnlock()

	if elem == nil {
		return nil, false
	}

	entry := elem.Value.(*common.Entry)
	if entry.IsExpired() {
		return nil, false
	}

	entry.Touch()

	s.mu.Lock()
	if elem2, ok := s.items[key]; ok && elem2 == elem {
		s.ll.MoveToFront(elem)
	}
	s.mu.Unlock()

	return entry, true
}

func (s *Shard) Set(entry *common.Entry) (*common.Entry, int64) {
	if s.useSyncMap {
		old, delta := s.syncmap.Set(entry)
		entry.Touch()
		return old, delta
	}

	if s.useLockfree {
		return s.lockfree.Set(entry)
	}

	key := entry.Key()
	newSize := int64(entry.Size())

	s.mu.Lock()

	var oldEntry *common.Entry
	var deltaBytes int64

	if elem := s.items[key]; elem != nil {
		oldEntry = elem.Value.(*common.Entry)
		deltaBytes = newSize - int64(oldEntry.Size())
		elem.Value = entry
		s.ll.MoveToFront(elem)
	} else {
		elem := s.ll.PushFront(entry)
		s.items[key] = elem
		deltaBytes = newSize
	}

	s.mu.Unlock()
	s.bytes.Add(deltaBytes)

	entry.Touch()

	return oldEntry, deltaBytes
}

func (s *Shard) Delete(key string) (*common.Entry, bool) {
	if s.useSyncMap {
		return s.syncmap.Delete(key)
	}

	if s.useLockfree {
		return s.lockfree.Delete(key)
	}

	s.mu.Lock()

	elem := s.items[key]
	if elem == nil {
		s.mu.Unlock()
		return nil, false
	}

	entry := elem.Value.(*common.Entry)
	entrySize := int64(entry.Size())

	delete(s.items, key)
	s.ll.Remove(elem)

	s.mu.Unlock()
	s.bytes.Add(-entrySize)

	return entry, true
}

func (s *Shard) Len() int {
	if s.useSyncMap {
		return s.syncmap.Len()
	}

	if s.useLockfree {
		return s.lockfree.Len()
	}

	s.mu.RLock()
	length := len(s.items)
	s.mu.RUnlock()
	return length
}

func (s *Shard) Bytes() int64 {
	if s.useSyncMap {
		return s.syncmap.Bytes()
	}

	if s.useLockfree {
		return s.lockfree.Bytes()
	}

	return s.bytes.Load()
}

func (s *Shard) EvictExpired() []*common.Entry {
	if s.useSyncMap {
		return s.syncmap.EvictExpired()
	}

	if s.useLockfree {
		return s.lockfree.EvictExpired()
	}

	s.mu.Lock()

	expired := make([]*common.Entry, 0, 64)
	toDelete := make([]*list.Element, 0, 64)
	var totalSize int64

	count := 0
	for elem := s.ll.Back(); elem != nil && count < 100; elem = elem.Prev() {
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
		delete(s.items, entry.Key())
		s.ll.Remove(elem)
	}

	s.mu.Unlock()

	if totalSize > 0 {
		s.bytes.Add(-totalSize)
	}

	return expired
}

func (s *Shard) Clear() {
	if s.useSyncMap {
		s.syncmap.Clear()
		return
	}

	if s.useLockfree {
		s.lockfree.Clear()
		return
	}

	s.mu.Lock()
	s.items = make(map[string]*list.Element, minCapacity)
	s.ll = list.New()
	s.mu.Unlock()
	s.bytes.Store(0)
}

func (s *Shard) GetAndTouch(key string) (*common.Entry, bool) {
	if s.useSyncMap {
		return s.syncmap.Get(key)
	}

	if s.useLockfree {
		return s.lockfree.GetAndTouch(key)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	elem := s.items[key]
	if elem == nil {
		return nil, false
	}

	entry := elem.Value.(*common.Entry)
	if entry.IsExpired() {
		return nil, false
	}

	s.ll.MoveToFront(elem)
	entry.Touch()

	return entry, true
}

func (s *Shard) GetFast(key string) (*common.Entry, bool) {
	if s.useLockfree {
		return s.lockfree.GetFast(key)
	}
	return s.Get(key)
}

func (s *Shard) SetFast(entry *common.Entry) (*common.Entry, int64) {
	if s.useLockfree {
		return s.lockfree.SetFast(entry)
	}
	return s.Set(entry)
}

func (s *Shard) Lock() {
	if !s.useLockfree && !s.useSyncMap {
		s.mu.Lock()
	}
}

func (s *Shard) Unlock() {
	if !s.useLockfree && !s.useSyncMap {
		s.mu.Unlock()
	}
}

func (s *Shard) RLock() {
	if !s.useLockfree && !s.useSyncMap {
		s.mu.RLock()
	}
}

func (s *Shard) RUnlock() {
	if !s.useLockfree && !s.useSyncMap {
		s.mu.RUnlock()
	}
}

func (s *Shard) GetItems() map[string]interface{} {
	if s.useSyncMap {
		return s.syncmap.GetItems()
	}

	if s.useLockfree {
		return s.lockfree.GetItems()
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	items := make(map[string]interface{}, len(s.items))
	for k, v := range s.items {
		items[k] = v
	}
	return items
}

func (s *Shard) GetLRUBack() interface{} {
	if s.useSyncMap {
		return nil
	}

	if s.useLockfree {
		return s.lockfree.GetLRUBack()
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ll.Back()
}

func (s *Shard) DeleteItem(key string) (size int, ok bool) {
	entry, ok := s.Delete(key)
	if !ok {
		return 0, false
	}
	return entry.Size(), true
}

func (s *Shard) GetBytes() int64 {
	return s.Bytes()
}

func (s *Shard) AddBytes(delta int64) {
	if s.useSyncMap {
		s.syncmap.AddBytes(delta)
		return
	}

	if s.useLockfree {
		s.lockfree.AddBytes(delta)
		return
	}

	s.bytes.Add(delta)
}

func (s *Shard) GetItemCount() int {
	return s.Len()
}

func (s *Shard) Compact() int {
	return len(s.EvictExpired())
}
