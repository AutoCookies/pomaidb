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

	// Placeholder nếu sau này implement syncmap shard
	// syncmap    *SyncMapShard
	useSyncMap bool
}

func NewShard() *Shard {
	return &Shard{
		items: make(map[string]*list.Element),
		ll:    list.New(),
	}
}

func NewLockFreeShardAdapter() *Shard {
	return &Shard{
		lockfree:    NewLockFreeShard(),
		useLockfree: true,
	}
}

func (s *Shard) AtomicMutate(key string, mutator func(old *common.Entry) (*common.Entry, error)) error {
	if s.useLockfree {
		return s.lockfree.AtomicMutate(key, mutator)
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
		// Logic xóa nếu mutator trả về nil
		if exists {
			s.deleteElement(elem)
		}
		return nil
	}

	// Logic update/insert
	if exists {
		elem.Value = newEntry
		s.ll.MoveToFront(elem)
		// Tính delta size: mới - cũ
		delta := int64(newEntry.Size()) - int64(oldEntry.Size())
		s.bytes.Add(delta)
	} else {
		elem := s.ll.PushFront(newEntry)
		s.items[key] = elem
		s.bytes.Add(int64(newEntry.Size()))
	}

	return nil
}

func (s *Shard) Get(key string) (*common.Entry, bool) {
	if s.useLockfree {
		return s.lockfree.Get(key)
	}

	s.mu.RLock()
	elem, ok := s.items[key]
	s.mu.RUnlock()

	if !ok {
		return nil, false
	}

	// LRU promotion (cần Write Lock nếu muốn chính xác tuyệt đối,
	// hoặc dùng try-lock để tránh contention cho read-heavy)
	// Ở đây dùng strategy: Read trước, nếu tồn tại thì promote sau (tùy chọn)
	// Để đơn giản và an toàn thread, ta không promote trong Get (hoặc dùng LockFreeShard cho perf)
	return elem.Value.(*common.Entry), true
}

// GetAndTouch lấy item và đẩy lên đầu LRU
func (s *Shard) GetAndTouch(key string) (*common.Entry, bool) {
	if s.useLockfree {
		return s.lockfree.GetAndTouch(key)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	elem, ok := s.items[key]
	if !ok {
		return nil, false
	}
	s.ll.MoveToFront(elem)
	return elem.Value.(*common.Entry), true
}

// Set thêm hoặc cập nhật entry. Trả về entry cũ (nếu có) để Free memory.
func (s *Shard) Set(entry *common.Entry) (*common.Entry, int64) {
	if s.useLockfree {
		return s.lockfree.Set(entry)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var oldEntry *common.Entry
	var delta int64

	if elem, ok := s.items[entry.Key()]; ok {
		oldEntry = elem.Value.(*common.Entry)
		elem.Value = entry
		s.ll.MoveToFront(elem)
		delta = int64(entry.Size()) - int64(oldEntry.Size())
	} else {
		elem := s.ll.PushFront(entry)
		s.items[entry.Key()] = elem
		delta = int64(entry.Size())
	}

	s.bytes.Add(delta)
	return oldEntry, delta
}

// DeleteItem xóa item và trả về Entry để caller xử lý Free memory (Arena)
func (s *Shard) DeleteItem(key string) (*common.Entry, bool) {
	if s.useLockfree {
		return s.lockfree.DeleteItem(key)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	elem, ok := s.items[key]
	if !ok {
		return nil, false
	}

	entry := s.deleteElement(elem)
	return entry, true
}

// deleteElement internal helper
func (s *Shard) deleteElement(elem *list.Element) *common.Entry {
	s.ll.Remove(elem)
	entry := elem.Value.(*common.Entry)
	delete(s.items, entry.Key())
	s.bytes.Add(-int64(entry.Size()))
	return entry
}

// Delete wrapper cho tương thích ngược nếu cần
func (s *Shard) Delete(key string) bool {
	_, ok := s.DeleteItem(key)
	return ok
}

func (s *Shard) Clear() {
	if s.useLockfree {
		s.lockfree.Clear()
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.items = make(map[string]*list.Element)
	s.ll.Init()
	s.bytes.Store(0)
}

func (s *Shard) Lock()    { s.mu.Lock() }
func (s *Shard) Unlock()  { s.mu.Unlock() }
func (s *Shard) RLock()   { s.mu.RLock() }
func (s *Shard) RUnlock() { s.mu.RUnlock() }

func (s *Shard) GetItems() map[string]interface{} {
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
	if s.useLockfree {
		return nil // LockFree chưa hỗ trợ chuẩn LRU pointer public
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ll.Back()
}

func (s *Shard) GetBytes() int64 {
	return s.bytes.Load()
}

func (s *Shard) Len() int {
	if s.useLockfree {
		return s.lockfree.Len()
	}

	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.items)
}

func (s *Shard) Bytes() int64 {
	if s.useLockfree {
		return s.lockfree.Bytes()
	}
	return s.bytes.Load()
}

func (s *Shard) EvictExpired() []*common.Entry {
	if s.useLockfree {
		return s.lockfree.EvictExpired()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var expired []*common.Entry

	// Duyệt danh sách items
	for _, elem := range s.items {
		entry := elem.Value.(*common.Entry)
		if entry.IsExpired() {
			s.deleteElement(elem)
			expired = append(expired, entry)
		}
	}

	return expired
}
