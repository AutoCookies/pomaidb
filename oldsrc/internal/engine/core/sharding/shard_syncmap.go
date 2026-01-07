package sharding

import (
	"sync"
	"sync/atomic"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
)

type SyncMapShard struct {
	items sync.Map
	bytes atomic.Int64
}

func (s *SyncMapShard) AtomicMutate(key string, mutator func(old *common.Entry) (*common.Entry, error)) error {
	for {
		val, loaded := s.items.Load(key)
		var oldEntry *common.Entry
		if loaded {
			oldEntry = val.(*common.Entry)
		}

		newEntry, err := mutator(oldEntry)
		if err != nil {
			return err
		}

		if newEntry == nil {
			return nil
		}

		newSize := int64(newEntry.Size())
		var oldSize int64
		if oldEntry != nil {
			oldSize = int64(oldEntry.Size())
		}

		if loaded {
			if s.items.CompareAndSwap(key, val, newEntry) {
				s.bytes.Add(newSize - oldSize)
				return nil
			}
		} else {
			if _, loaded = s.items.LoadOrStore(key, newEntry); !loaded {
				s.bytes.Add(newSize)
				return nil
			}
		}
	}
}

func NewSyncMapShard() *SyncMapShard {
	return &SyncMapShard{}
}

func (s *SyncMapShard) Get(key string) (*common.Entry, bool) {
	val, ok := s.items.Load(key)
	if !ok {
		return nil, false
	}

	entry := val.(*common.Entry)
	if entry.IsExpired() {
		return nil, false
	}

	return entry, true
}

func (s *SyncMapShard) Set(entry *common.Entry) (*common.Entry, int64) {
	key := entry.Key()
	newSize := int64(entry.Size())

	val, loaded := s.items.LoadOrStore(key, entry)

	if loaded {
		oldEntry := val.(*common.Entry)
		deltaBytes := newSize - int64(oldEntry.Size())
		s.items.Store(key, entry)
		s.bytes.Add(deltaBytes)
		return oldEntry, deltaBytes
	}

	s.bytes.Add(newSize)
	return nil, newSize
}

func (s *SyncMapShard) Delete(key string) (*common.Entry, bool) {
	val, ok := s.items.LoadAndDelete(key)
	if !ok {
		return nil, false
	}

	entry := val.(*common.Entry)
	s.bytes.Add(-int64(entry.Size()))

	return entry, true
}

func (s *SyncMapShard) Bytes() int64 {
	return s.bytes.Load()
}

func (s *SyncMapShard) Len() int {
	count := 0
	s.items.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

func (s *SyncMapShard) Clear() {
	s.items.Range(func(key, _ interface{}) bool {
		s.items.Delete(key)
		return true
	})
	s.bytes.Store(0)
}

func (s *SyncMapShard) EvictExpired() []*common.Entry {
	expired := make([]*common.Entry, 0, 64)

	s.items.Range(func(key, val interface{}) bool {
		entry := val.(*common.Entry)
		if entry.IsExpired() {
			expired = append(expired, entry)
			s.items.Delete(key)
			s.bytes.Add(-int64(entry.Size()))
		}
		return len(expired) < 100
	})

	return expired
}

func (s *SyncMapShard) GetItems() map[string]interface{} {
	items := make(map[string]interface{})
	s.items.Range(func(key, val interface{}) bool {
		items[key.(string)] = val
		return true
	})
	return items
}

func (s *SyncMapShard) AddBytes(delta int64) {
	s.bytes.Add(delta)
}
