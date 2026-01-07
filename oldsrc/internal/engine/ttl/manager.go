package ttl

import (
	"context"
	"time"
)

type Manager struct {
	store   StoreInterface
	cleaner *PPPCleaner
}

func NewManager(store StoreInterface) *Manager {
	m := &Manager{
		store: store,
	}

	m.cleaner = NewPPPCleaner(m, store)
	return m
}

func (m *Manager) Start(ctx context.Context) {
	m.cleaner.Start(ctx)
}

func (m *Manager) TTLRemaining(key string) (time.Duration, bool) {
	if key == "" {
		return 0, false
	}

	shard := m.store.GetShard(key)
	shard.RLock()

	items := shard.GetItems()
	elem, ok := items[key]
	if !ok {
		shard.RUnlock()
		return 0, false
	}

	entry := extractEntry(elem)
	if entry == nil {
		shard.RUnlock()
		return 0, false
	}

	exp := entry.ExpireAt()
	if exp == 0 {
		shard.RUnlock()
		return 0, true
	}

	now := time.Now().UnixNano()
	if now > exp {
		shard.RUnlock()
		go m.deleteAsync(key)
		return 0, false
	}

	shard.RUnlock()
	return time.Duration(exp - now), true
}

func (m *Manager) deleteAsync(key string) {
	shard := m.store.GetShard(key)
	shard.Lock()
	defer shard.Unlock()

	if size, ok := shard.DeleteItem(key); ok {
		shard.AddBytes(-int64(size))
		m.store.AddTotalBytes(-int64(size))
		if mc := m.store.GetGlobalMemCtrl(); mc != nil {
			mc.Release(int64(size))
		}
	}
}

func extractEntry(elem interface{}) EntryInterface {
	if entry, ok := elem.(EntryInterface); ok {
		return entry
	}
	return nil
}
