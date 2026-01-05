package data_types

import (
	"errors"
	"hash/fnv"
	"sync"

	ds "github.com/AutoCookies/pomai-cache/packages/ds/hash"
)

const (
	HashShards    = 64
	HashShardMask = HashShards - 1
)

// HashShard quản lý một phần keyspace
type HashShard struct {
	mu sync.RWMutex
	// Value là interface (FlatHash hoặc MapHash)
	items map[string]ds.HashObject
}

type HashStore struct {
	shards [HashShards]*HashShard
}

func NewHashStore() *HashStore {
	store := &HashStore{}
	for i := 0; i < HashShards; i++ {
		store.shards[i] = &HashShard{
			items: make(map[string]ds.HashObject),
		}
	}
	return store
}

func (h *HashStore) getShard(key string) *HashShard {
	hasher := fnv.New32a()
	hasher.Write([]byte(key))
	return h.shards[hasher.Sum32()&HashShardMask]
}

func (h *HashStore) HSet(key, field string, value []byte) error {
	if key == "" || field == "" {
		return errors.New("empty key or field")
	}

	shard := h.getShard(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	obj, exists := shard.items[key]
	if !exists {
		// Mặc định khởi tạo là FlatHash (Tiết kiệm RAM)
		flat := ds.NewFlatHash()
		flat.Set(field, value)
		shard.items[key] = flat
		return nil
	}

	// Logic Adaptive: Kiểm tra xem có cần Upgrade không
	if obj.IsFlat() {
		// Nếu vượt quá ngưỡng -> Biến hình thành MapHash
		if obj.Len() >= ds.MaxFlatSize {
			flat := obj.(*ds.FlatHash)
			newMap := flat.ToMap()
			newMap.Set(field, value)
			shard.items[key] = newMap // Replace
			return nil
		}
	}

	obj.Set(field, value)
	return nil
}

func (h *HashStore) HGet(key, field string) ([]byte, bool) {
	shard := h.getShard(key)
	shard.mu.RLock()
	defer shard.mu.RUnlock()

	obj, exists := shard.items[key]
	if !exists {
		return nil, false
	}
	return obj.Get(field)
}

func (h *HashStore) HDel(key, field string) bool {
	shard := h.getShard(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	obj, exists := shard.items[key]
	if !exists {
		return false
	}

	deleted := obj.Delete(field)
	if deleted {
		// Clean up nếu rỗng để giải phóng RAM
		if obj.Len() == 0 {
			delete(shard.items, key)
		}
		// TODO (Optional): Downgrade từ Map về Flat nếu size nhỏ lại?
		// Thường thì ít khi làm vậy để tránh thrashing, nhưng có thể implement.
	}
	return deleted
}

func (h *HashStore) HExists(key, field string) bool {
	val, ok := h.HGet(key, field)
	return ok && val != nil
}

func (h *HashStore) HGetAll(key string) map[string][]byte {
	shard := h.getShard(key)
	shard.mu.RLock()
	defer shard.mu.RUnlock()

	obj, exists := shard.items[key]
	if !exists {
		return nil
	}
	return obj.GetAll()
}

func (h *HashStore) Clear() {
	for _, shard := range h.shards {
		shard.mu.Lock()
		shard.items = make(map[string]ds.HashObject)
		shard.mu.Unlock()
	}
}
