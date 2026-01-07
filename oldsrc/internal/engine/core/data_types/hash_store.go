package data_types

import (
	"errors"
	"hash/fnv"
	"sync"

	"github.com/AutoCookies/pomai-cache/shared/ds/hashtable"
)

const (
	HashShards    = 256
	HashShardMask = HashShards - 1
)

type HashShard struct {
	mu    sync.RWMutex
	items map[string]hashtable.HashObject
}

type HashStore struct {
	shards [HashShards]*HashShard
}

func NewHashStore() *HashStore {
	store := &HashStore{}
	for i := 0; i < HashShards; i++ {
		store.shards[i] = &HashShard{
			items: make(map[string]hashtable.HashObject),
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
		obj = hashtable.NewPacked()
		shard.items[key] = obj
	}

	newObj, upgraded := obj.Set(field, value)
	if upgraded {
		shard.items[key] = newObj
	}

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

func (h *HashStore) HExists(key, field string) bool {
	shard := h.getShard(key)
	shard.mu.RLock()
	defer shard.mu.RUnlock()

	obj, exists := shard.items[key]
	if !exists {
		return false
	}
	_, found := obj.Get(field)
	return found
}

func (h *HashStore) HDel(key, field string) bool {
	shard := h.getShard(key)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	obj, exists := shard.items[key]
	if !exists {
		return false
	}

	deleted, newLen := obj.Delete(field)
	if deleted {
		if newLen == 0 {
			delete(shard.items, key)
		}
	}
	return deleted
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
		shard.items = make(map[string]hashtable.HashObject)
		shard.mu.Unlock()
	}
}

func (h *HashStore) Stats() map[string]interface{} {
	totalKeys := 0
	packedKeys := 0
	pphtKeys := 0
	totalBytes := 0

	for _, s := range h.shards {
		s.mu.RLock()
		totalKeys += len(s.items)
		for _, v := range s.items {
			if v.IsPacked() {
				packedKeys++
			} else {
				pphtKeys++
			}
			totalBytes += v.SizeInBytes()
		}
		s.mu.RUnlock()
	}

	return map[string]interface{}{
		"hash_total":  totalKeys,
		"hash_packed": packedKeys,
		"hash_ppht":   pphtKeys,
		"hash_bytes":  totalBytes,
	}
}
