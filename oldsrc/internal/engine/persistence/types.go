// File: internal/engine/persistence/types.go
package persistence

import (
	"io"
)

// SnapshotVersion represents snapshot format version
const (
	SnapshotVersionV1 = 1 // Original KV only
	SnapshotVersionV2 = 2 // KV + ZSet support
)

// SnapshotItem represents serializable cache item
type SnapshotItem struct {
	Type uint8 // 0 = KV, 1 = ZSet

	// KV fields
	Key        string
	Value      []byte
	ExpireAt   int64
	LastAccess int64
	Accesses   uint64
	CreatedAt  int64

	// ZSet fields
	ZMembers []ZMember
}

// ZMember represents sorted set member
type ZMember struct {
	Member string
	Score  float64
}

// Snapshotter defines snapshot operations
type Snapshotter interface {
	SnapshotTo(w io.Writer) error
	RestoreFrom(r io.Reader) error
}

// StoreInterface defines required store methods
type StoreInterface interface {
	GetShards() []ShardInterface
	GetZSets() map[string]ZSetInterface
	GetBloomFilter() BloomFilterInterface
	AddTotalBytes(delta int64)
}

// ShardInterface defines shard operations
type ShardInterface interface {
	Lock()
	Unlock()
	RLock()
	RUnlock()
	GetItems() map[string]interface{}
	SetItem(key string, entry EntryInterface)
	GetBytes() int64
	AddBytes(delta int64)
}

// EntryInterface defines entry operations
type EntryInterface interface {
	Key() string
	Value() []byte
	Size() int
	ExpireAt() int64
	LastAccess() int64
	Accesses() uint64
	CreatedAt() int64
	IsExpired() bool
}

// ZSetInterface defines sorted set operations
type ZSetInterface interface {
	Dump() []ZSetNode
	Add(member string, score float64)
}

// ZSetNode represents sorted set node
type ZSetNode struct {
	Member string
	Score  float64
}

// BloomFilterInterface defines bloom filter operations
type BloomFilterInterface interface {
	Add(key string)
}
