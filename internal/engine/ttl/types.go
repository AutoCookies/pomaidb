package ttl

type StoreInterface interface {
	GetShard(key string) ShardInterface
	GetShards() []ShardInterface
	GetBloomFilter() BloomFilterInterface
	SetBloomFilter(bloom BloomFilterInterface)
	GetGlobalMemCtrl() MemoryController
	AddTotalBytes(delta int64)
}

type ShardInterface interface {
	Lock()
	Unlock()
	RLock()
	RUnlock()
	GetItems() map[string]interface{}
	GetItemCount() int
	DeleteItem(key string) (size int, ok bool)
	GetBytes() int64
	AddBytes(delta int64)
}

type EntryInterface interface {
	Key() string
	Size() int
	ExpireAt() int64
	IsExpired() bool
	GetHints() []string

	PredictNext() int64
}

type BloomFilterInterface interface {
	MayContain(key string) bool
	Add(key string)
	Clear()
	Count() uint64
}

type MemoryController interface {
	Release(bytes int64)
}
