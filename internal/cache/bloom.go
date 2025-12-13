// File: internal/cache/bloom. go
package cache

import (
	"hash/fnv"
	"sync/atomic"
)

type BloomFilter struct {
	bits []uint64
	size uint64
	k    uint64 // number of hash functions
}

func NewBloomFilter(size uint64, k uint64) *BloomFilter {
	numWords := (size + 63) / 64
	return &BloomFilter{
		bits: make([]uint64, numWords),
		size: size,
		k:    k,
	}
}

func (b *BloomFilter) Add(key string) {
	for i := uint64(0); i < b.k; i++ {
		idx := b.hash(key, i) % b.size
		wordIdx := idx / 64
		bitIdx := idx % 64

		old := atomic.LoadUint64(&b.bits[wordIdx])
		atomic.StoreUint64(&b.bits[wordIdx], old|(1<<bitIdx))
	}
}

func (b *BloomFilter) MayContain(key string) bool {
	for i := uint64(0); i < b.k; i++ {
		idx := b.hash(key, i) % b.size
		wordIdx := idx / 64
		bitIdx := idx % 64

		if atomic.LoadUint64(&b.bits[wordIdx])&(1<<bitIdx) == 0 {
			return false // Definitely not present
		}
	}
	return true // Possibly present
}

func (b *BloomFilter) hash(key string, seed uint64) uint64 {
	h := fnv.New64a()
	h.Write([]byte(key))
	h.Write([]byte{byte(seed)})
	return h.Sum64()
}
