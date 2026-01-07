package hashtable

import (
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/cespare/xxhash/v2"
)

const (
	// Configuration for "Pomegranate" Segments
	// kSlotNum = 14 to fit within CPU Cache Line (along with metadata)
	kSlotNum     = 14
	kStashNum    = 2
	kGranuleSize = 64 // 64 Bytes alignment
)

// PPE Interface mock
type Predictor interface {
	IsHot(key string) bool
}

type DefaultPredictor struct{}

func (d DefaultPredictor) IsHot(key string) bool { return true }

// -----------------------------------------------------------------------------
// Structure: Granule Slot (Aligned)
// -----------------------------------------------------------------------------

type SlotHeader struct {
	HashTag uint16 // 16-bit fingerprint for fast rejection
	KeyLen  uint16
	ValLen  uint32
}

type Entry struct {
	Key   string
	Value []byte
	Meta  SlotHeader
}

// -----------------------------------------------------------------------------
// Structure: Bucket (The "Fruit")
// -----------------------------------------------------------------------------

type Bucket struct {
	Slots [kSlotNum]Entry
	Stash [kStashNum]Entry
	Count int32
	// Cache line padding to prevent false sharing between buckets
	_ [kGranuleSize - unsafe.Sizeof(int32(0))]byte
}

// -----------------------------------------------------------------------------
// Structure: Layer (The "Skin") - Scalable
// -----------------------------------------------------------------------------

type Layer struct {
	Buckets []Bucket
	Mask    uint64 // len(Buckets) - 1
	Size    uint64 // Total capacity
}

func NewLayer(size uint64) *Layer {
	realSize := uint64(1)
	for realSize < size {
		realSize <<= 1
	}
	return &Layer{
		Buckets: make([]Bucket, realSize),
		Mask:    realSize - 1,
		Size:    realSize,
	}
}

// -----------------------------------------------------------------------------
// Structure: PPHT (Pomai Pomegranate Hash Table)
// -----------------------------------------------------------------------------

type PPHT struct {
	shards    []*pphtShard
	shardMask uint64
	predictor Predictor
}

type pphtShard struct {
	mu     sync.RWMutex
	layers []*Layer
	count  uint64
}

// New creates a new PPHT optimized for concurrent access.
// initialSize: Expected items. shards: Granularity of locks (power of 2).
func New(initialSize int, shards int) *PPHT {
	if shards <= 0 {
		shards = 16 // Default for object-level locking
	}
	numShards := uint64(1)
	for numShards < uint64(shards) {
		numShards <<= 1
	}

	ht := &PPHT{
		shards:    make([]*pphtShard, numShards),
		shardMask: numShards - 1,
		predictor: DefaultPredictor{},
	}

	// Distribute capacity across shards
	sizePerShard := uint64(initialSize) / numShards
	if sizePerShard < 16 {
		sizePerShard = 16
	}

	for i := range ht.shards {
		ht.shards[i] = &pphtShard{
			layers: []*Layer{NewLayer(sizePerShard / kSlotNum)},
		}
	}

	return ht
}

func (ht *PPHT) hash(key string) uint64 {
	return xxhash.Sum64String(key)
}

// --- Implement HashObject Interface ---

func (ht *PPHT) IsPacked() bool { return false }

func (ht *PPHT) Len() int {
	total := uint64(0)
	for _, s := range ht.shards {
		s.mu.RLock()
		total += s.count
		s.mu.RUnlock()
	}
	return int(total)
}

// SizeInBytes returns approximate memory usage
func (ht *PPHT) SizeInBytes() int {
	total := 0
	for _, s := range ht.shards {
		s.mu.RLock()
		for _, l := range s.layers {
			total += len(l.Buckets) * int(unsafe.Sizeof(Bucket{}))
		}
		s.mu.RUnlock()
	}
	return total
}

func (ht *PPHT) Set(key string, value []byte) (HashObject, bool) {
	h := ht.hash(key)
	shardIdx := h & ht.shardMask
	shard := ht.shards[shardIdx]

	shard.mu.Lock()
	defer shard.mu.Unlock()

	tag := uint16(h >> 48)

	// 1. Update existing
	for _, layer := range shard.layers {
		bIdx := h & layer.Mask
		bucket := &layer.Buckets[bIdx]

		for i := 0; i < kSlotNum; i++ {
			if bucket.Slots[i].Meta.HashTag == tag && bucket.Slots[i].Key == key {
				bucket.Slots[i].Value = value
				bucket.Slots[i].Meta.ValLen = uint32(len(value))
				return ht, false // No upgrade needed
			}
		}
		for i := 0; i < kStashNum; i++ {
			if bucket.Stash[i].Meta.HashTag == tag && bucket.Stash[i].Key == key {
				bucket.Stash[i].Value = value
				bucket.Stash[i].Meta.ValLen = uint32(len(value))
				return ht, false
			}
		}
	}

	// 2. Insert new
	ht.insertToLayer(shard, 0, key, value, h, tag)
	return ht, false
}

func (ht *PPHT) insertToLayer(shard *pphtShard, layerIdx int, key string, value []byte, h uint64, tag uint16) {
	// Grow if needed
	if layerIdx >= len(shard.layers) {
		prevSize := shard.layers[len(shard.layers)-1].Size
		newLayer := NewLayer(prevSize * 2)
		shard.layers = append(shard.layers, newLayer)
	}

	layer := shard.layers[layerIdx]
	bIdx := h & layer.Mask
	bucket := &layer.Buckets[bIdx]

	// Try Main Slots
	for i := 0; i < kSlotNum; i++ {
		if bucket.Slots[i].Key == "" {
			bucket.Slots[i] = Entry{
				Key: key, Value: value,
				Meta: SlotHeader{HashTag: tag, KeyLen: uint16(len(key)), ValLen: uint32(len(value))},
			}
			atomic.AddInt32(&bucket.Count, 1)
			shard.count++
			return
		}
	}

	// Try Stash
	for i := 0; i < kStashNum; i++ {
		if bucket.Stash[i].Key == "" {
			bucket.Stash[i] = Entry{
				Key: key, Value: value,
				Meta: SlotHeader{HashTag: tag, KeyLen: uint16(len(key)), ValLen: uint32(len(value))},
			}
			atomic.AddInt32(&bucket.Count, 1)
			shard.count++
			return
		}
	}

	// Cascade to next layer (Cuckoo-like eviction/move could be implemented here, but layering is safer)
	ht.insertToLayer(shard, layerIdx+1, key, value, h, tag)
}

func (ht *PPHT) Get(key string) ([]byte, bool) {
	h := ht.hash(key)
	shard := ht.shards[h&ht.shardMask]

	shard.mu.RLock()
	tag := uint16(h >> 48)

	for lIdx, layer := range shard.layers {
		bIdx := h & layer.Mask
		bucket := &layer.Buckets[bIdx]

		if bucket.Count == 0 {
			continue
		}

		// SIMD-friendly linear scan
		for i := 0; i < kSlotNum; i++ {
			if bucket.Slots[i].Meta.HashTag == tag {
				if bucket.Slots[i].Key == key {
					val := bucket.Slots[i].Value
					shard.mu.RUnlock()

					// PPE Promotion: Move hot items to Layer 0
					if lIdx > 0 && ht.predictor.IsHot(key) {
						go ht.promote(key, val)
					}
					return val, true
				}
			}
		}
		for i := 0; i < kStashNum; i++ {
			if bucket.Stash[i].Meta.HashTag == tag {
				if bucket.Stash[i].Key == key {
					val := bucket.Stash[i].Value
					shard.mu.RUnlock()
					if lIdx > 0 && ht.predictor.IsHot(key) {
						go ht.promote(key, val)
					}
					return val, true
				}
			}
		}
	}
	shard.mu.RUnlock()
	return nil, false
}

func (ht *PPHT) promote(key string, value []byte) {
	// Remove from cold layer and re-insert (which defaults to Layer 0)
	ht.Delete(key)
	ht.Set(key, value)
}

func (ht *PPHT) Delete(key string) (bool, int) {
	h := ht.hash(key)
	shard := ht.shards[h&ht.shardMask]

	shard.mu.Lock()
	defer shard.mu.Unlock()

	tag := uint16(h >> 48)

	for _, layer := range shard.layers {
		bIdx := h & layer.Mask
		bucket := &layer.Buckets[bIdx]

		if bucket.Count == 0 {
			continue
		}

		for i := 0; i < kSlotNum; i++ {
			if bucket.Slots[i].Meta.HashTag == tag && bucket.Slots[i].Key == key {
				bucket.Slots[i] = Entry{} // Clear
				atomic.AddInt32(&bucket.Count, -1)
				shard.count--
				return true, int(shard.count) // Return new local count (approx)
			}
		}
		for i := 0; i < kStashNum; i++ {
			if bucket.Stash[i].Meta.HashTag == tag && bucket.Stash[i].Key == key {
				bucket.Stash[i] = Entry{}
				atomic.AddInt32(&bucket.Count, -1)
				shard.count--
				return true, int(shard.count)
			}
		}
	}
	return false, int(shard.count)
}

func (ht *PPHT) GetAll() map[string][]byte {
	result := make(map[string][]byte)
	for _, s := range ht.shards {
		s.mu.RLock()
		for _, layer := range s.layers {
			for _, b := range layer.Buckets {
				if b.Count == 0 {
					continue
				}
				for i := 0; i < kSlotNum; i++ {
					if b.Slots[i].Key != "" {
						result[b.Slots[i].Key] = b.Slots[i].Value
					}
				}
				for i := 0; i < kStashNum; i++ {
					if b.Stash[i].Key != "" {
						result[b.Stash[i].Key] = b.Stash[i].Value
					}
				}
			}
		}
		s.mu.RUnlock()
	}
	return result
}
