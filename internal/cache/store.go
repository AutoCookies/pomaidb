package cache

import (
	"container/list"
	"context"
	"encoding/gob"
	"errors"
	"hash/fnv"
	"io"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

func init() {
	// Seed the global PRNG once per process.
	rand.Seed(time.Now().UnixNano())
}

// snapshotItem is used for persistence encoding.
type snapshotItem struct {
	Key        string
	Value      []byte
	ExpireAt   int64
	LastAccess int64
	Accesses   uint64
}

type entry struct {
	key        string
	value      []byte
	size       int
	expireAt   int64 // unix nano, 0 means no expiry
	lastAccess int64 // unix nano, used for approximate LRU sampling
	accesses   uint64
}

type shard struct {
	mu    sync.RWMutex
	items map[string]*list.Element
	ll    *list.List
	bytes int64
}

type Store struct {
	shards        []*shard
	shardCount    uint32
	capacityBytes int64

	// total bytes across all shards (atomic)
	totalBytesAtomic int64

	// freqBoost used to combine frequency into priority (nanoseconds multiplier)
	// priority = lastAccess + freqBoost * accesses
	freqBoost int64

	hits      uint64
	misses    uint64
	evictions uint64
}

// NewStore creates a sharded in-memory store with default settings.
// shardCount must be > 0. If <=0 default 256 is used.
// capacityBytes == 0 means "no capacity limit".
func NewStore(shardCount int) *Store {
	return NewStoreWithOptions(shardCount, 0)
}

// NewStoreWithOptions creates a sharded store with specified shard count and capacity in bytes.
func NewStoreWithOptions(shardCount int, capacityBytes int64) *Store {
	if shardCount <= 0 {
		shardCount = 256
	}

	s := &Store{
		shards:        make([]*shard, shardCount),
		shardCount:    uint32(shardCount),
		capacityBytes: capacityBytes,
		// default freqBoost: 1 millisecond per access (in nanoseconds)
		freqBoost: 1_000_000,
	}
	for i := 0; i < shardCount; i++ {
		s.shards[i] = &shard{
			items: make(map[string]*list.Element),
			ll:    list.New(),
		}
	}
	return s
}

// SetFreqBoost sets the boost multiplier (in nanoseconds) applied per access when computing priority.
// Larger values make frequency matter more (i.e., frequently accessed keys are less likely to be evicted).
func (s *Store) SetFreqBoost(nanoseconds int64) {
	atomic.StoreInt64(&s.freqBoost, nanoseconds)
}

func (s *Store) getShard(key string) *shard {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	idx := h.Sum32() % s.shardCount
	return s.shards[int(idx)]
}

// hashToShardIndex computes the shard index for a key.
// (helper to use in Put when calling evictIfNeeded)
func (s *Store) hashToShardIndex(key string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return int(h.Sum32() % s.shardCount)
}

// Put stores a key with value and optional ttl. ttl <= 0 means no expiration.
// A copy of value is stored. If capacityBytes is set, Put will trigger eviction
// loops until the store is under capacity.
func (s *Store) Put(key string, value []byte, ttl time.Duration) {
	if key == "" {
		return
	}
	vcopy := make([]byte, len(value))
	copy(vcopy, value)

	var expireAt int64
	if ttl > 0 {
		expireAt = time.Now().Add(ttl).UnixNano()
	}
	now := time.Now().UnixNano()

	sh := s.getShard(key)
	sh.mu.Lock()
	// update existing
	if elem, ok := sh.items[key]; ok {
		ent := elem.Value.(*entry)
		// adjust bytes
		oldSize := ent.size
		ent.value = vcopy
		ent.size = len(vcopy)
		ent.expireAt = expireAt
		ent.lastAccess = now
		atomic.AddUint64(&ent.accesses, 1)
		sh.bytes += int64(ent.size - oldSize)
		// move to front as most recently used (per-shard ordering for sampling)
		sh.ll.MoveToFront(elem)
		// update global total
		atomic.AddInt64(&s.totalBytesAtomic, int64(ent.size-oldSize))
		sh.mu.Unlock()
	} else {
		// insert new
		ent := &entry{
			key:        key,
			value:      vcopy,
			size:       len(vcopy),
			expireAt:   expireAt,
			lastAccess: now,
			accesses:   1,
		}
		elem := sh.ll.PushFront(ent)
		sh.items[key] = elem
		sh.bytes += int64(ent.size)
		atomic.AddInt64(&s.totalBytesAtomic, int64(ent.size))
		sh.mu.Unlock()
	}

	// Evict if we have a capacity set
	if s.capacityBytes > 0 {
		s.evictIfNeeded(s.hashToShardIndex(key))
	}
}

// Get returns a copy of the value and true if found and not expired.
// If key expired or not present returns (nil, false).
func (s *Store) Get(key string) ([]byte, bool) {
	if key == "" {
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}
	sh := s.getShard(key)

	// Fast path: read lock to check presence/expiry
	sh.mu.RLock()
	elem, ok := sh.items[key]
	if !ok {
		sh.mu.RUnlock()
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}
	ent := elem.Value.(*entry)
	now := time.Now().UnixNano()
	if ent.expireAt != 0 && now > ent.expireAt {
		// expired -> upgrade to write lock to remove
		sh.mu.RUnlock()

		sh.mu.Lock()
		// re-check
		elem2, ok2 := sh.items[key]
		if !ok2 {
			// removed by someone else
			sh.mu.Unlock()
			atomic.AddUint64(&s.misses, 1)
			return nil, false
		}
		ent2 := elem2.Value.(*entry)
		if ent2.expireAt != 0 && time.Now().UnixNano() > ent2.expireAt {
			// still expired -> remove
			delete(sh.items, key)
			sh.ll.Remove(elem2)
			sh.bytes -= int64(ent2.size)
			atomic.AddInt64(&s.totalBytesAtomic, -int64(ent2.size))
			sh.mu.Unlock()
			atomic.AddUint64(&s.misses, 1)
			return nil, false
		}
		// not expired anymore -> move to front, update lastAccess and return
		ent2.lastAccess = time.Now().UnixNano()
		atomic.AddUint64(&ent2.accesses, 1)
		sh.ll.MoveToFront(elem2)
		out := make([]byte, len(ent2.value))
		copy(out, ent2.value)
		sh.mu.Unlock()
		atomic.AddUint64(&s.hits, 1)
		return out, true
	}

	// not expired: we want to bump lastAccess and move to front (update recency).
	// release read lock and acquire write lock to move element safely.
	sh.mu.RUnlock()

	sh.mu.Lock()
	// re-get to avoid races
	elem2, ok2 := sh.items[key]
	if !ok2 {
		// disappeared
		sh.mu.Unlock()
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}
	ent2 := elem2.Value.(*entry)
	// if expired between locks, handle similarly
	if ent2.expireAt != 0 && time.Now().UnixNano() > ent2.expireAt {
		// remove
		delete(sh.items, key)
		sh.ll.Remove(elem2)
		sh.bytes -= int64(ent2.size)
		atomic.AddInt64(&s.totalBytesAtomic, -int64(ent2.size))
		sh.mu.Unlock()
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}
	// update lastAccess and move to front
	ent2.lastAccess = time.Now().UnixNano()
	atomic.AddUint64(&ent2.accesses, 1)
	sh.ll.MoveToFront(elem2)
	out := make([]byte, len(ent2.value))
	copy(out, ent2.value)
	sh.mu.Unlock()
	atomic.AddUint64(&s.hits, 1)
	return out, true
}

// Delete removes a key if present.
func (s *Store) Delete(key string) {
	if key == "" {
		return
	}
	sh := s.getShard(key)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if elem, ok := sh.items[key]; ok {
		ent := elem.Value.(*entry)
		delete(sh.items, key)
		sh.ll.Remove(elem)
		sh.bytes -= int64(ent.size)
		atomic.AddInt64(&s.totalBytesAtomic, -int64(ent.size))
	}
}

// TTLRemaining returns remaining TTL and true if key exists and has expiration.
// If key not found returns (0, false). If key exists without expiration returns (0, true).
func (s *Store) TTLRemaining(key string) (time.Duration, bool) {
	if key == "" {
		return 0, false
	}
	sh := s.getShard(key)
	sh.mu.RLock()
	elem, ok := sh.items[key]
	if !ok {
		sh.mu.RUnlock()
		return 0, false
	}
	ent := elem.Value.(*entry)
	if ent.expireAt == 0 {
		sh.mu.RUnlock()
		return 0, true
	}
	remain := time.Until(time.Unix(0, ent.expireAt))
	if remain <= 0 {
		sh.mu.RUnlock()
		// eager remove
		sh.mu.Lock()
		if elem2, ok2 := sh.items[key]; ok2 {
			ent2 := elem2.Value.(*entry)
			if ent2.expireAt != 0 && time.Now().UnixNano() > ent2.expireAt {
				delete(sh.items, key)
				sh.ll.Remove(elem2)
				sh.bytes -= int64(ent2.size)
				atomic.AddInt64(&s.totalBytesAtomic, -int64(ent2.size))
			}
		}
		sh.mu.Unlock()
		return 0, false
	}
	sh.mu.RUnlock()
	return remain, true
}

// candidate represents a sampled eviction candidate.
type candidate struct {
	shardIdx int
	key      string
	priority int64
	size     int
}

// evictIfNeeded will evict entries while total bytes exceed capacity.
// It uses sampling across shards to pick approximate-LRU/LFU hybrid candidates.
// startShard is used as a hint to start sampling from a shard first.
func (s *Store) evictIfNeeded(startShard int) {
	if s.capacityBytes <= 0 {
		return
	}
	// quick check
	if atomic.LoadInt64(&s.totalBytesAtomic) <= s.capacityBytes {
		return
	}

	shardCount := int(s.shardCount)
	start := startShard % shardCount
	if start < 0 {
		start = 0
	}

	// sampling parameters (tunable)
	const sampleShardLimit = 16 // number of shards to sample per eviction round
	const perShardSamples = 2   // elements sampled per shard
	const maxCandidates = 128   // cap candidates to avoid huge allocations

	for atomic.LoadInt64(&s.totalBytesAtomic) > s.capacityBytes {
		// collect candidates
		cands := make([]candidate, 0, 32)
		// pick a set of shards to sample
		for i := 0; i < sampleShardLimit && i < shardCount; i++ {
			idx := (start + rand.Intn(shardCount)) % shardCount
			sh := s.shards[idx]
			// read-lock shard and sample up to perShardSamples elements from back (LRU side)
			sh.mu.RLock()
			elem := sh.ll.Back()
			count := 0
			for elem != nil && count < perShardSamples {
				ent := elem.Value.(*entry)
				// compute priority = lastAccess + freqBoost * accesses
				priority := ent.lastAccess + int64(atomic.LoadUint64(&ent.accesses))*atomic.LoadInt64(&s.freqBoost)
				cands = append(cands, candidate{
					shardIdx: idx,
					key:      ent.key,
					priority: priority,
					size:     ent.size,
				})
				elem = elem.Prev()
				count++
				if len(cands) >= maxCandidates {
					break
				}
			}
			sh.mu.RUnlock()
			if len(cands) >= maxCandidates {
				break
			}
		}

		if len(cands) == 0 {
			// nothing to evict
			break
		}

		// pick the candidate with smallest priority (old & low-frequency)
		best := cands[0]
		for _, c := range cands[1:] {
			if c.priority < best.priority {
				best = c
			}
		}

		// attempt to evict best candidate: acquire shard write lock and verify still present and priority unchanged
		sh := s.shards[best.shardIdx]
		sh.mu.Lock()
		elem, ok := sh.items[best.key]
		if !ok {
			// someone else removed it, continue
			sh.mu.Unlock()
			start = (start + 1) % shardCount
			continue
		}
		ent := elem.Value.(*entry)
		// re-calc priority
		curPriority := ent.lastAccess + int64(atomic.LoadUint64(&ent.accesses))*atomic.LoadInt64(&s.freqBoost)
		if curPriority != best.priority {
			// changed, skip
			sh.mu.Unlock()
			start = (start + 1) % shardCount
			continue
		}
		// remove
		delete(sh.items, best.key)
		sh.ll.Remove(elem)
		sh.bytes -= int64(ent.size)
		sh.mu.Unlock()

		atomic.AddUint64(&s.evictions, 1)
		atomic.AddInt64(&s.totalBytesAtomic, -int64(ent.size))

		// advance start so next round samples different shards
		start = (start + 1) % shardCount
	}
}

// totalBytesSnapshot returns the approximate total bytes (atomic read).
func (s *Store) totalBytesSnapshot() int64 {
	return atomic.LoadInt64(&s.totalBytesAtomic)
}

func (s *Store) totalBytes() int64 {
	// kept for backward compatibility; reads atomically
	return s.totalBytesSnapshot()
}

// SnapshotTo writes a snapshot of the current store into the given writer using gob encoding.
// Only non-expired keys are written.
func (s *Store) SnapshotTo(w io.Writer) error {
	enc := gob.NewEncoder(w)
	// encode header: version
	if err := enc.Encode(int(1)); err != nil {
		return err
	}

	for _, sh := range s.shards {
		sh.mu.RLock()
		for _, elem := range sh.items {
			ent := elem.Value.(*entry)
			// skip expired
			if ent.expireAt != 0 && time.Now().UnixNano() > ent.expireAt {
				continue
			}
			item := snapshotItem{
				Key:        ent.key,
				Value:      ent.value,
				ExpireAt:   ent.expireAt,
				LastAccess: ent.lastAccess,
				Accesses:   atomic.LoadUint64(&ent.accesses),
			}
			if err := enc.Encode(&item); err != nil {
				sh.mu.RUnlock()
				return err
			}
		}
		sh.mu.RUnlock()
	}
	return nil
}

// RestoreFrom reads a snapshot encoded by SnapshotTo and restores entries into the store.
// Existing entries may be overwritten by snapshot values.
func (s *Store) RestoreFrom(r io.Reader) error {
	dec := gob.NewDecoder(r)
	var version int
	if err := dec.Decode(&version); err != nil {
		return err
	}
	if version != 1 {
		return errors.New("unsupported snapshot version")
	}

	for {
		var item snapshotItem
		if err := dec.Decode(&item); err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		// skip expired items
		if item.ExpireAt != 0 && time.Now().UnixNano() > item.ExpireAt {
			continue
		}

		// insert directly into shard to preserve lastAccess and accesses
		sh := s.getShard(item.Key)
		sh.mu.Lock()
		// if exists, subtract old size and remove
		if elem, ok := sh.items[item.Key]; ok {
			old := elem.Value.(*entry)
			delete(sh.items, item.Key)
			sh.ll.Remove(elem)
			sh.bytes -= int64(old.size)
			atomic.AddInt64(&s.totalBytesAtomic, -int64(old.size))
		}
		ent := &entry{
			key:        item.Key,
			value:      item.Value,
			size:       len(item.Value),
			expireAt:   item.ExpireAt,
			lastAccess: item.LastAccess,
			accesses:   item.Accesses,
		}
		elem := sh.ll.PushFront(ent)
		sh.items[item.Key] = elem
		sh.bytes += int64(ent.size)
		atomic.AddInt64(&s.totalBytesAtomic, int64(ent.size))
		sh.mu.Unlock()
	}
}

// Stats holds basic store metrics.
type Stats struct {
	Hits       uint64
	Misses     uint64
	Items      int64
	Bytes      int64
	Capacity   int64
	Evictions  uint64
	ShardCount int
	FreqBoost  int64
}

// Stats returns aggregated statistics.
func (s *Store) Stats() Stats {
	var totalItems int64
	var totalBytes int64
	for _, sh := range s.shards {
		sh.mu.RLock()
		totalItems += int64(len(sh.items))
		totalBytes += sh.bytes
		sh.mu.RUnlock()
	}
	return Stats{
		Hits:       atomic.LoadUint64(&s.hits),
		Misses:     atomic.LoadUint64(&s.misses),
		Items:      totalItems,
		Bytes:      totalBytes,
		Capacity:   s.capacityBytes,
		Evictions:  atomic.LoadUint64(&s.evictions),
		ShardCount: int(s.shardCount),
		FreqBoost:  atomic.LoadInt64(&s.freqBoost),
	}
}

func (s *Store) StartCleanup(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.cleanupExpired()
			}
		}
	}()
}

func (s *Store) cleanupExpired() {
	now := time.Now().UnixNano()
	cleaned := 0

	for _, sh := range s.shards {
		sh.mu.Lock()
		toDelete := []string{}

		for key, elem := range sh.items {
			ent := elem.Value.(*entry)
			if ent.expireAt != 0 && now > ent.expireAt {
				toDelete = append(toDelete, key)
			}
		}

		for _, key := range toDelete {
			if elem, ok := sh.items[key]; ok {
				ent := elem.Value.(*entry)
				delete(sh.items, key)
				sh.ll.Remove(elem)
				sh.bytes -= int64(ent.size)
				atomic.AddInt64(&s.totalBytesAtomic, -int64(ent.size))
				cleaned++
			}
		}

		sh.mu.Unlock()
	}

	if cleaned > 0 {
		log.Printf("[CLEANUP] Removed %d expired keys across all users", cleaned)
	}
}

func (s *Store) CleanupExpired() int {
	now := time.Now().UnixNano()
	cleaned := 0

	for _, sh := range s.shards {
		sh.mu.Lock()
		toDelete := []string{}

		for key, elem := range sh.items {
			ent := elem.Value.(*entry)
			if ent.expireAt != 0 && now > ent.expireAt {
				toDelete = append(toDelete, key)
			}
		}

		for _, key := range toDelete {
			if elem, ok := sh.items[key]; ok {
				ent := elem.Value.(*entry)
				delete(sh.items, key)
				sh.ll.Remove(elem)
				sh.bytes -= int64(ent.size)
				atomic.AddInt64(&s.totalBytesAtomic, -int64(ent.size))
				cleaned++
			}
		}

		sh.mu.Unlock()
	}

	return cleaned
}
