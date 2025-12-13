// File: internal/cache/store.go
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
	rand.Seed(time.Now().UnixNano())
}

type snapshotItem struct {
	Key        string
	Value      []byte
	ExpireAt   int64
	LastAccess int64
	Accesses   uint64
	CreatedAt  int64
}

type entry struct {
	key        string
	value      []byte
	size       int
	expireAt   int64
	lastAccess int64
	accesses   uint64
	createdAt  int64
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

	totalBytesAtomic int64
	freqBoost        int64

	hits      uint64
	misses    uint64
	evictions uint64

	adaptiveTTL *AdaptiveTTL // ✅ Adaptive TTL engine
	bloom       *BloomFilter // ✅ Bloom filter for fast negative lookups
	bloomStats  BloomStats   // ✅ Bloom filter statistics
}

// ✅ Bloom filter statistics
type BloomStats struct {
	Hits              uint64 // Bloom said "maybe" and key was found
	Misses            uint64 // Bloom said "maybe" but key not found (false positive)
	Avoided           uint64 // Bloom said "no" - avoided shard lock
	FalsePositiveRate float64
}

func NewStore(shardCount int) *Store {
	return NewStoreWithOptions(shardCount, 0)
}

func NewStoreWithOptions(shardCount int, capacityBytes int64) *Store {
	if shardCount <= 0 {
		shardCount = 256
	}

	s := &Store{
		shards:        make([]*shard, shardCount),
		shardCount:    uint32(shardCount),
		capacityBytes: capacityBytes,
		freqBoost:     1_000_000,
	}
	for i := 0; i < shardCount; i++ {
		s.shards[i] = &shard{
			items: make(map[string]*list.Element),
			ll:    list.New(),
		}
	}
	return s
}

// ✅ EnableBloomFilter enables bloom filter for fast negative lookups
// size: number of bits (e.g., 10_000_000 for ~1.2MB memory)
// k: number of hash functions (3-5 is typical, 4 is optimal for ~1% false positive rate)
func (s *Store) EnableBloomFilter(size uint64, k uint64) {
	s.bloom = NewBloomFilter(size, k)
	log.Printf("[BLOOM FILTER] Enabled:  size=%d bits (~%. 2fMB), k=%d hash functions",
		size, float64(size)/8/1024/1024, k)

	// Populate bloom filter with existing keys
	populated := 0
	for _, sh := range s.shards {
		sh.mu.RLock()
		for key := range sh.items {
			s.bloom.Add(key)
			populated++
		}
		sh.mu.RUnlock()
	}

	if populated > 0 {
		log.Printf("[BLOOM FILTER] Populated with %d existing keys", populated)
	}
}

// ✅ DisableBloomFilter disables bloom filter
func (s *Store) DisableBloomFilter() {
	s.bloom = nil
	log.Println("[BLOOM FILTER] Disabled")
}

// ✅ GetBloomStats returns bloom filter statistics
func (s *Store) GetBloomStats() BloomStats {
	hits := atomic.LoadUint64(&s.bloomStats.Hits)
	misses := atomic.LoadUint64(&s.bloomStats.Misses)
	avoided := atomic.LoadUint64(&s.bloomStats.Avoided)

	stats := BloomStats{
		Hits:    hits,
		Misses:  misses,
		Avoided: avoided,
	}

	// Calculate false positive rate
	if hits+misses > 0 {
		stats.FalsePositiveRate = float64(misses) / float64(hits+misses) * 100
	}

	return stats
}

func (s *Store) EnableAdaptiveTTL(minTTL, maxTTL time.Duration) {
	s.adaptiveTTL = NewAdaptiveTTL(minTTL, maxTTL)
	log.Printf("[ADAPTIVE TTL] Enabled: min=%v, max=%v", minTTL, maxTTL)
}

func (s *Store) DisableAdaptiveTTL() {
	s.adaptiveTTL = nil
	log.Println("[ADAPTIVE TTL] Disabled")
}

func (s *Store) SetFreqBoost(nanoseconds int64) {
	atomic.StoreInt64(&s.freqBoost, nanoseconds)
}

func (s *Store) getShard(key string) *shard {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	idx := h.Sum32() % s.shardCount
	return s.shards[int(idx)]
}

func (s *Store) hashToShardIndex(key string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(key))
	return int(h.Sum32() % s.shardCount)
}

func (s *Store) Put(key string, value []byte, ttl time.Duration) {
	if key == "" {
		return
	}
	vcopy := make([]byte, len(value))
	copy(vcopy, value)

	now := time.Now().UnixNano()
	var expireAt int64

	sh := s.getShard(key)
	sh.mu.Lock()

	if elem, ok := sh.items[key]; ok {
		ent := elem.Value.(*entry)
		oldSize := ent.size

		if s.adaptiveTTL != nil && ttl > 0 {
			age := time.Since(time.Unix(0, ent.createdAt))
			accesses := atomic.LoadUint64(&ent.accesses)
			adaptiveTTL := s.adaptiveTTL.ComputeTTL(accesses, age)
			expireAt = time.Now().Add(adaptiveTTL).UnixNano()
		} else if ttl > 0 {
			expireAt = time.Now().Add(ttl).UnixNano()
		}

		ent.value = vcopy
		ent.size = len(vcopy)
		ent.expireAt = expireAt
		ent.lastAccess = now
		atomic.AddUint64(&ent.accesses, 1)
		sh.bytes += int64(ent.size - oldSize)
		sh.ll.MoveToFront(elem)
		atomic.AddInt64(&s.totalBytesAtomic, int64(ent.size-oldSize))
		sh.mu.Unlock()
	} else {
		if ttl > 0 {
			if s.adaptiveTTL != nil {
				adaptiveTTL := s.adaptiveTTL.ComputeTTL(0, 0)
				expireAt = time.Now().Add(adaptiveTTL).UnixNano()
			} else {
				expireAt = time.Now().Add(ttl).UnixNano()
			}
		}

		ent := &entry{
			key:        key,
			value:      vcopy,
			size:       len(vcopy),
			expireAt:   expireAt,
			lastAccess: now,
			accesses:   1,
			createdAt:  now,
		}
		elem := sh.ll.PushFront(ent)
		sh.items[key] = elem
		sh.bytes += int64(ent.size)
		atomic.AddInt64(&s.totalBytesAtomic, int64(ent.size))

		// ✅ Add to bloom filter
		if s.bloom != nil {
			s.bloom.Add(key)
		}

		sh.mu.Unlock()
	}

	if s.capacityBytes > 0 {
		s.evictIfNeeded(s.hashToShardIndex(key))
	}
}

func (s *Store) PutAdaptive(key string, value []byte, baseTTL time.Duration) {
	s.Put(key, value, baseTTL)
}

func (s *Store) RefreshTTL(key string) bool {
	if key == "" || s.adaptiveTTL == nil {
		return false
	}

	sh := s.getShard(key)
	sh.mu.Lock()
	defer sh.mu.Unlock()

	elem, ok := sh.items[key]
	if !ok {
		return false
	}

	ent := elem.Value.(*entry)
	if ent.expireAt == 0 {
		return false
	}

	age := time.Since(time.Unix(0, ent.createdAt))
	accesses := atomic.LoadUint64(&ent.accesses)
	newTTL := s.adaptiveTTL.ComputeTTL(accesses, age)
	ent.expireAt = time.Now().Add(newTTL).UnixNano()

	return true
}

func (s *Store) Get(key string) ([]byte, bool) {
	if key == "" {
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}

	// ✅ Fast path: Check bloom filter first
	if s.bloom != nil {
		if !s.bloom.MayContain(key) {
			// Definitely not present - avoid shard lock!
			atomic.AddUint64(&s.misses, 1)
			atomic.AddUint64(&s.bloomStats.Avoided, 1)
			return nil, false
		}
		// Bloom says "maybe" - need to check actual store
	}

	sh := s.getShard(key)

	sh.mu.RLock()
	elem, ok := sh.items[key]
	if !ok {
		sh.mu.RUnlock()
		atomic.AddUint64(&s.misses, 1)

		// ✅ Bloom filter said "maybe" but key not found (false positive)
		if s.bloom != nil {
			atomic.AddUint64(&s.bloomStats.Misses, 1)
		}

		return nil, false
	}

	// ✅ Bloom filter was correct (true positive)
	if s.bloom != nil {
		atomic.AddUint64(&s.bloomStats.Hits, 1)
	}

	ent := elem.Value.(*entry)
	now := time.Now().UnixNano()
	if ent.expireAt != 0 && now > ent.expireAt {
		sh.mu.RUnlock()

		sh.mu.Lock()
		elem2, ok2 := sh.items[key]
		if !ok2 {
			sh.mu.Unlock()
			atomic.AddUint64(&s.misses, 1)
			return nil, false
		}
		ent2 := elem2.Value.(*entry)
		if ent2.expireAt != 0 && time.Now().UnixNano() > ent2.expireAt {
			delete(sh.items, key)
			sh.ll.Remove(elem2)
			sh.bytes -= int64(ent2.size)
			atomic.AddInt64(&s.totalBytesAtomic, -int64(ent2.size))

			// Note: We don't remove from bloom filter (it's append-only)
			// False positives will eventually be fixed by bloom filter rebuild

			sh.mu.Unlock()
			atomic.AddUint64(&s.misses, 1)
			return nil, false
		}
		ent2.lastAccess = time.Now().UnixNano()
		atomic.AddUint64(&ent2.accesses, 1)
		sh.ll.MoveToFront(elem2)
		out := make([]byte, len(ent2.value))
		copy(out, ent2.value)
		sh.mu.Unlock()
		atomic.AddUint64(&s.hits, 1)
		return out, true
	}

	sh.mu.RUnlock()

	sh.mu.Lock()
	elem2, ok2 := sh.items[key]
	if !ok2 {
		sh.mu.Unlock()
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}
	ent2 := elem2.Value.(*entry)
	if ent2.expireAt != 0 && time.Now().UnixNano() > ent2.expireAt {
		delete(sh.items, key)
		sh.ll.Remove(elem2)
		sh.bytes -= int64(ent2.size)
		atomic.AddInt64(&s.totalBytesAtomic, -int64(ent2.size))
		sh.mu.Unlock()
		atomic.AddUint64(&s.misses, 1)
		return nil, false
	}
	ent2.lastAccess = time.Now().UnixNano()
	atomic.AddUint64(&ent2.accesses, 1)
	sh.ll.MoveToFront(elem2)

	if s.adaptiveTTL != nil && ent2.expireAt != 0 {
		age := time.Since(time.Unix(0, ent2.createdAt))
		accesses := atomic.LoadUint64(&ent2.accesses)
		newTTL := s.adaptiveTTL.ComputeTTL(accesses, age)
		ent2.expireAt = time.Now().Add(newTTL).UnixNano()
	}

	out := make([]byte, len(ent2.value))
	copy(out, ent2.value)
	sh.mu.Unlock()
	atomic.AddUint64(&s.hits, 1)
	return out, true
}

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

		// Note:  Bloom filter is append-only, we don't remove keys
		// This creates false positives but they're harmless
	}
}

// ✅ RebuildBloomFilter rebuilds bloom filter from scratch
// Call this periodically to reduce false positive rate after many deletions
func (s *Store) RebuildBloomFilter() int {
	if s.bloom == nil {
		return 0
	}

	// Get bloom filter config
	oldSize := s.bloom.size
	oldK := s.bloom.k

	// Create new bloom filter
	s.bloom = NewBloomFilter(oldSize, oldK)

	// Repopulate with current keys
	count := 0
	for _, sh := range s.shards {
		sh.mu.RLock()
		for key := range sh.items {
			s.bloom.Add(key)
			count++
		}
		sh.mu.RUnlock()
	}

	// Reset bloom stats
	atomic.StoreUint64(&s.bloomStats.Hits, 0)
	atomic.StoreUint64(&s.bloomStats.Misses, 0)
	atomic.StoreUint64(&s.bloomStats.Avoided, 0)

	log.Printf("[BLOOM FILTER] Rebuilt with %d keys", count)
	return count
}

func (s *Store) TTLRemaining(key string) (time.Duration, bool) {
	if key == "" {
		return 0, false
	}

	// ✅ Bloom filter check
	if s.bloom != nil && !s.bloom.MayContain(key) {
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

type candidate struct {
	shardIdx int
	key      string
	priority int64
	size     int
}

func (s *Store) evictIfNeeded(startShard int) {
	if s.capacityBytes <= 0 {
		return
	}
	if atomic.LoadInt64(&s.totalBytesAtomic) <= s.capacityBytes {
		return
	}

	shardCount := int(s.shardCount)
	start := startShard % shardCount
	if start < 0 {
		start = 0
	}

	const sampleShardLimit = 16
	const perShardSamples = 2
	const maxCandidates = 128

	for atomic.LoadInt64(&s.totalBytesAtomic) > s.capacityBytes {
		cands := make([]candidate, 0, 32)
		for i := 0; i < sampleShardLimit && i < shardCount; i++ {
			idx := (start + rand.Intn(shardCount)) % shardCount
			sh := s.shards[idx]
			sh.mu.RLock()
			elem := sh.ll.Back()
			count := 0
			for elem != nil && count < perShardSamples {
				ent := elem.Value.(*entry)
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
			break
		}

		best := cands[0]
		for _, c := range cands[1:] {
			if c.priority < best.priority {
				best = c
			}
		}

		sh := s.shards[best.shardIdx]
		sh.mu.Lock()
		elem, ok := sh.items[best.key]
		if !ok {
			sh.mu.Unlock()
			start = (start + 1) % shardCount
			continue
		}
		ent := elem.Value.(*entry)
		curPriority := ent.lastAccess + int64(atomic.LoadUint64(&ent.accesses))*atomic.LoadInt64(&s.freqBoost)
		if curPriority != best.priority {
			sh.mu.Unlock()
			start = (start + 1) % shardCount
			continue
		}
		delete(sh.items, best.key)
		sh.ll.Remove(elem)
		sh.bytes -= int64(ent.size)
		sh.mu.Unlock()

		atomic.AddUint64(&s.evictions, 1)
		atomic.AddInt64(&s.totalBytesAtomic, -int64(ent.size))

		start = (start + 1) % shardCount
	}
}

func (s *Store) totalBytesSnapshot() int64 {
	return atomic.LoadInt64(&s.totalBytesAtomic)
}

func (s *Store) totalBytes() int64 {
	return s.totalBytesSnapshot()
}

func (s *Store) SnapshotTo(w io.Writer) error {
	enc := gob.NewEncoder(w)
	if err := enc.Encode(int(1)); err != nil {
		return err
	}

	for _, sh := range s.shards {
		sh.mu.RLock()
		for _, elem := range sh.items {
			ent := elem.Value.(*entry)
			if ent.expireAt != 0 && time.Now().UnixNano() > ent.expireAt {
				continue
			}
			item := snapshotItem{
				Key:        ent.key,
				Value:      ent.value,
				ExpireAt:   ent.expireAt,
				LastAccess: ent.lastAccess,
				Accesses:   atomic.LoadUint64(&ent.accesses),
				CreatedAt:  ent.createdAt,
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
		if item.ExpireAt != 0 && time.Now().UnixNano() > item.ExpireAt {
			continue
		}

		sh := s.getShard(item.Key)
		sh.mu.Lock()
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
			createdAt:  item.CreatedAt,
		}
		elem := sh.ll.PushFront(ent)
		sh.items[item.Key] = elem
		sh.bytes += int64(ent.size)
		atomic.AddInt64(&s.totalBytesAtomic, int64(ent.size))

		// ✅ Add restored keys to bloom filter
		if s.bloom != nil {
			s.bloom.Add(item.Key)
		}

		sh.mu.Unlock()
	}
}

type Stats struct {
	Hits           uint64
	Misses         uint64
	Items          int64
	Bytes          int64
	Capacity       int64
	Evictions      uint64
	ShardCount     int
	FreqBoost      int64
	AdaptiveTTL    bool
	AdaptiveMinTTL string
	AdaptiveMaxTTL string
	BloomEnabled   bool    // ✅ Bloom filter status
	BloomFPRate    float64 // ✅ False positive rate
	BloomAvoided   uint64  // ✅ Lookups avoided by bloom
}

func (s *Store) Stats() Stats {
	var totalItems int64
	var totalBytes int64
	for _, sh := range s.shards {
		sh.mu.RLock()
		totalItems += int64(len(sh.items))
		totalBytes += sh.bytes
		sh.mu.RUnlock()
	}

	stats := Stats{
		Hits:         atomic.LoadUint64(&s.hits),
		Misses:       atomic.LoadUint64(&s.misses),
		Items:        totalItems,
		Bytes:        totalBytes,
		Capacity:     s.capacityBytes,
		Evictions:    atomic.LoadUint64(&s.evictions),
		ShardCount:   int(s.shardCount),
		FreqBoost:    atomic.LoadInt64(&s.freqBoost),
		AdaptiveTTL:  s.adaptiveTTL != nil,
		BloomEnabled: s.bloom != nil,
	}

	if s.adaptiveTTL != nil {
		stats.AdaptiveMinTTL = s.adaptiveTTL.minTTL.String()
		stats.AdaptiveMaxTTL = s.adaptiveTTL.maxTTL.String()
	}

	if s.bloom != nil {
		bloomStats := s.GetBloomStats()
		stats.BloomFPRate = bloomStats.FalsePositiveRate
		stats.BloomAvoided = bloomStats.Avoided
	}

	return stats
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

		// ✅ Rebuild bloom filter after cleanup if many keys were removed
		if s.bloom != nil && cleaned > 1000 {
			s.RebuildBloomFilter()
		}
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
