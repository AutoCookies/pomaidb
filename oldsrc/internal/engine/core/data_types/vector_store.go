package data_types

import (
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/metrics"
	al "github.com/AutoCookies/pomai-cache/shared/al/vector"
	ds "github.com/AutoCookies/pomai-cache/shared/ds/vector"
)

const ShardCount = 64

type insertReq struct {
	index string
	key   string
	vec   []float32
	meta  map[string]interface{}
}

type VectorStore struct {
	shards      []*VectorShard
	insertQueue chan insertReq
	workers     int
	stopCh      chan struct{}
}

type VectorShard struct {
	mu           sync.RWMutex
	indices      map[string]*VectorIndex
	neighborPool sync.Pool
	idPool       sync.Pool
}

type VectorIndex struct {
	name       string
	dim        int
	arena      *ds.VectorArena
	hnsw       *al.HNSW
	metadata   map[uint32]map[string]interface{}
	latMu      sync.Mutex
	latBuf     []float64
	latPos     int
	latCount   int
	latCap     int
	lastAdjust time.Time
	searches   uint64
	inserts    uint64
}

func NewVectorStore() *VectorStore {
	vs := &VectorStore{
		shards:      make([]*VectorShard, ShardCount),
		insertQueue: make(chan insertReq, 65536),
		stopCh:      make(chan struct{}),
		workers:     0,
	}
	for i := 0; i < ShardCount; i++ {
		vs.shards[i] = &VectorShard{
			indices: make(map[string]*VectorIndex),
			neighborPool: sync.Pool{
				New: func() interface{} { return make([]uint32, 0, 256) },
			},
			idPool: sync.Pool{
				New: func() interface{} { return make([]uint32, 0, 256) },
			},
		}
	}
	go vs.adaptiveLoop()
	return vs
}

func (vs *VectorStore) StartInsertWorkers(n int) {
	if n <= 0 {
		return
	}
	if vs.workers > 0 {
		return
	}
	vs.workers = n
	for i := 0; i < n; i++ {
		go func() {
			for {
				select {
				case req := <-vs.insertQueue:
					_ = vs.Insert(req.index, req.key, req.vec, req.meta)
				case <-vs.stopCh:
					return
				}
			}
		}()
	}
}

func (vs *VectorStore) StopInsertWorkers() {
	if vs.workers == 0 {
		return
	}
	close(vs.stopCh)
	vs.workers = 0
}

func (vs *VectorStore) EnqueueInsert(indexName, key string, vec []float32, meta map[string]interface{}) {
	vecCopy := make([]float32, len(vec))
	copy(vecCopy, vec)
	select {
	case vs.insertQueue <- insertReq{index: indexName, key: key, vec: vecCopy, meta: meta}:
	default:
	}
}

func (vs *VectorStore) getShard(name string) *VectorShard {
	return vs.shards[xxhash.Sum64String(name)%uint64(ShardCount)]
}

func (vs *VectorStore) CreateIndex(name string, dim int, distType string) error {
	shard := vs.getShard(name)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	if _, exists := shard.indices[name]; exists {
		return fmt.Errorf("index exists")
	}

	arena := ds.NewVectorArena(dim, 1000)
	hnsw := al.NewHNSW(dim, 16, 200, arena)
	hnsw.SetNormalizedVectors(true)

	shard.indices[name] = &VectorIndex{
		name:       name,
		dim:        dim,
		arena:      arena,
		hnsw:       hnsw,
		metadata:   make(map[uint32]map[string]interface{}, 1024),
		latBuf:     make([]float64, 1024),
		latCap:     1024,
		lastAdjust: time.Now(),
	}

	metrics.SetEf(name, hnsw.GetEfSearch())
	metrics.SetArenaSize(name, hnsw.Size())

	return nil
}

func (vs *VectorStore) Insert(indexName, key string, vec []float32, meta map[string]interface{}) error {
	shard := vs.getShard(indexName)
	shard.mu.RLock()
	idx, ok := shard.indices[indexName]
	shard.mu.RUnlock()

	if !ok {
		return fmt.Errorf("index not found")
	}

	if idx == nil || idx.hnsw == nil {
		return fmt.Errorf("index not initialized")
	}

	if idx.hnsw.IsNormalized() {
		al.NormalizeInPlace(vec)
	}

	if err := idx.hnsw.Insert(key, vec); err != nil {
		return err
	}
	atomic.AddUint64(&idx.inserts, 1)
	metrics.IncInsert(idx.name)
	metrics.SetArenaSize(idx.name, idx.hnsw.Size())

	if len(meta) > 0 {
		if id, found := idx.arena.GetID(key); found {
			shard.mu.Lock()
			idx.metadata[id] = meta
			shard.mu.Unlock()
		}
	}
	return nil
}

func (vs *VectorStore) Search(indexName string, query []float32, k int) ([]VectorSearchResult, error) {
	start := time.Now()
	shard := vs.getShard(indexName)
	shard.mu.RLock()
	idx, ok := shard.indices[indexName]
	shard.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("index not found")
	}

	if idx.hnsw.IsNormalized() {
		al.NormalizeInPlace(query)
	}

	rawResults := idx.hnsw.Search(query, k)
	elapsed := time.Since(start).Seconds() * 1000.0
	idx.recordLatency(elapsed)
	atomic.AddUint64(&idx.searches, 1)
	metrics.ObserveSearchLatency(idx.name, elapsed)

	results := make([]VectorSearchResult, len(rawResults))

	shard.mu.RLock()
	for i, r := range rawResults {
		key := idx.arena.GetKey(r.ID)
		results[i] = VectorSearchResult{
			ID:       key,
			Distance: r.Distance,
			Metadata: idx.metadata[r.ID],
		}
	}
	shard.mu.RUnlock()

	return results, nil
}

func (idx *VectorIndex) recordLatency(ms float64) {
	idx.latMu.Lock()
	if idx.latCap == 0 {
		idx.latCap = 1024
		idx.latBuf = make([]float64, idx.latCap)
	}
	idx.latBuf[idx.latPos] = ms
	idx.latPos = (idx.latPos + 1) % idx.latCap
	if idx.latCount < idx.latCap {
		idx.latCount++
	}
	idx.latMu.Unlock()
}

func (vs *VectorStore) adaptiveLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		for _, shard := range vs.shards {
			shard.mu.RLock()
			for _, idx := range shard.indices {
				p99 := idx.computeP99()
				if p99 == 0 {
					continue
				}
				currentEf := idx.hnsw.GetEfSearch()
				newEf := currentEf
				if p99 > 80.0 {
					newEf = int(float64(currentEf) * 0.8)
				} else if p99 < 30.0 {
					newEf = int(float64(currentEf) * 1.1)
				}
				if newEf < 1 {
					newEf = 1
				}
				if newEf > 1000 {
					newEf = 1000
				}
				if newEf != currentEf {
					idx.hnsw.SetEfSearch(newEf)
					metrics.SetEf(idx.name, newEf)
				}
				metrics.SetArenaSize(idx.name, idx.hnsw.Size())
			}
			shard.mu.RUnlock()
		}
	}
}

func (idx *VectorIndex) computeP99() float64 {
	idx.latMu.Lock()
	count := idx.latCount
	if count == 0 {
		idx.latMu.Unlock()
		return 0
	}
	buf := make([]float64, count)
	if idx.latPos == 0 && count == idx.latCap {
		copy(buf, idx.latBuf)
	} else {
		start := (idx.latPos - idx.latCount + idx.latCap) % idx.latCap
		for i := 0; i < count; i++ {
			buf[i] = idx.latBuf[(start+i)%idx.latCap]
		}
	}
	idx.latMu.Unlock()

	sort.Float64s(buf)
	pIdx := int(float64(len(buf))*0.99) - 1
	if pIdx < 0 {
		pIdx = 0
	}
	if pIdx >= len(buf) {
		pIdx = len(buf) - 1
	}
	return buf[pIdx]
}

func (vs *VectorStore) Delete(indexName, id string) error {
	shard := vs.getShard(indexName)
	shard.mu.RLock()
	idx, ok := shard.indices[indexName]
	shard.mu.RUnlock()

	if !ok {
		return fmt.Errorf("index not found")
	}

	_ = idx.hnsw.Delete(id)
	metrics.SetArenaSize(idx.name, idx.hnsw.Size())
	return nil
}

func (vs *VectorStore) DropIndex(name string) error {
	shard := vs.getShard(name)
	shard.mu.Lock()
	defer shard.mu.Unlock()
	delete(shard.indices, name)
	return nil
}

func (vs *VectorStore) ListIndices() []string {
	var names []string
	for _, shard := range vs.shards {
		shard.mu.RLock()
		for name := range shard.indices {
			names = append(names, name)
		}
		shard.mu.RUnlock()
	}
	return names
}

func (vs *VectorStore) IndexStats(name string) (map[string]interface{}, error) {
	shard := vs.getShard(name)
	shard.mu.RLock()
	idx, ok := shard.indices[name]
	shard.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("index not found")
	}

	return map[string]interface{}{
		"name":      idx.name,
		"dimension": idx.dim,
		"size":      idx.hnsw.Size(),
		"ef_search": idx.hnsw.GetEfSearch(),
	}, nil
}

func (vs *VectorStore) SetGlobalEfSearch(ef int) {
	for _, shard := range vs.shards {
		shard.mu.RLock()
		for _, idx := range shard.indices {
			idx.hnsw.SetEfSearch(ef)
			metrics.SetEf(idx.name, ef)
		}
		shard.mu.RUnlock()
	}
}

func (vs *VectorStore) HasIndex(name string) bool {
	shard := vs.getShard(name)
	shard.mu.RLock()
	defer shard.mu.RUnlock()
	_, ok := shard.indices[name]
	return ok
}

type VectorSearchResult struct {
	ID       string
	Distance float32
	Metadata map[string]interface{}
}

func (r VectorSearchResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"id":       r.ID,
		"distance": r.Distance,
		"metadata": r.Metadata,
	})
}
