package data_types

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"sync"

	al "github.com/AutoCookies/pomai-cache/packages/al/vector"
	ds "github.com/AutoCookies/pomai-cache/packages/ds/vector"
)

const ShardCount = 64

type VectorStore struct {
	shards []*VectorShard
}

type VectorShard struct {
	mu      sync.RWMutex
	indices map[string]*VectorIndex
}

type VectorIndex struct {
	name     string
	dim      int
	arena    *ds.VectorArena
	hnsw     *al.HNSW
	metadata map[uint32]map[string]interface{}
}

func NewVectorStore() *VectorStore {
	vs := &VectorStore{shards: make([]*VectorShard, ShardCount)}
	for i := 0; i < ShardCount; i++ {
		vs.shards[i] = &VectorShard{indices: make(map[string]*VectorIndex)}
	}
	return vs
}

func (vs *VectorStore) getShard(name string) *VectorShard {
	h := fnv.New32a()
	h.Write([]byte(name))
	return vs.shards[h.Sum32()%uint32(ShardCount)]
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

	shard.indices[name] = &VectorIndex{
		name:     name,
		dim:      dim,
		arena:    arena,
		hnsw:     hnsw,
		metadata: make(map[uint32]map[string]interface{}),
	}
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

	err := idx.hnsw.Insert(key, vec)
	if err != nil {
		return err
	}

	if len(meta) > 0 {
		id, found := idx.arena.GetID(key)
		if found {
			shard.mu.Lock()
			idx.metadata[id] = meta
			shard.mu.Unlock()
		}
	}
	return nil
}

func (vs *VectorStore) Search(indexName string, query []float32, k int) ([]VectorSearchResult, error) {
	shard := vs.getShard(indexName)
	shard.mu.RLock()
	idx, ok := shard.indices[indexName]
	shard.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("index not found")
	}

	rawResults := idx.hnsw.Search(query, k)
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

func (vs *VectorStore) Delete(indexName, id string) error {
	shard := vs.getShard(indexName)
	shard.mu.RLock()
	idx, ok := shard.indices[indexName]
	shard.mu.RUnlock()

	if !ok {
		return fmt.Errorf("index not found")
	}

	found := idx.hnsw.Delete(id)
	if !found {
		return nil
	}
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
