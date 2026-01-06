package vector

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"

	ds "github.com/AutoCookies/pomai-cache/shared/ds/vector"
)

const (
	DefaultM              = 16
	DefaultEfConstruction = 200
	DefaultEfSearch       = 50
)

type HNSW struct {
	m              int
	mMax           int
	mMax0          int
	efConstruction int
	efSearch       atomic.Int32
	ml             float64
	maxLevel       int
	arena          *ds.VectorArena
	entryPoint     uint32
	hasEntry       bool
	distFunc       DistanceFunc
	visitedPool    *sync.Pool
	resultPool     *sync.Pool
	neighborPool   *sync.Pool
	vecBufPool     *sync.Pool
	rng            *rand.Rand
	mu             sync.RWMutex
	normalized     bool
}

func NewHNSW(dim, m, efConstruction int, arena *ds.VectorArena) *HNSW {
	h := &HNSW{
		m:              m,
		mMax:           m,
		mMax0:          m * 2,
		efConstruction: efConstruction,
		ml:             1.0 / math.Log(float64(m)),
		maxLevel:       16,
		arena:          arena,
		distFunc:       CosineSIMD,
		rng:            rand.New(rand.NewSource(rand.Int63())),
		visitedPool: &sync.Pool{
			New: func() interface{} { return make([]bool, 0, 1024) },
		},
		resultPool: &sync.Pool{
			New: func() interface{} { return &resultHeap{data: make([]SearchResult, 0, 64)} },
		},
		neighborPool: &sync.Pool{
			New: func() interface{} { return make([]uint32, 0, 256) },
		},
		vecBufPool: &sync.Pool{
			New: func() interface{} { return make([][]float32, 0, 16) },
		},
	}
	h.efSearch.Store(int32(DefaultEfSearch))
	return h
}

func (h *HNSW) SetNormalizedVectors(n bool) {
	h.normalized = n
	if n {
		h.distFunc = nil
	} else {
		h.distFunc = CosineSIMD
	}
}

func (h *HNSW) IsNormalized() bool {
	return h.normalized
}

func (h *HNSW) GetEfSearch() int {
	return int(h.efSearch.Load())
}

func (h *HNSW) SetEfSearch(ef int) {
	if ef < 1 {
		ef = 1
	}
	h.efSearch.Store(int32(ef))
}

func (h *HNSW) Size() int {
	if h.arena == nil {
		return 0
	}
	return h.arena.Size()
}

func (h *HNSW) Insert(key string, vec []float32) error {
	level := h.randomLevel()
	id, isNew := h.arena.AllocNode(key, vec, level)
	if !isNew {
		return nil
	}

	h.mu.RLock()
	entryID := h.entryPoint
	hasEntry := h.hasEntry
	h.mu.RUnlock()

	if !hasEntry {
		h.mu.Lock()
		h.entryPoint = id
		h.hasEntry = true
		h.mu.Unlock()
		return nil
	}

	currObj := entryID
	currDist := h.dist(vec, currObj)

	entryNode := h.getNode(entryID)
	maxL := entryNode.Level

	for l := maxL; l > level; l-- {
		changed := true
		for changed {
			changed = false
			currNode := h.getNode(currObj)
			currNode.Mu.RLock()
			conns := currNode.Connections[l]
			candidates := make([]uint32, len(conns))
			copy(candidates, conns)
			currNode.Mu.RUnlock()

			for _, candID := range candidates {
				d := h.dist(vec, candID)
				if d < currDist {
					currDist = d
					currObj = candID
					changed = true
				}
			}
		}
	}

	for l := int(math.Min(float64(maxL), float64(level))); l >= 0; l-- {
		candidates := h.searchLayer(vec, currObj, h.efConstruction, l)
		neighbors := h.selectNeighbors(candidates, h.m)

		nodeData := h.getNode(id)
		for _, n := range neighbors {
			nodeData.Connections[l] = append(nodeData.Connections[l], n.ID)
		}

		for _, n := range neighbors {
			neighborID := n.ID
			neighborNode := h.getNode(neighborID)
			neighborNode.Mu.Lock()
			neighborNode.Connections[l] = append(neighborNode.Connections[l], id)
			neighborNode.Mu.Unlock()
		}

		if len(candidates) > 0 {
			currObj = candidates[0].ID
		}
	}

	h.mu.Lock()
	if level > h.getNode(h.entryPoint).Level {
		h.entryPoint = id
	}
	h.mu.Unlock()

	return nil
}

func (h *HNSW) Search(query []float32, k int) []SearchResult {
	h.mu.RLock()
	entryID := h.entryPoint
	hasEntry := h.hasEntry
	h.mu.RUnlock()

	if !hasEntry {
		return nil
	}

	ef := int(h.efSearch.Load())
	if k > ef {
		ef = k
	}

	currObj := entryID
	entryNode := h.getNode(entryID)
	maxL := entryNode.Level
	currDist := h.dist(query, currObj)

	for l := maxL; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			currNode := h.getNode(currObj)
			currNode.Mu.RLock()
			conns := currNode.Connections[l]
			candidates := make([]uint32, len(conns))
			copy(candidates, conns)
			currNode.Mu.RUnlock()

			for _, candID := range candidates {
				d := h.dist(query, candID)
				if d < currDist {
					currDist = d
					currObj = candID
					changed = true
				}
			}
		}
	}

	results := h.searchLayer(query, currObj, ef, 0)
	if len(results) > k {
		results = results[:k]
	}
	return results
}

func (h *HNSW) searchLayer(query []float32, entryPoint uint32, ef int, level int) []SearchResult {
	visited := h.visitedPool.Get().([]bool)
	currentSize := h.arena.Size()

	if cap(visited) < currentSize {
		visited = make([]bool, currentSize+1024)
	} else {
		visited = visited[:currentSize]
		for i := range visited {
			visited[i] = false
		}
	}
	defer h.visitedPool.Put(visited)

	candidates := h.resultPool.Get().(*resultHeap)
	candidates.data = candidates.data[:0]
	results := h.resultPool.Get().(*resultHeap)
	results.data = results.data[:0]
	defer h.resultPool.Put(candidates)
	defer h.resultPool.Put(results)

	d := h.dist(query, entryPoint)
	res := SearchResult{ID: entryPoint, Distance: d}
	heap.Push(candidates, res)
	heap.Push(results, res)

	if entryPoint < uint32(len(visited)) {
		visited[entryPoint] = true
	}

	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(SearchResult)
		if curr.Distance > results.data[0].Distance {
			break
		}

		currNode := h.getNode(curr.ID)
		if currNode == nil {
			continue
		}

		currNode.Mu.RLock()
		conns := currNode.Connections[level]

		neighBuf := h.neighborPool.Get().([]uint32)
		if cap(neighBuf) < len(conns) {
			neighBuf = make([]uint32, 0, len(conns))
		} else {
			neighBuf = neighBuf[:0]
		}
		neighBuf = append(neighBuf, conns...)
		currNode.Mu.RUnlock()

		validNeighbors := neighBuf[:0]
		for _, nID := range neighBuf {
			if nID >= uint32(len(visited)) {
				continue
			}
			if !visited[nID] {
				visited[nID] = true
				if !h.arena.IsDeleted(nID) {
					validNeighbors = append(validNeighbors, nID)
				}
			}
		}

		batchSize := 16
		for i := 0; i < len(validNeighbors); i += batchSize {
			end := i + batchSize
			if end > len(validNeighbors) {
				end = len(validNeighbors)
			}
			batch := validNeighbors[i:end]
			vecs := h.arena.GetVectors(batch)
			for j, nID := range batch {
				vec := vecs[j]
				var dist float32
				if h.normalized {
					dist = 1.0 - DotProductSIMD(query, vec)
				} else if h.distFunc != nil {
					dist = h.distFunc(query, vec)
				} else {
					dist = CosineSIMD(query, vec)
				}

				if dist < results.data[0].Distance || results.Len() < ef {
					neighborRes := SearchResult{ID: nID, Distance: dist}
					heap.Push(candidates, neighborRes)
					heap.Push(results, neighborRes)
					if results.Len() > ef {
						heap.Pop(results)
					}
				}
			}
		}

		h.neighborPool.Put(neighBuf)
	}

	finalRes := make([]SearchResult, results.Len())
	for i := len(finalRes) - 1; i >= 0; i-- {
		finalRes[i] = heap.Pop(results).(SearchResult)
	}
	return finalRes
}

func (h *HNSW) dist(v []float32, id uint32) float32 {
	vec := h.arena.GetVector(id)
	if h.normalized {
		return 1.0 - DotProductSIMD(v, vec)
	}
	if h.distFunc != nil {
		return h.distFunc(v, vec)
	}
	return CosineSIMD(v, vec)
}

func (h *HNSW) getNode(id uint32) *ds.NodeData {
	return h.arena.GetNode(id)
}

func (h *HNSW) randomLevel() int {
	lvl := 0
	for h.rng.Float64() < h.ml && lvl < h.maxLevel {
		lvl++
	}
	return lvl
}

func (h *HNSW) Delete(key string) bool {
	h.mu.RLock()
	id, exists := h.arena.GetID(key)
	h.mu.RUnlock()
	if !exists {
		return false
	}
	h.arena.MarkDeleted(id)
	return true
}

type SearchResult struct {
	ID       uint32
	Distance float32
}

type resultHeap struct{ data []SearchResult }

func (h *resultHeap) Len() int           { return len(h.data) }
func (h *resultHeap) Less(i, j int) bool { return h.data[i].Distance > h.data[j].Distance }
func (h *resultHeap) Swap(i, j int)      { h.data[i], h.data[j] = h.data[j], h.data[i] }
func (h *resultHeap) Push(x interface{}) { h.data = append(h.data, x.(SearchResult)) }
func (h *resultHeap) Pop() interface{} {
	n := len(h.data)
	x := h.data[n-1]
	h.data = h.data[0 : n-1]
	return x
}

func (h *HNSW) selectNeighbors(candidates []SearchResult, m int) []SearchResult {
	if len(candidates) <= m {
		return candidates
	}
	return candidates[:m]
}
