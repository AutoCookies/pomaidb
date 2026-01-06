package vector

import (
	"hash/fnv"
	"sync"
	"sync/atomic"
)

const (
	ChunkSize = 1024
)

type VectorArena struct {
	dim     int
	mu      sync.RWMutex
	chunks  [][]float32
	nodes   []*NodeData
	strToID map[uint64]uint32
	idToStr []string
	count   atomic.Uint32
}

type NodeData struct {
	Mu          sync.RWMutex
	Level       int
	Connections [][]uint32
	Deleted     bool
}

func NewVectorArena(dim int, initialCap int) *VectorArena {
	return &VectorArena{
		dim:     dim,
		chunks:  make([][]float32, 0, 16),
		nodes:   make([]*NodeData, 0, initialCap),
		strToID: make(map[uint64]uint32),
		idToStr: make([]string, 0, initialCap),
	}
}

func (arena *VectorArena) GetVector(id uint32) []float32 {
	chunkIdx := int(id) / ChunkSize
	offset := (int(id) % ChunkSize) * arena.dim

	arena.mu.RLock()
	chunk := arena.chunks[chunkIdx]
	arena.mu.RUnlock()

	return chunk[offset : offset+arena.dim]
}

func (arena *VectorArena) AllocNode(key string, vec []float32, level int) (uint32, bool) {
	arena.mu.Lock()
	defer arena.mu.Unlock()

	h := fnv.New64a()
	h.Write([]byte(key))
	keyHash := h.Sum64()

	if _, exists := arena.strToID[keyHash]; exists {
		return 0, false
	}

	id := uint32(len(arena.nodes))
	chunkIdx := int(id) / ChunkSize
	offset := (int(id) % ChunkSize) * arena.dim

	if chunkIdx >= len(arena.chunks) {
		newChunk := make([]float32, ChunkSize*arena.dim)
		arena.chunks = append(arena.chunks, newChunk)
	}

	copy(arena.chunks[chunkIdx][offset:], vec)

	arena.strToID[keyHash] = id
	arena.idToStr = append(arena.idToStr, key)

	node := &NodeData{
		Level:       level,
		Connections: make([][]uint32, level+1),
	}
	for i := 0; i <= level; i++ {
		node.Connections[i] = make([]uint32, 0, 16)
	}
	arena.nodes = append(arena.nodes, node)

	arena.count.Store(uint32(len(arena.nodes)))

	return id, true
}

func (arena *VectorArena) Size() int {
	return int(arena.count.Load())
}

func (arena *VectorArena) MarkDeleted(id uint32) {
	arena.mu.RLock()
	defer arena.mu.RUnlock()

	if int(id) < len(arena.nodes) {
		node := arena.nodes[id]
		node.Mu.Lock()
		node.Deleted = true
		node.Mu.Unlock()
	}
}

func (arena *VectorArena) IsDeleted(id uint32) bool {
	arena.mu.RLock()
	defer arena.mu.RUnlock()

	if int(id) >= len(arena.nodes) {
		return true
	}
	return arena.nodes[id].Deleted
}

func (arena *VectorArena) GetKey(id uint32) string {
	arena.mu.RLock()
	defer arena.mu.RUnlock()
	if int(id) < len(arena.idToStr) {
		return arena.idToStr[id]
	}
	return ""
}

func (arena *VectorArena) GetNode(id uint32) *NodeData {
	arena.mu.RLock()
	defer arena.mu.RUnlock()
	if int(id) < len(arena.nodes) {
		return arena.nodes[id]
	}
	return nil
}

func (arena *VectorArena) GetID(key string) (uint32, bool) {
	h := fnv.New64a()
	h.Write([]byte(key))
	keyHash := h.Sum64()

	arena.mu.RLock()
	defer arena.mu.RUnlock()
	id, ok := arena.strToID[keyHash]
	return id, ok
}

func (arena *VectorArena) GetVectors(ids []uint32) [][]float32 {
	arena.mu.RLock()
	defer arena.mu.RUnlock()

	vecs := make([][]float32, len(ids))
	for i, id := range ids {
		chunkIdx := int(id) / ChunkSize
		offset := (int(id) % ChunkSize) * arena.dim

		if chunkIdx < len(arena.chunks) {
			vecs[i] = arena.chunks[chunkIdx][offset : offset+arena.dim]
		}
	}
	return vecs
}
