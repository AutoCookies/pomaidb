package graph

import (
	"sync"
)

// CSRGraph: Compressed Sparse Row - Cấu trúc tĩnh tối ưu cho đọc và thuật toán
// Layout bộ nhớ liền mạch giúp CPU Prefetcher hoạt động tối đa hiệu suất.
type CSRGraph struct {
	Indptr   []int32   // Chỉ số bắt đầu các cạnh của node i
	Indices  []int32   // Danh sách các node kề
	Weights  []float64 // Trọng số tương ứng
	NumNodes int
}

// MutableGraph: Adjacency List - Cấu trúc động tối ưu cho ghi (Add Edge)
type MutableGraph struct {
	mu sync.RWMutex

	// Adjacency List: node_idx -> []neighbors
	Adj     [][]int32
	Weights [][]float64

	// Cache
	cachedCSR *CSRGraph
	dirty     bool
}

func NewMutableGraph() *MutableGraph {
	return &MutableGraph{
		Adj:     make([][]int32, 0),
		Weights: make([][]float64, 0),
	}
}

// EnsureCapacity: Đảm bảo mảng đủ lớn để chứa nodeIndex
func (mg *MutableGraph) EnsureCapacity(nodeIndex int32) {
	currentCap := int32(len(mg.Adj))
	if nodeIndex < currentCap {
		return
	}

	// Mở rộng mảng động
	newCap := nodeIndex + 1
	for i := currentCap; i < newCap; i++ {
		mg.Adj = append(mg.Adj, make([]int32, 0))
		mg.Weights = append(mg.Weights, make([]float64, 0))
	}
}

// AddEdge: Thêm cạnh (Thread-safe ở level Store, nhưng ở đây ta không lock để tối ưu raw speed nếu caller đã lock)
func (mg *MutableGraph) AddEdge(u, v int32, w float64) {
	// Caller (Store) chịu trách nhiệm đảm bảo u, v hợp lệ qua EnsureCapacity
	mg.Adj[u] = append(mg.Adj[u], v)
	mg.Weights[u] = append(mg.Weights[u], w)
	mg.dirty = true
}

// ToCSR: Chuyển đổi Adjacency List sang CSR (Snapshot)
func (mg *MutableGraph) ToCSR() *CSRGraph {
	if !mg.dirty && mg.cachedCSR != nil {
		return mg.cachedCSR
	}

	numNodes := len(mg.Adj)
	numEdges := 0
	for _, neighbors := range mg.Adj {
		numEdges += len(neighbors)
	}

	indptr := make([]int32, numNodes+1)
	indices := make([]int32, 0, numEdges)
	weights := make([]float64, 0, numEdges)

	offset := int32(0)
	for i, neighbors := range mg.Adj {
		indptr[i] = offset
		indices = append(indices, neighbors...)
		weights = append(weights, mg.Weights[i]...)
		offset += int32(len(neighbors))
	}
	indptr[numNodes] = offset

	csr := &CSRGraph{
		Indptr:   indptr,
		Indices:  indices,
		Weights:  weights,
		NumNodes: numNodes,
	}

	mg.cachedCSR = csr
	mg.dirty = false
	return csr
}
