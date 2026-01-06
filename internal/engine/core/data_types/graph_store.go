package data_types

import (
	"fmt"
	"hash/fnv"
	"sync"

	al "github.com/AutoCookies/pomai-cache/shared/al/graph"
	ds "github.com/AutoCookies/pomai-cache/shared/ds/graph"
)

const (
	// GraphShards: Số lượng phân vùng để giảm lock contention cho ID Mapping.
	// 64 là con số lý tưởng cho CPU 4-16 cores.
	GraphShards    = 64
	GraphShardMask = GraphShards - 1
)

// GraphIDShard: Quản lý mapping String -> Int32 cho một tập con các node.
// Tách biệt hoàn toàn với TypeRegistry của TimeStream để tránh xung đột.
type GraphIDShard struct {
	mu       sync.RWMutex
	strToInt map[string]int32
}

// GraphStoreWrapper: Wrapper bao đóng một đồ thị cụ thể.
// Nó quản lý việc chuyển đổi ID và đồng bộ hóa truy cập vào cấu trúc dữ liệu lõi.
type GraphStoreWrapper struct {
	// 1. ID Mapping Layer (Sharded) -> 10/10 Concurrency
	shards [GraphShards]*GraphIDShard

	// Global map ngược (Int -> String) - Chỉ append, ít khi bị lock contention
	intToStrMu sync.RWMutex
	intToStr   []string

	// 2. Data Storage Layer (Core DS)
	// Dùng lock riêng cho data để tách biệt với logic map ID
	dataMu sync.RWMutex
	data   *ds.MutableGraph
}

// NewGraphStoreWrapper khởi tạo wrapper với 64 shards
func NewGraphStoreWrapper() *GraphStoreWrapper {
	wrapper := &GraphStoreWrapper{
		data:     ds.NewMutableGraph(),
		intToStr: make([]string, 0, 1024), // Pre-alloc một chút để đỡ resize ban đầu
	}
	for i := 0; i < GraphShards; i++ {
		wrapper.shards[i] = &GraphIDShard{
			strToInt: make(map[string]int32),
		}
	}
	return wrapper
}

// getShard chọn shard dựa trên hash của nodeID
func (g *GraphStoreWrapper) getShard(nodeID string) *GraphIDShard {
	h := fnv.New32a()
	h.Write([]byte(nodeID))
	return g.shards[h.Sum32()&GraphShardMask]
}

// getOrCreateIndex: Chuyển đổi String ID -> Int32 ID với hiệu năng cực cao.
// Sử dụng Sharded Lock + Double-Checked Locking.
func (g *GraphStoreWrapper) getOrCreateIndex(nodeID string) int32 {
	shard := g.getShard(nodeID)

	// 1. Fast Path: Read Lock (99% cases)
	shard.mu.RLock()
	idx, ok := shard.strToInt[nodeID]
	shard.mu.RUnlock()
	if ok {
		return idx
	}

	// 2. Slow Path: Write Lock (Only for new nodes)
	shard.mu.Lock()
	// Double check để tránh race condition
	if idx, ok = shard.strToInt[nodeID]; ok {
		shard.mu.Unlock()
		return idx
	}

	// Tạo ID mới (Critical Section: Global Sequence)
	g.intToStrMu.Lock()
	newIdx := int32(len(g.intToStr))
	g.intToStr = append(g.intToStr, nodeID)
	g.intToStrMu.Unlock()

	// Lưu vào shard map
	shard.strToInt[nodeID] = newIdx
	shard.mu.Unlock()

	// Báo cho Core DS mở rộng bộ nhớ (nếu cần)
	// Lưu ý: Ta lock dataMu nhanh ở đây để đảm bảo an toàn cho mảng bên dưới
	g.dataMu.Lock()
	g.data.EnsureCapacity(newIdx)
	g.dataMu.Unlock()

	return newIdx
}

// -----------------------------------------------------------------------------
// GRAPH STORE (Manager)
// -----------------------------------------------------------------------------

type GraphStore struct {
	graphs sync.Map // map[string]*GraphStoreWrapper
}

func NewGraphStore() *GraphStore {
	return &GraphStore{}
}

func (gs *GraphStore) CreateGraph(name string) error {
	if _, exists := gs.graphs.Load(name); exists {
		return fmt.Errorf("graph exists")
	}
	gs.graphs.Store(name, NewGraphStoreWrapper())
	return nil
}

func (gs *GraphStore) getGraph(name string) (*GraphStoreWrapper, error) {
	val, ok := gs.graphs.Load(name)
	if !ok {
		// Lazy init: Tự tạo graph nếu chưa có (Tiện lợi cho dev/bench)
		wrapper := NewGraphStoreWrapper()
		actual, loaded := gs.graphs.LoadOrStore(name, wrapper)
		if loaded {
			return actual.(*GraphStoreWrapper), nil
		}
		return wrapper, nil
	}
	return val.(*GraphStoreWrapper), nil
}

// AddEdge: Thêm cạnh với hiệu năng cao nhất.
// Luồng xử lý:
// 1. Map ID song song (không chặn nhau nhờ Sharding).
// 2. Lock Core DS trong thời gian cực ngắn (chỉ để append vào slice).
func (gs *GraphStore) AddEdge(graphName, from, to string, weight float64) error {
	g, err := gs.getGraph(graphName)
	if err != nil {
		return err
	}

	// Step 1: Lấy ID (Fully Concurrent)
	u := g.getOrCreateIndex(from)
	v := g.getOrCreateIndex(to)

	// Step 2: Ghi vào cấu trúc dữ liệu (Serialized per graph but extremely fast)
	// Vì ds.MutableGraph dùng Slice of Slices, thao tác append chỉ tốn vài nanosecond.
	g.dataMu.Lock()
	g.data.AddEdge(u, v, weight)
	g.dataMu.Unlock()

	return nil
}

// AddNode: Chỉ đảm bảo Node ID tồn tại trong hệ thống
func (gs *GraphStore) AddNode(graphName, nodeID string) error {
	g, err := gs.getGraph(graphName)
	if err != nil {
		return err
	}
	g.getOrCreateIndex(nodeID)
	return nil
}

func (gs *GraphStore) NodeExists(graphName, nodeID string) bool {
	g, err := gs.getGraph(graphName)
	if err != nil {
		return false
	}
	shard := g.getShard(nodeID)
	shard.mu.RLock()
	defer shard.mu.RUnlock()
	_, ok := shard.strToInt[nodeID]
	return ok
}

// GetNeighbors: Lấy danh sách hàng xóm (1-hop)
func (gs *GraphStore) GetNeighbors(graphName, nodeID string) ([]string, error) {
	g, err := gs.getGraph(graphName)
	if err != nil {
		return nil, err
	}

	// 1. Lấy ID int32 của node
	shard := g.getShard(nodeID)
	shard.mu.RLock()
	uID, ok := shard.strToInt[nodeID]
	shard.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("node not found")
	}

	// 2. Đọc trực tiếp từ Core DS (Read Lock)
	// MutableGraph.Adj là mảng phẳng, truy cập rất nhanh
	var neighborInts []int32
	g.dataMu.RLock()
	// Cần kiểm tra biên vì uID có thể mới được tạo nhưng Adj chưa kịp grow (trường hợp hiếm race)
	// Nhưng getOrCreateIndex đã gọi EnsureCapacity nên an toàn.
	// Tuy nhiên ds.MutableGraph.Adj là slice, ta cần truy cập an toàn.
	// Giả sử ds.MutableGraph expose Adj (public field) như file graph.go trước đó.
	if int(uID) < len(g.data.Adj) {
		src := g.data.Adj[uID]
		// Copy ra ngoài để tránh race sau khi unlock
		neighborInts = make([]int32, len(src))
		copy(neighborInts, src)
	}
	g.dataMu.RUnlock()

	// 3. Map ngược lại String
	if len(neighborInts) == 0 {
		return []string{}, nil
	}

	g.intToStrMu.RLock()
	defer g.intToStrMu.RUnlock()

	result := make([]string, len(neighborInts))
	for i, idx := range neighborInts {
		if int(idx) < len(g.intToStr) {
			result[i] = g.intToStr[idx]
		}
	}

	return result, nil
}

// -----------------------------------------------------------------------------
// COMPUTE DELEGATION (Gọi xuống shared/al)
// -----------------------------------------------------------------------------

func (gs *GraphStore) PageRank(graphName string, iterations int) (map[string]float64, error) {
	g, err := gs.getGraph(graphName)
	if err != nil {
		return nil, err
	}

	// Snapshot dữ liệu sang CSR để tính toán (Read Lock)
	g.dataMu.Lock()
	csr := g.data.ToCSR() // ToCSR có thể thay đổi cachedCSR nên cần Lock
	g.dataMu.Unlock()

	// Chạy thuật toán (Pure Compute - No locks needed on GraphStore)
	rawScores := al.PageRank(csr, iterations, 0.85, 0.0001)

	// Map kết quả
	g.intToStrMu.RLock()
	defer g.intToStrMu.RUnlock()

	result := make(map[string]float64, len(rawScores))
	for i, score := range rawScores {
		if i < len(g.intToStr) {
			result[g.intToStr[i]] = score
		}
	}
	return result, nil
}

func (gs *GraphStore) ShortestPath(graphName, from, to string, maxDepth int) ([]string, error) {
	g, err := gs.getGraph(graphName)
	if err != nil {
		return nil, err
	}

	// Lookup IDs
	shardU := g.getShard(from)
	shardU.mu.RLock()
	uID, ok1 := shardU.strToInt[from]
	shardU.mu.RUnlock()

	shardV := g.getShard(to)
	shardV.mu.RLock()
	vID, ok2 := shardV.strToInt[to]
	shardV.mu.RUnlock()

	if !ok1 || !ok2 {
		return nil, fmt.Errorf("node not found")
	}

	// Snapshot CSR
	g.dataMu.Lock()
	csr := g.data.ToCSR()
	g.dataMu.Unlock()

	// Compute
	pathIndices := al.ShortestPathBFS(csr, uID, vID, maxDepth)
	if pathIndices == nil {
		return nil, fmt.Errorf("no path found")
	}

	// Map result
	g.intToStrMu.RLock()
	defer g.intToStrMu.RUnlock()

	pathStrs := make([]string, len(pathIndices))
	for i, idx := range pathIndices {
		pathStrs[i] = g.intToStr[idx]
	}

	return pathStrs, nil
}

func (gs *GraphStore) ListGraphs() []string {
	names := make([]string, 0)
	gs.graphs.Range(func(key, _ interface{}) bool {
		names = append(names, key.(string))
		return true
	})
	return names
}
