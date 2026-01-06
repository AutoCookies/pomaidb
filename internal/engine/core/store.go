package core

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"hash/fnv"
	"io"
	"log"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/snappy"
	"golang.org/x/sync/singleflight"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
	"github.com/AutoCookies/pomai-cache/internal/engine/core/data_types"
	"github.com/AutoCookies/pomai-cache/internal/engine/core/sharding"
	"github.com/AutoCookies/pomai-cache/internal/engine/core/storage"
	"github.com/AutoCookies/pomai-cache/internal/engine/eviction"
	"github.com/AutoCookies/pomai-cache/shared/ds/arena"
	"github.com/AutoCookies/pomai-cache/shared/ds/bloom"
	"github.com/AutoCookies/pomai-cache/shared/ds/sketch"
	"github.com/AutoCookies/pomai-cache/shared/ds/skiplist"
	"github.com/AutoCookies/pomai-cache/shared/ds/timestream"
)

var (
	ErrEmptyKey            = errors.New("empty key")
	ErrInsufficientStorage = errors.New("insufficient storage")
	ErrValueNotInteger     = errors.New("value is not an integer")
	ErrKeyNotFound         = errors.New("key not found")
	ErrCorruptData         = errors.New("corrupted chunk data")
)

var GlobalMemCtrl common.MemoryController

const (
	chunkHeaderMagic   = "\xff\xfeCHNK"
	chunkSizeThreshold = 1024
	fixedChunkSize     = 4096
	pgusMagicByte      = 0x02
)

var hashPool = sync.Pool{
	New: func() any {
		return fnv.New32a()
	},
}

type Store struct {
	config     *common.StoreConfig
	chunkStore *data_types.ChunkStore

	shards     []*sharding.Shard
	shardCount uint32
	shardMask  uint32

	totalBytesAtomic int64

	bloom *bloom.SBF
	g     singleflight.Group

	zsets map[string]*skiplist.Skiplist
	zmu   sync.RWMutex

	evictionManager *eviction.Manager
	evictionMetrics *eviction.EvictionMetrics
	evictionCtx     context.Context
	evictionCancel  context.CancelFunc
	freqSketch      *sketch.Sketch

	vectorStore *data_types.VectorStore
	graphStore  *data_types.GraphStore
	timeStream  *data_types.TimeStreamStore
	bitmapStore *data_types.BitmapStore
	cdcEnabled  atomic.Bool
	cdcStream   string
	hashStore   *data_types.HashStore

	plgStore *data_types.PLGStore
	pgus     *storage.PGUS

	vectorSearchEMA    float64
	vectorSearchEMAMu  sync.Mutex
	vectorSearchAlpha  float64
	vectorSearchTarget time.Duration
	adaptiveEfEnabled  bool
	picStore           *data_types.PICStore
	pmcStore           *data_types.PMCStore

	_      [56]byte // CPU Caching Padding make the hot fields to be on separate cache lines
	hits   atomic.Uint64
	_      [56]byte
	misses atomic.Uint64
	_      [56]byte
	arena  *arena.LuuArena
}

func NewStore(shardCount int) *Store {
	config := common.DefaultStoreConfig()
	config.ShardCount = shardCount
	return NewStoreWithConfig(config)
}

func NewStoreWithOptions(shardCount int, capacityBytes int64) *Store {
	config := common.DefaultStoreConfig()
	config.ShardCount = shardCount
	config.CapacityBytes = capacityBytes
	return NewStoreWithConfig(config)
}

func NewStoreWithConfig(config *common.StoreConfig) *Store {
	if config.ShardCount <= 0 {
		config.ShardCount = 256
	}

	shardCount := nextPowerOf2(config.ShardCount)
	config.ShardCount = shardCount

	ctx, cancel := context.WithCancel(context.Background())
	dataDir := "./data"

	s := &Store{
		config:          config,
		shards:          make([]*sharding.Shard, shardCount),
		shardCount:      uint32(shardCount),
		shardMask:       uint32(shardCount - 1),
		chunkStore:      data_types.NewChunkStore(),
		zsets:           make(map[string]*skiplist.Skiplist),
		evictionMetrics: &eviction.EvictionMetrics{},
		evictionCtx:     ctx,
		evictionCancel:  cancel,
		vectorStore:     data_types.NewVectorStore(),
		graphStore:      data_types.NewGraphStore(),
		timeStream:      data_types.NewTimeStreamStore(),
		bitmapStore:     data_types.NewBitmapStore(),
		cdcStream:       "__cdc_log__",
		hashStore:       data_types.NewHashStore(),
		picStore:        data_types.NewPICStore(),
		pmcStore:        data_types.NewPMCStore(),
		plgStore:        data_types.NewPLGStore(),
		pgus:            storage.NewPGUS(dataDir + "/pomai_virtual.dat"),
		arena:           arena.NewLuuArena(),
	}

	s.cdcEnabled.Store(false)
	s.freqSketch = sketch.New(1<<16, 4)

	for i := 0; i < shardCount; i++ {
		s.shards[i] = sharding.NewLockFreeShardAdapter()
	}

	s.evictionManager = eviction.NewManager(s)

	s.vectorSearchEMA = 0
	s.vectorSearchAlpha = 0.12
	s.vectorSearchTarget = 50 * time.Millisecond
	s.adaptiveEfEnabled = true

	return s
}

func (s *Store) PLGAddEdge(node1, node2 string, weight float64) {
	if s.evictionManager != nil {
		now := time.Now().UnixNano()
		s.evictionManager.RecordAccess(node1, now)
		s.evictionManager.RecordAccess(node2, now)
	}
	s.plgStore.AddEdge(node1, node2, weight)
}

func (s *Store) PLGExtractCluster(startNode string, minDensity float64) []string {
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(startNode, time.Now().UnixNano())
	}
	return s.plgStore.ExtractCluster(startNode, minDensity)
}

func (s *Store) SetBit(key string, offset uint64, value int) (int, error) {
	return s.bitmapStore.SetBit(key, offset, value)
}

func (s *Store) GetBit(key string, offset uint64) (int, error) {
	return s.bitmapStore.GetBit(key, offset)
}

func (s *Store) BitCount(key string, start, end int64) (int64, error) {
	return s.bitmapStore.BitCount(key, start, end)
}

func (s *Store) StreamAppend(stream string, id string, val float64, metadata map[string]interface{}) error {
	return s.timeStream.Append(stream, &timestream.Event{
		ID:        id,
		Value:     val,
		Metadata:  metadata,
		Timestamp: time.Now().UnixNano(),
		Type:      "generic",
	})
}

func (s *Store) StreamAppendBatch(stream string, events []*timestream.Event) error {
	return s.timeStream.AppendBatch(stream, events)
}

func (s *Store) StreamRange(stream string, start, end int64) ([]*timestream.Event, error) {
	return s.timeStream.Range(stream, start, end, nil)
}

func (s *Store) StreamWindow(stream string, windowStr string, aggType string) (map[int64]float64, error) {
	dur, err := time.ParseDuration(windowStr)
	if err != nil {
		return nil, err
	}
	return s.timeStream.Window(stream, dur, aggType)
}

func (s *Store) StreamDetectAnomaly(stream string, threshold float64) ([]*timestream.Event, error) {
	if threshold <= 0 {
		threshold = 2.5
	}
	return s.timeStream.DetectAnomaly(stream, threshold)
}

func (s *Store) StreamForecast(stream string, horizonStr string) (float64, error) {
	dur, err := time.ParseDuration(horizonStr)
	if err != nil {
		return 0, err
	}
	return s.timeStream.Forecast(stream, dur)
}

func (s *Store) StreamDetectPattern(stream string, types []string, withinStr string) ([][]*timestream.Event, error) {
	dur, err := time.ParseDuration(withinStr)
	if err != nil {
		return nil, err
	}
	return s.timeStream.DetectPattern(stream, types, dur)
}

func (s *Store) ReadGroup(stream, group string, count int) ([]*timestream.Event, error) {
	return s.timeStream.ReadGroup(stream, group, count)
}

func graphNodeKey(graphName, nodeID string) string {
	return "g:n:" + graphName + ":" + nodeID
}

func graphEdgeKey(graphName, from, to string) string {
	return "g:e:" + graphName + ":" + from + ":" + to
}

func (s *Store) CreateGraph(name string) error {
	return s.graphStore.CreateGraph(name)
}

func (s *Store) AddGraphNode(graphName, nodeID string, properties map[string]interface{}) error {
	// 1. Đăng ký ID vào Graph Engine (Để sau này map được String <-> Int)
	// Hàm này trong graph_store cần expose ra, hoặc ta gọi AddEdge ảo.
	// Tuy nhiên, cách tốt nhất là cập nhật graph_store để có hàm EnsureNode.
	// Tạm thời ta dùng trick: AddEdge(nodeID, nodeID, 0) nếu graph_store chưa hỗ trợ AddNode riêng,
	// nhưng tốt nhất hãy thêm hàm AddNode vào graph_store (xem bên dưới).
	if err := s.graphStore.AddNode(graphName, nodeID); err != nil {
		return err
	}

	// 2. Lưu Properties vào KV Store (Cache dữ liệu)
	if len(properties) > 0 {
		propBytes, err := json.Marshal(properties)
		if err != nil {
			return err
		}
		// Lưu key dạng: g:n:mygraph:user1
		return s.Put(graphNodeKey(graphName, nodeID), propBytes, 0)
	}
	return nil
}

func (s *Store) AddGraphEdge(graphName, from, to string, weight float64, props map[string]interface{}) error {
	if s.evictionManager != nil {
		now := time.Now().UnixNano()
		s.evictionManager.RecordAccess(from, now)
		s.evictionManager.RecordAccess(to, now)
	}

	if err := s.graphStore.AddEdge(graphName, from, to, weight); err != nil {
		return err
	}

	if len(props) > 0 {
		propBytes, err := json.Marshal(props)
		if err != nil {
			return err
		}
		return s.Put(graphEdgeKey(graphName, from, to), propBytes, 0)
	}
	return nil
}

func (s *Store) GraphGetNode(graphName, nodeID string) (map[string]interface{}, error) {
	val, ok := s.Get(graphNodeKey(graphName, nodeID))
	if !ok {
		// Node có thể tồn tại trong Topology nhưng không có props
		// Check Topology
		if !s.graphStore.NodeExists(graphName, nodeID) {
			return nil, ErrKeyNotFound
		}
		return map[string]interface{}{"id": nodeID}, nil
	}

	var props map[string]interface{}
	if err := json.Unmarshal(val, &props); err != nil {
		return nil, err
	}
	props["id"] = nodeID
	return props, nil
}

func (s *Store) GraphNeighbors(graphName, nodeID string, depth int) ([]map[string]interface{}, error) {
	// 1. Dùng Graph Engine để lấy IDs (Siêu nhanh)
	// Cần thêm hàm GetNeighbors vào graph_store (dựa trên BFS hoặc CSR Slice)
	neighborIDs, err := s.graphStore.GetNeighbors(graphName, nodeID)
	if err != nil {
		return nil, err
	}

	// 2. Hydrate dữ liệu (Lấy thịt đắp vào xương)
	results := make([]map[string]interface{}, 0, len(neighborIDs))
	for _, id := range neighborIDs {
		// Pipeline Get (nếu tối ưu kỹ hơn thì dùng MGet)
		nodeProps, _ := s.GraphGetNode(graphName, id)
		if nodeProps == nil {
			nodeProps = map[string]interface{}{"id": id}
		}
		results = append(results, nodeProps)
	}
	return results, nil
}

func (s *Store) GraphShortestPath(graphName, from, to string, maxDepth int) ([]string, error) {
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(from, time.Now().UnixNano())
	}
	return s.graphStore.ShortestPath(graphName, from, to, maxDepth)
}

func (s *Store) GraphPageRank(graphName string, iterations int) (map[string]float64, error) {
	return s.graphStore.PageRank(graphName, iterations)
}

func (s *Store) ListGraphs() []string {
	return s.graphStore.ListGraphs()
}

func (s *Store) PutWithEmbedding(key string, value []byte, embedding []float32, ttl time.Duration) error {
	if key == "" {
		return ErrEmptyKey
	}

	var norm float64
	for _, v := range embedding {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range embedding {
			embedding[i] = float32(float64(embedding[i]) / norm)
		}
	}

	if err := s.Put(key, value, ttl); err != nil {
		return err
	}

	defaultIndex := "default_embeddings"
	if !s.vectorStore.HasIndex(defaultIndex) {
		if err := s.vectorStore.CreateIndex(defaultIndex, len(embedding), "cosine"); err != nil {
			return fmt.Errorf("create vector index failed: %w", err)
		}
	}

	metadata := map[string]interface{}{
		"key":        key,
		"created_at": time.Now().Unix(),
	}

	if err := s.vectorStore.Insert(defaultIndex, key, embedding, metadata); err != nil {
		return fmt.Errorf("vector insert failed: %w", err)
	}

	return nil
}

func (s *Store) SearchVector(indexName string, query []float32, k int) ([]data_types.VectorSearchResult, error) {
	if s.vectorStore == nil {
		return nil, fmt.Errorf("vector store not initialized")
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("empty query")
	}

	const maxAttempts = 3
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		start := time.Now()
		results, err := s.vectorStore.Search(indexName, query, k)
		elapsed := time.Since(start)

		s.updateVectorSearchEMA(elapsed)

		if err == nil {
			if s.adaptiveEfEnabled {
				s.maybeReduceEfOnLatency()
			}
			return results, nil
		}

		lastErr = err
		time.Sleep(time.Duration(attempt*10) * time.Millisecond)
	}

	return nil, fmt.Errorf("vector search failed after retries: %w", lastErr)
}

func (s *Store) SetAdaptiveEfEnabled(enabled bool) {
	s.adaptiveEfEnabled = enabled
}

func (s *Store) updateVectorSearchEMA(d time.Duration) {
	s.vectorSearchEMAMu.Lock()
	defer s.vectorSearchEMAMu.Unlock()
	ms := float64(d.Milliseconds())
	if s.vectorSearchEMA == 0 {
		s.vectorSearchEMA = ms
	} else {
		a := s.vectorSearchAlpha
		s.vectorSearchEMA = a*ms + (1.0-a)*s.vectorSearchEMA
	}
}

func (s *Store) maybeReduceEfOnLatency() {
	s.vectorSearchEMAMu.Lock()
	emaMs := s.vectorSearchEMA
	s.vectorSearchEMAMu.Unlock()

	if emaMs == 0 {
		return
	}

	if time.Duration(emaMs)*time.Millisecond > s.vectorSearchTarget {
		stats, err := s.vectorStore.IndexStats("default_embeddings")
		if err != nil {
			return
		}
		currentEf := parseEfFromStats(stats)
		if currentEf <= 1 {
			return
		}
		newEf := int(math.Max(1, math.Floor(float64(currentEf)*0.8)))
		s.vectorStore.SetGlobalEfSearch(newEf)
		log.Printf("[TUNER] Reduced ef from %d to %d due to high vector EMA=%.2fms", currentEf, newEf, emaMs)
	}
}

func parseEfFromStats(stats map[string]interface{}) int {
	if stats == nil {
		return 0
	}
	keys := []string{"ef", "ef_search", "global_ef", "efSearch"}
	for _, k := range keys {
		if v, ok := stats[k]; ok {
			switch t := v.(type) {
			case int:
				return t
			case int32:
				return int(t)
			case int64:
				return int(t)
			case float64:
				return int(t)
			case string:
				if parsed, err := strconv.Atoi(t); err == nil {
					return parsed
				}
			}
		}
	}
	return 0
}

func (s *Store) PutWithEmbeddingFast(key string, value []byte, embedding []float32, ttl time.Duration) error {
	if key == "" {
		return ErrEmptyKey
	}

	var norm float64
	for _, v := range embedding {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range embedding {
			embedding[i] = float32(float64(embedding[i]) / norm)
		}
	}

	if err := s.PutFast(key, value, ttl); err != nil {
		return err
	}

	defaultIndex := "default_embeddings"
	if !s.vectorStore.HasIndex(defaultIndex) {
		if err := s.vectorStore.CreateIndex(defaultIndex, len(embedding), "cosine"); err != nil {
			return fmt.Errorf("create vector index failed: %w", err)
		}
	}

	metadata := map[string]interface{}{
		"key":        key,
		"created_at": time.Now().Unix(),
	}

	if err := s.vectorStore.Insert(defaultIndex, key, embedding, metadata); err != nil {
		return fmt.Errorf("vector insert failed: %w", err)
	}

	return nil
}

func (s *Store) DeleteVector(indexName, id string) error {
	return s.vectorStore.Delete(indexName, id)
}

func (s *Store) DropVectorIndex(name string) error {
	return s.vectorStore.DropIndex(name)
}

func (s *Store) ListVectorIndices() []string {
	return s.vectorStore.ListIndices()
}

func (s *Store) VectorIndexStats(name string) (map[string]interface{}, error) {
	return s.vectorStore.IndexStats(name)
}

func (s *Store) SetGlobalEfSearch(ef int) {
	if s.vectorStore != nil {
		s.vectorStore.SetGlobalEfSearch(ef)
	}
}

type SemanticResult struct {
	Key      string
	Value    []byte
	Distance float32
	Metadata map[string]interface{}
}

func (s *Store) Shutdown() {
	if s.evictionCancel != nil {
		s.evictionCancel()
	}
}

func (s *Store) getShardFast(key string) *sharding.Shard {
	h := hashPool.Get().(hash.Hash32)
	h.Reset()
	h.Write([]byte(key))
	idx := h.Sum32() & s.shardMask
	hashPool.Put(h)
	return s.shards[idx]
}

func (s *Store) getShard(key string) *sharding.Shard {
	return s.getShardFast(key)
}

func (s *Store) hashToShardIndex(key string) int {
	h := hashPool.Get().(hash.Hash32)
	h.Reset()
	h.Write([]byte(key))
	idx := int(h.Sum32() & s.shardMask)
	hashPool.Put(h)
	return idx
}

func (s *Store) GetShards() []*sharding.Shard {
	return s.shards
}

func (s *Store) isChunked(val []byte) bool {
	return len(val) >= len(chunkHeaderMagic) && string(val[:len(chunkHeaderMagic)]) == chunkHeaderMagic
}

func (s *Store) resolveChunks(val []byte) ([]byte, bool) {
	if len(val) < len(chunkHeaderMagic)+8 {
		return nil, false
	}

	reader := bytes.NewReader(val[len(chunkHeaderMagic):])

	var origSize uint32
	var numChunks uint32

	if err := binary.Read(reader, binary.LittleEndian, &origSize); err != nil {
		return nil, false
	}
	if err := binary.Read(reader, binary.LittleEndian, &numChunks); err != nil {
		return nil, false
	}

	out := make([]byte, 0, origSize)
	chunkIDBuf := make([]byte, 8)

	for i := uint32(0); i < numChunks; i++ {
		if _, err := io.ReadFull(reader, chunkIDBuf); err != nil {
			return nil, false
		}
		id := data_types.ChunkID(string(chunkIDBuf))
		chunkData, ok := s.chunkStore.GetChunk(id)
		if !ok {
			return nil, false
		}
		out = append(out, chunkData...)
	}

	if uint32(len(out)) != origSize {
		return nil, false
	}

	return out, true
}

func (s *Store) cleanupChunks(val []byte) {
	if len(val) < len(chunkHeaderMagic)+8 {
		return
	}

	reader := bytes.NewReader(val[len(chunkHeaderMagic)+4:])
	var numChunks uint32
	if err := binary.Read(reader, binary.LittleEndian, &numChunks); err != nil {
		return
	}

	ids := make([]data_types.ChunkID, 0, numChunks)
	chunkIDBuf := make([]byte, 8)

	for i := uint32(0); i < numChunks; i++ {
		if _, err := io.ReadFull(reader, chunkIDBuf); err != nil {
			break
		}
		ids = append(ids, data_types.ChunkID(string(chunkIDBuf)))
	}

	go s.chunkStore.DecRefBatch(ids)
}

func (s *Store) putChunked(key string, value []byte, ttl time.Duration) error {
	numChunks := (len(value) + fixedChunkSize - 1) / fixedChunkSize
	chunks := make([][]byte, 0, numChunks)

	for i := 0; i < len(value); i += fixedChunkSize {
		end := i + fixedChunkSize
		if end > len(value) {
			end = len(value)
		}
		chunks = append(chunks, value[i:end])
	}

	ids, err := s.chunkStore.PutBatch(chunks)
	if err != nil {
		return err
	}

	buf := bytes.NewBuffer(make([]byte, 0, len(chunkHeaderMagic)+8+(len(ids)*8)))
	buf.WriteString(chunkHeaderMagic)
	binary.Write(buf, binary.LittleEndian, uint32(len(value)))
	binary.Write(buf, binary.LittleEndian, uint32(len(ids)))

	for _, id := range ids {
		buf.WriteString(string(id))
	}

	entry := common.NewEntry(key, buf.Bytes(), ttl)
	shard := s.getShardFast(key)
	_, deltaBytes := shard.Set(entry)
	atomic.AddInt64(&s.totalBytesAtomic, deltaBytes)

	s.trackCDC("PUT", key, value)
	if s.bloom != nil {
		s.bloom.Add(key)
	}
	if s.freqSketch != nil {
		s.freqSketch.Increment(key)
	}

	return nil
}

func (s *Store) Get(key string) ([]byte, bool) {
	if key == "" {
		return nil, false
	}
	if s.bloom != nil && !s.bloom.Exists(key) {
		s.misses.Add(1)
		return nil, false
	}

	shard := s.getShardFast(key)
	entry, ok := shard.Get(key)
	if !ok {
		s.misses.Add(1) // Miss
		return nil, false
	}
	s.hits.Add(1) // Hit
	s.recordAccess(key)

	// ... logic xử lý chunk/pgus giữ nguyên ...
	val := entry.Value()
	if len(val) > 0 && val[0] == pgusMagicByte {
		return s.resolvePGUS(val)
	}
	if s.isChunked(val) {
		return s.resolveChunks(val)
	}
	return val, true
}

func (s *Store) GetFast(key string) ([]byte, bool) {
	if key == "" {
		return nil, false
	}

	if s.bloom != nil && !s.bloom.Exists(key) {
		return nil, false
	}

	shard := s.getShardFast(key)
	entry, ok := shard.Get(key)
	if !ok {
		s.misses.Add(1)
		return nil, false
	}

	s.hits.Add(1)

	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(key, time.Now().UnixNano())
	}

	val := entry.Value()
	if s.isChunked(val) {
		return s.resolveChunks(val)
	}

	return val, true
}

func (s *Store) Put(key string, value []byte, ttl time.Duration) error {
	if key == "" {
		return ErrEmptyKey
	}

	if len(value) >= chunkSizeThreshold {
		return s.putPGUS(key, value, ttl)
	}

	if s.config.CapacityBytes > 0 &&
		atomic.LoadInt64(&s.totalBytesAtomic) >= s.config.CapacityBytes {
		return ErrInsufficientStorage
	}

	// Lưu Inline (Raw)
	entry := common.NewEntry(key, value, ttl)
	shard := s.getShardFast(key)
	_, deltaBytes := shard.Set(entry)
	atomic.AddInt64(&s.totalBytesAtomic, deltaBytes)

	s.postWriteOps(key, value)

	return nil
}

func (s *Store) putPGUS(key string, value []byte, ttl time.Duration) error {

	granuleIDs := s.pgus.Write(value)

	metaLen := 1 + 4 + (len(granuleIDs) * 8)
	metaBuf := make([]byte, metaLen)

	metaBuf[0] = pgusMagicByte
	binary.BigEndian.PutUint32(metaBuf[1:5], uint32(len(granuleIDs)))

	for i, id := range granuleIDs {
		offset := 5 + (i * 8)
		binary.BigEndian.PutUint64(metaBuf[offset:], uint64(id))
	}

	entry := common.NewEntry(key, metaBuf, ttl)
	shard := s.getShardFast(key)
	_, deltaBytes := shard.Set(entry)
	atomic.AddInt64(&s.totalBytesAtomic, deltaBytes)

	s.postWriteOps(key, value)
	return nil
}

func (s *Store) PutFast(key string, value []byte, ttl time.Duration) error {
	if key == "" {
		return ErrEmptyKey
	}

	if len(value) >= chunkSizeThreshold {
		return s.putChunked(key, value, ttl)
	}

	if len(value) > 4096 {
		if ptr, err := s.offloadPGUS(value); err == nil {
			value = ptr
		}
	}

	destBuf, slab := s.arena.Alloc(len(value))
	copy(destBuf, value)

	entry := common.NewEntry(key, destBuf, ttl)
	if slab != nil {
		entry.SetSlab(slab)
	}

	shard := s.getShardFast(key)

	if oldEntry, ok := shard.Get(key); ok {
		var oldSlabPtr *arena.Slab
		if ref := oldEntry.GetSlab(); ref != nil {
			if casted, ok := ref.(*arena.Slab); ok {
				oldSlabPtr = casted
			}
		}
		s.arena.Free(oldEntry.Value(), oldSlabPtr)
	}

	_, deltaBytes := shard.Set(entry)
	atomic.AddInt64(&s.totalBytesAtomic, deltaBytes)

	if s.bloom != nil {
		s.bloom.Add(key)
	}
	if s.freqSketch != nil {
		s.freqSketch.Increment(key)
	}

	s.postWriteOps(key, value)

	return nil
}

func (s *Store) Delete(key string) bool {
	shard := s.getShardFast(key)

	entry, ok := shard.Get(key)
	if ok {
		shard.Delete(key)
		atomic.AddInt64(&s.totalBytesAtomic, -int64(entry.Size()))

		var slabPtr *arena.Slab
		if ref := entry.GetSlab(); ref != nil {
			if casted, ok := ref.(*arena.Slab); ok {
				slabPtr = casted
			}
		}

		s.arena.Free(entry.Value(), slabPtr)

		if len(entry.Value()) >= 5 && string(entry.Value()[:1]) == "P" {
			s.cleanupPGUS(entry.Value())
		}
		return true
	}
	return false
}
func (s *Store) Exists(key string) bool {
	if key == "" {
		return false
	}

	if s.bloom != nil && !s.bloom.Exists(key) {
		return false
	}

	shard := s.getShardFast(key)
	_, ok := shard.Get(key)
	return ok
}

func (s *Store) Incr(key string, delta int64) (int64, error) {
	if key == "" {
		return 0, common.ErrEmptyKey
	}

	shard := s.getShardFast(key)
	var resultVal int64

	mutator := func(oldEntry *common.Entry) (*common.Entry, error) {
		var currentVal int64 = 0

		if oldEntry != nil {
			raw := oldEntry.Value()
			var valStr string

			if s.isChunked(raw) {
				decoded, ok := s.resolveChunks(raw)
				if !ok {
					return nil, common.ErrCorruptData
				}
				valStr = string(decoded)
			} else if len(raw) > 0 {
				if raw[0] == 1 {
					decoded, err := snappy.Decode(nil, raw[1:])
					if err != nil {
						return nil, fmt.Errorf("corrupted data: %w", err)
					}
					valStr = string(decoded)
				} else {
					valStr = string(raw[1:])
				}
			}

			val, err := strconv.ParseInt(valStr, 10, 64)
			if err != nil {
				return nil, common.ErrValueNotInteger
			}
			currentVal = val

			if s.isChunked(raw) {
				s.cleanupChunks(raw)
			}
		}

		newVal := currentVal + delta
		resultVal = newVal

		newValBytes := []byte(strconv.FormatInt(newVal, 10))
		finalData := make([]byte, len(newValBytes)+1)
		finalData[0] = 0
		copy(finalData[1:], newValBytes)

		return common.NewEntry(key, finalData, 0), nil
	}

	err := shard.AtomicMutate(key, mutator)
	if err != nil {
		return 0, err
	}

	return resultVal, nil
}

func (s *Store) MSet(items map[string][]byte, ttl time.Duration) error {
	if len(items) == 0 {
		return nil
	}
	for key, val := range items {
		if err := s.Put(key, val, ttl); err != nil {
			return fmt.Errorf("mset failed at key %s: %w", key, err)
		}
	}
	return nil
}

func (s *Store) Clear() {
	for _, shard := range s.shards {
		shard.Clear()
	}

	if s.hashStore != nil {
		s.hashStore.Clear()
	}

	if s.plgStore != nil {
		s.plgStore.Clear()
	}

	atomic.StoreInt64(&s.totalBytesAtomic, 0)
}

func (s *Store) SetTenantID(tenantID string) {
	if tenantID == "" {
		tenantID = "default"
	}
	s.config.TenantID = tenantID
}

func (s *Store) GetConfig() *common.StoreConfig {
	return s.config
}

func (s *Store) GetShard(key string) eviction.ShardInterface {
	return s.getShardFast(key)
}

func (s *Store) GetShardByIndex(idx int) eviction.ShardInterface {
	if idx < 0 || idx >= len(s.shards) {
		return nil
	}
	return s.shards[idx]
}

func (s *Store) GetShardCount() int {
	return int(s.shardCount)
}

func (s *Store) GetCapacityBytes() int64 {
	return s.config.CapacityBytes
}

func (s *Store) GetTotalBytes() int64 {
	return atomic.LoadInt64(&s.totalBytesAtomic)
}

func (s *Store) AddTotalBytes(delta int64) {
	atomic.AddInt64(&s.totalBytesAtomic, delta)
}

func (s *Store) GetTenantID() string {
	return s.config.TenantID
}

func (s *Store) GetFreqEstimator() eviction.FreqEstimator {
	if s.freqSketch == nil {
		return nil
	}
	return &freqEstimatorWrapper{sketch: s.freqSketch}
}

type freqEstimatorWrapper struct {
	sketch *sketch.Sketch
}

func (w *freqEstimatorWrapper) Estimate(key string) uint32 {
	if w.sketch == nil {
		return 0
	}
	return w.sketch.Estimate(key)
}

func (w *freqEstimatorWrapper) Increment(key string) {
	if w.sketch != nil {
		w.sketch.Increment(key)
	}
}

func (s *Store) GetGlobalMemCtrl() eviction.MemoryController {
	if GlobalMemCtrl == nil {
		return nil
	}
	return &memoryControllerWrapper{mc: GlobalMemCtrl}
}

func (s *Store) AddEviction() {}

func (s *Store) ForceEvictBytes(targetBytes int64) int64 {
	if s.evictionManager == nil {
		return 0
	}
	return s.evictionManager.BatchEvict(targetBytes)
}

type memoryControllerWrapper struct {
	mc common.MemoryController
}

func (w *memoryControllerWrapper) Release(bytes int64) {
	if w.mc != nil {
		w.mc.Release(bytes)
	}
}

func (w *memoryControllerWrapper) Reserve(bytes int64) bool {
	if w.mc != nil {
		return w.mc.Reserve(bytes)
	}
	return true
}

func (w *memoryControllerWrapper) Used() int64 {
	if w.mc != nil {
		return w.mc.Used()
	}
	return 0
}

func (w *memoryControllerWrapper) Capacity() int64 {
	if w.mc != nil {
		return w.mc.Capacity()
	}
	return 0
}

func (s *Store) EvictionStats() common.EvictionMetrics {
	return common.EvictionMetrics{}
}

func (s *Store) Stats() common.Stats {
	var totalItems int64
	var totalBytes int64

	for _, shard := range s.shards {
		totalItems += int64(shard.Len())
		totalBytes += shard.Bytes()
	}

	return common.Stats{
		Items:      totalItems,
		Bytes:      totalBytes,
		Capacity:   s.config.CapacityBytes,
		ShardCount: int(s.shardCount),
		TenantID:   s.config.TenantID,
	}
}

func (s *Store) GetHits() uint64 {
	return s.hits.Load()
}

func (s *Store) GetMisses() uint64 {
	return s.misses.Load()
}

func (s *Store) GetAvgLatency() float64 {
	s.vectorSearchEMAMu.Lock()
	defer s.vectorSearchEMAMu.Unlock()
	return s.vectorSearchEMA
}

func (s *Store) GetEvictions() uint64 {
	return 0
}

func (s *Store) ResetStats() {}

func (s *Store) Serialize() (io.Reader, error) {
	allEntries := make(map[string][]byte)

	for _, shard := range s.shards {
		items := shard.GetItems()
		for key, val := range items {
			if elem, ok := val.(*common.Entry); ok {
				if !elem.IsExpired() {
					valBytes := elem.Value()
					if s.isChunked(valBytes) {
						if resolved, ok := s.resolveChunks(valBytes); ok {
							allEntries[key] = resolved
						}
					} else {
						allEntries[key] = valBytes
					}
				}
			}
		}
	}

	data, err := json.Marshal(allEntries)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize: %w", err)
	}

	return bytes.NewReader(data), nil
}

func (s *Store) RestoreFrom(r io.Reader) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("failed to read restore data: %w", err)
	}

	if len(data) == 0 {
		return nil
	}

	var entries map[string][]byte
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("failed to deserialize store data: %w", err)
	}

	restoredCount := 0
	for key, value := range entries {
		if err := s.Put(key, value, 0); err != nil {
			log.Printf("Failed to restore key %s: %v", key, err)
		} else {
			restoredCount++
		}
	}

	log.Printf("Restored %d/%d entries to tenant '%s'",
		restoredCount, len(entries), s.config.TenantID)

	return nil
}

func (s *Store) SnapshotTo(w io.Writer) error {
	reader, err := s.Serialize()
	if err != nil {
		return err
	}
	_, err = io.Copy(w, reader)
	return err
}

func nextPowerOf2(n int) int {
	if n <= 1 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++
	return n
}

func (s *Store) EvictExpired() int {
	totalEvicted := 0
	for i := 0; i < int(s.shardCount); i++ {
		shard := s.shards[i]
		expired := shard.EvictExpired()
		for _, e := range expired {
			if s.isChunked(e.Value()) {
				s.cleanupChunks(e.Value())
			}
		}
		totalEvicted += len(expired)
	}
	return totalEvicted
}

func (s *Store) EnableCDC(enabled bool) {
	s.cdcEnabled.Store(enabled)
}

func (s *Store) trackCDC(op string, key string, val []byte) {
	if !s.cdcEnabled.Load() {
		return
	}

	numVal := 1.0
	if op == "DEL" {
		numVal = 0.0
	}

	meta := map[string]interface{}{
		"op":  op,
		"key": key,
	}

	if len(val) > 0 && len(val) < 1024 {
		meta["val"] = string(val)
	}

	go s.timeStream.Append(s.cdcStream, &timestream.Event{
		Type:      "cdc",
		Value:     numVal,
		Metadata:  meta,
		Timestamp: time.Now().UnixNano(),
	})
}

func (s *Store) GetChanges(groupName string, count int) ([]*timestream.Event, error) {
	return s.timeStream.ReadGroup(s.cdcStream, groupName, count)
}

func (s *Store) getOrCreateZSet(key string) *skiplist.Skiplist {
	s.zmu.RLock()
	zs, ok := s.zsets[key]
	s.zmu.RUnlock()

	if ok {
		return zs
	}

	s.zmu.Lock()
	defer s.zmu.Unlock()

	if zs, ok = s.zsets[key]; ok {
		return zs
	}

	zs = skiplist.New()
	s.zsets[key] = zs
	return zs
}

func (s *Store) ZAdd(key string, score float64, member string) error {
	zs := s.getOrCreateZSet(key)
	zs.Insert(member, score)
	return nil
}

func (s *Store) ZRem(key string, member string) bool {
	s.zmu.RLock()
	zs, ok := s.zsets[key]
	s.zmu.RUnlock()

	if !ok {
		return false
	}
	return zs.Delete(member)
}

func (s *Store) ZScore(key string, member string) (float64, bool) {
	s.zmu.RLock()
	zs, ok := s.zsets[key]
	s.zmu.RUnlock()

	if !ok {
		return 0, false
	}
	return zs.GetScore(member)
}

func (s *Store) ZRank(key string, member string) int {
	s.zmu.RLock()
	zs, ok := s.zsets[key]
	s.zmu.RUnlock()

	if !ok {
		return -1
	}
	return zs.GetRank(member)
}

func (s *Store) ZRange(key string, start, stop int) []skiplist.Element {
	s.zmu.RLock()
	zs, ok := s.zsets[key]
	s.zmu.RUnlock()

	if !ok {
		return nil
	}
	return zs.GetRange(start, stop)
}

func (s *Store) ZCard(key string) int {
	s.zmu.RLock()
	zs, ok := s.zsets[key]
	s.zmu.RUnlock()

	if !ok {
		return 0
	}
	return zs.Card()
}

func (s *Store) HSet(key, field string, value []byte) error {
	return s.hashStore.HSet(key, field, value)
}

func (s *Store) HGet(key, field string) ([]byte, bool) {
	return s.hashStore.HGet(key, field)
}

func (s *Store) HDel(key, field string) bool {
	return s.hashStore.HDel(key, field)
}

func (s *Store) HExists(key, field string) bool {
	return s.hashStore.HExists(key, field)
}

func (s *Store) HGetAll(key string) map[string][]byte {
	return s.hashStore.HGetAll(key)
}

func (s *Store) MGet(keys []string) map[string][]byte {
	if len(keys) == 0 {
		return nil
	}

	results := make(map[string][]byte, len(keys))

	for _, key := range keys {
		if val, ok := s.Get(key); ok {
			results[key] = val
		}
	}

	return results
}

func (s *Store) SemanticSearch(query []float32, k int) ([]SemanticResult, error) {
	if s.vectorStore == nil {
		return nil, fmt.Errorf("vector store not initialized")
	}
	if len(query) == 0 {
		return nil, fmt.Errorf("empty query")
	}
	defaultIndex := "default_embeddings"
	vectorResults, err := s.vectorStore.Search(defaultIndex, query, k)
	if err != nil {
		return nil, err
	}

	results := make([]SemanticResult, 0, len(vectorResults))
	for _, vr := range vectorResults {
		var value []byte
		if s != nil {
			if v, ok := s.Get(vr.ID); ok {
				value = v
			} else {
				value = nil
			}
		}
		results = append(results, SemanticResult{
			Key:      vr.ID,
			Value:    value,
			Distance: vr.Distance,
			Metadata: vr.Metadata,
		})
	}

	return results, nil
}

func (s *Store) PICAppend(chainID, prompt string, response []byte, metadata map[string]string) error {
	if chainID == "" {
		return ErrEmptyKey
	}
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(chainID, time.Now().UnixNano())
	}
	return s.picStore.Append(chainID, prompt, response, metadata)
}

func (s *Store) PICGet(chainID string, idx int) (data_types.InferenceChain, error) {
	if chainID == "" {
		return data_types.InferenceChain{}, ErrEmptyKey
	}
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(chainID, time.Now().UnixNano())
	}
	return s.picStore.Get(chainID, idx)
}

func (s *Store) MatrixSet(key string, rows, cols int, data []float32) error {
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(key, time.Now().UnixNano())
	}
	return s.pmcStore.Set(key, rows, cols, data)
}

func (s *Store) MatrixGet(key string) (data_types.Matrix, error) {
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(key, time.Now().UnixNano())
	}
	return s.pmcStore.Get(key)
}

func (s *Store) MatrixAdd(key1, key2 string) (data_types.Matrix, error) {
	return s.pmcStore.Add(key1, key2)
}

func (s *Store) MatrixMultiply(key1, key2 string) (data_types.Matrix, error) {
	return s.pmcStore.Multiply(key1, key2)
}

func (s *Store) offloadPGUS(data []byte) ([]byte, error) {
	ids := s.pgus.Write(data)
	if len(ids) == 0 {
		return nil, errors.New("pgus write failed: no ids returned")
	}

	buf := new(bytes.Buffer)
	buf.WriteString("P")

	if err := binary.Write(buf, binary.BigEndian, uint32(len(ids))); err != nil {
		return nil, err
	}
	for _, id := range ids {
		if err := binary.Write(buf, binary.BigEndian, uint64(id)); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

func (s *Store) resolvePGUS(meta []byte) ([]byte, bool) {
	if len(meta) < 5 || meta[0] != 'P' {
		return nil, false
	}

	count := binary.BigEndian.Uint32(meta[1:5])

	expectedLen := 5 + int(count)*8
	if len(meta) != expectedLen {
		return nil, false
	}

	ids := make([]storage.GranuleID, count)
	for i := 0; i < int(count); i++ {
		offset := 5 + (i * 8)
		ids[i] = storage.GranuleID(binary.BigEndian.Uint64(meta[offset:]))
	}

	return s.pgus.Read(ids), true
}

func (s *Store) cleanupPGUS(meta []byte) {
	if len(meta) < 5 || meta[0] != 'P' {
		return
	}
}

// FreeEntry implements eviction.StoreInterface expected by eviction.Manager.
// It finalizes freeing an evicted entry: returns arena memory, decrefs chunk/pgus and updates total bytes.
func (s *Store) FreeEntry(entry *common.Entry) {
	if entry == nil {
		return
	}

	val := entry.Value()

	// If chunked, decrement chunk refs asynchronously
	if s.isChunked(val) {
		s.cleanupChunks(val)
	}

	// If PGUS metadata ('P' prefix), cleanup PGUS
	if len(val) > 0 {
		if val[0] == 'P' || val[0] == pgusMagicByte {
			s.cleanupPGUS(val)
		}
	}

	// Free arena/slab memory if applicable
	var slabPtr *arena.Slab
	if ref := entry.GetSlab(); ref != nil {
		if casted, ok := ref.(*arena.Slab); ok {
			slabPtr = casted
		}
	}
	s.arena.Free(val, slabPtr)

	// Update total bytes counter
	atomic.AddInt64(&s.totalBytesAtomic, -int64(entry.Size()))
}

func (s *Store) recordAccess(key string) {
	if s.evictionManager != nil {
		s.evictionManager.RecordAccess(key, time.Now().UnixNano())
	}
	if s.freqSketch != nil {
		s.freqSketch.Increment(key)
	}
}

func (s *Store) postWriteOps(key string, value []byte) {
	s.trackCDC("PUT", key, value)
	if s.bloom != nil {
		s.bloom.Add(key)
	}
	if s.freqSketch != nil {
		s.freqSketch.Increment(key)
	}
}
