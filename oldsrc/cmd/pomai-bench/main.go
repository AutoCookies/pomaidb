package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/adapter/tcp"
)

const (
	OpGraphAddEdge      = 0x60
	OpGraphShortestPath = 0x61
	OpGraphPageRank     = 0x62
	OpGraphNeighbors    = 0x63
	OpGraphGetNode      = 0x64
)

var (
	addr            = flag.String("addr", "localhost:7600", "Server address")
	clients         = flag.Int("clients", 50, "Concurrent clients")
	requests        = flag.Int("requests", 1000000, "Total requests")
	pipeline        = flag.Int("pipeline", 128, "Pipeline batch size")
	dataSize        = flag.Int("data-size", 128, "Value size bytes")
	workloadRatio   = flag.Float64("ratio", 0.5, "Read ratio 0.0-1.0")
	runs            = flag.Int("runs", 3, "Benchmark runs")
	modeFlag        = flag.String("mode", "", "Mode selection")
	aiMode          = flag.Bool("ai-mode", false, "AI workload")
	vectorDim       = flag.Int("vector-dim", 128, "Vector dimension")
	semanticNoise   = flag.Float64("semantic-noise", 0.1, "Search noise")
	graphMode       = flag.Bool("graph-mode", false, "Graph workload")
	graphNodes      = flag.Int("graph-nodes", 100000, "Graph nodes")
	graphCompute    = flag.Bool("graph-compute", false, "Enable PageRank/Pathfinding")
	prIters         = flag.Int("pr-iters", 20, "PageRank iterations")
	pathDepth       = flag.Int("path-depth", 6, "Shortest path max depth")
	timeMode        = flag.Bool("time-mode", false, "TimeStream workload")
	timeStreams     = flag.Int("time-streams", 100, "Stream count")
	eventsPerStream = flag.Int("events-per-stream", 1000, "Events per stream")
	bitmapMode      = flag.Bool("bitmap-mode", false, "Bitmap workload")
	bitmapMaxOffset = flag.Uint64("bitmap-size", 100000000, "Max bit offset")
	zsetMode        = flag.Bool("zset-mode", false, "ZSet workload")
	zsetKeys        = flag.Int("zset-keys", 1000, "ZSet keys")
	zsetMembers     = flag.Int("zset-members", 10000, "ZSet members")
	hashMode        = flag.Bool("hash-mode", false, "Hash workload")
	hashFields      = flag.Int("hash-fields", 10, "Hash fields")
	picMode         = flag.Bool("pic-mode", false, "PIC workload")
	picChains       = flag.Int("pic-chains", 1000, "Inference chains")
	matrixMode      = flag.Bool("matrix-mode", false, "Matrix workload")
	matrixDim       = flag.Int("matrix-dim", 64, "Matrix dimension")
	plgMode         = flag.Bool("plg-mode", false, "Pomai Luu Graph workload")
	plgNodes        = flag.Int("plg-nodes", 10000, "PLG total nodes")
	hotRatio        = flag.Float64("hot-ratio", 0.0, "Percentage of requests hitting 10% of keys")
)

type BenchmarkResult struct {
	Duration   time.Duration
	TotalOps   uint64
	Throughput float64
	Bandwidth  float64
	AvgLatency float64
	MinLatency float64
	MaxLatency float64
	Latencies  []time.Duration
	Errors     uint64
}

type Config struct {
	addr            string
	clients         int
	requests        int
	pipeline        int
	dataSize        int
	ratio           float64
	runs            int
	hotRatio        float64
	aiMode          bool
	vectorDim       int
	semanticNoise   float64
	graphMode       bool
	graphNodes      int
	graphCompute    bool
	prIters         int
	pathDepth       int
	timeMode        bool
	timeStreams     int
	eventsPerStream int
	bitmapMode      bool
	bitmapMaxOffset uint64
	zsetMode        bool
	zsetKeys        int
	zsetMembers     int
	hashMode        bool
	hashFields      int
	picMode         bool
	picChains       int
	matrixMode      bool
	matrixDim       int
	plgMode         bool
	plgNodes        int
	Mode            string
}

type AIWorkload struct {
	keys       []string
	embeddings map[string][]float64
	mu         sync.RWMutex
}

type GraphWorkload struct {
	nodeIDs   []string
	graphName string
}

type TimeWorkload struct {
	streamKeys []string
}

type ZSetWorkload struct {
	keys []string
}

type PICWorkload struct {
	chainIDs []string
}

type PLGWorkload struct {
	nodes []string
}

type VectorPutReq struct {
	Data   []byte    `json:"data"`
	Vector []float32 `json:"vector"`
	TTL    string    `json:"ttl"`
}

type VectorSearchReq struct {
	Vector []float32 `json:"vector"`
	K      int       `json:"k"`
}

type BitSetReq struct {
	Offset uint64 `json:"offset"`
	Value  int    `json:"value"`
}

type BitGetReq struct {
	Offset uint64 `json:"offset"`
}

type BitCountReq struct {
	Start int64 `json:"start"`
	End   int64 `json:"end"`
}

type ZAddReq struct {
	Score  float64 `json:"score"`
	Member string  `json:"member"`
}

type ZRangeReq struct {
	Start int `json:"start"`
	Stop  int `json:"stop"`
}

func main() {
	flag.Parse()

	cfg := &Config{
		addr:            *addr,
		clients:         *clients,
		requests:        *requests,
		pipeline:        *pipeline,
		dataSize:        *dataSize,
		hotRatio:        *hotRatio,
		ratio:           *workloadRatio,
		runs:            *runs,
		aiMode:          *aiMode,
		vectorDim:       *vectorDim,
		semanticNoise:   *semanticNoise,
		graphMode:       *graphMode,
		graphNodes:      *graphNodes,
		graphCompute:    *graphCompute,
		prIters:         *prIters,
		pathDepth:       *pathDepth,
		timeMode:        *timeMode,
		timeStreams:     *timeStreams,
		eventsPerStream: *eventsPerStream,
		bitmapMode:      *bitmapMode,
		bitmapMaxOffset: *bitmapMaxOffset,
		zsetMode:        *zsetMode,
		zsetKeys:        *zsetKeys,
		zsetMembers:     *zsetMembers,
		hashMode:        *hashMode,
		hashFields:      *hashFields,
		picMode:         *picMode,
		picChains:       *picChains,
		matrixMode:      *matrixMode,
		matrixDim:       *matrixDim,
		plgMode:         *plgMode,
		plgNodes:        *plgNodes,
		Mode:            strings.ToLower(strings.TrimSpace(*modeFlag)),
	}

	if cfg.Mode != "" {
		resetConfig(cfg)
		switch cfg.Mode {
		case "standard", "kv":
		case "ai", "semantic", "vector":
			cfg.aiMode = true
		case "graph":
			cfg.graphMode = true
			cfg.graphCompute = true
		case "stream", "time":
			cfg.timeMode = true
		case "bitmap":
			cfg.bitmapMode = true
		case "zset":
			cfg.zsetMode = true
		case "hash":
			cfg.hashMode = true
		case "pic", "inference", "llm":
			cfg.picMode = true
		case "matrix", "pmc":
			cfg.matrixMode = true
		case "plg", "luu":
			cfg.plgMode = true
		default:
			log.Printf("Unknown mode %s", cfg.Mode)
		}
	}

	printBanner(cfg)

	var aiWorkload *AIWorkload
	if cfg.aiMode {
		aiWorkload = initAIWorkload(cfg)
	}

	var graphWorkload *GraphWorkload
	if cfg.graphMode {
		graphWorkload = initGraphWorkload(cfg)
	}

	var timeWorkload *TimeWorkload
	if cfg.timeMode {
		timeWorkload = initTimeWorkload(cfg)
	}

	var zsetWorkload *ZSetWorkload
	if cfg.zsetMode {
		zsetWorkload = initZSetWorkload(cfg)
	}

	var picWorkload *PICWorkload
	if cfg.picMode {
		picWorkload = initPICWorkload(cfg)
	}

	var plgWorkload *PLGWorkload
	if cfg.plgMode {
		plgWorkload = initPLGWorkload(cfg)
	}

	fmt.Printf("Warming up %d requests\n", cfg.requests/10)
	runBenchmarkPhase(cfg, cfg.requests/10, aiWorkload, graphWorkload, timeWorkload, zsetWorkload, picWorkload, plgWorkload, true)

	var results []BenchmarkResult
	for i := 0; i < cfg.runs; i++ {
		fmt.Printf("\nRun %d/%d:\n", i+1, cfg.runs)
		res := runBenchmarkPhase(cfg, cfg.requests, aiWorkload, graphWorkload, timeWorkload, zsetWorkload, picWorkload, plgWorkload, false)
		printResult(i+1, res)
		results = append(results, res)
		time.Sleep(1 * time.Second)
	}

	printAverageResults(results)
}

func resetConfig(cfg *Config) {
	cfg.aiMode = false
	cfg.graphMode = false
	cfg.timeMode = false
	cfg.bitmapMode = false
	cfg.zsetMode = false
	cfg.hashMode = false
	cfg.picMode = false
	cfg.matrixMode = false
	cfg.plgMode = false
	cfg.graphCompute = false
}

func initAIWorkload(cfg *Config) *AIWorkload {
	count := 100000
	keys := make([]string, count)
	embeddings := make(map[string][]float64)
	for i := 0; i < count; i++ {
		key := fmt.Sprintf("vec:%d", i)
		keys[i] = key
		embeddings[key] = generateRandomVector(cfg.vectorDim)
	}
	return &AIWorkload{keys: keys, embeddings: embeddings}
}

func initGraphWorkload(cfg *Config) *GraphWorkload {
	nodeIDs := make([]string, cfg.graphNodes)
	for i := 0; i < cfg.graphNodes; i++ {
		nodeIDs[i] = fmt.Sprintf("node_%d", i)
	}
	return &GraphWorkload{nodeIDs: nodeIDs, graphName: "bench_graph"}
}

func initTimeWorkload(cfg *Config) *TimeWorkload {
	keys := make([]string, cfg.timeStreams)
	for i := 0; i < cfg.timeStreams; i++ {
		keys[i] = fmt.Sprintf("stream:%d", i)
	}
	return &TimeWorkload{streamKeys: keys}
}

func initZSetWorkload(cfg *Config) *ZSetWorkload {
	keys := make([]string, cfg.zsetKeys)
	for i := 0; i < cfg.zsetKeys; i++ {
		keys[i] = fmt.Sprintf("leaderboard:%d", i)
	}
	return &ZSetWorkload{keys: keys}
}

func initPICWorkload(cfg *Config) *PICWorkload {
	ids := make([]string, cfg.picChains)
	for i := 0; i < cfg.picChains; i++ {
		ids[i] = fmt.Sprintf("session:%d", i)
	}
	return &PICWorkload{chainIDs: ids}
}

func initPLGWorkload(cfg *Config) *PLGWorkload {
	nodes := make([]string, cfg.plgNodes)
	for i := 0; i < cfg.plgNodes; i++ {
		nodes[i] = fmt.Sprintf("seed_%d", i)
	}
	return &PLGWorkload{nodes: nodes}
}

func runBenchmarkPhase(cfg *Config, totalReqs int, aiWorkload *AIWorkload, graphWorkload *GraphWorkload, timeWorkload *TimeWorkload, zsetWorkload *ZSetWorkload, picWorkload *PICWorkload, plgWorkload *PLGWorkload, warmup bool) BenchmarkResult {
	var wg sync.WaitGroup
	wg.Add(cfg.clients)

	start := time.Now()
	opsPerClient := totalReqs / cfg.clients
	latencies := make([]time.Duration, totalReqs)
	var latencyIdx int64 = -1
	var totalErrors uint64
	var totalBytes uint64

	dummyValue := make([]byte, cfg.dataSize)
	rand.Read(dummyValue)

	for i := 0; i < cfg.clients; i++ {
		go func(clientID int) {
			defer wg.Done()
			client, err := tcp.NewClient(cfg.addr)
			if err != nil {
				if !warmup {
					atomic.AddUint64(&totalErrors, 1)
				}
				return
			}
			defer client.Close()

			batchSize := cfg.pipeline
			if batchSize <= 0 {
				batchSize = 1
			}

			pipelineBuffer := make([]tcp.Packet, 0, batchSize)
			responseBuffer := make([]tcp.Packet, batchSize)

			for j := 0; j < opsPerClient; j++ {
				isRead := rand.Float64() < cfg.ratio
				var packet tcp.Packet

				if cfg.matrixMode {
					packet = buildMatrixPacket(cfg, clientID, j, isRead)
				} else if cfg.plgMode {
					packet = buildPLGPacket(cfg, plgWorkload, clientID, j, isRead)
				} else if cfg.zsetMode {
					packet = buildZSetPacket(cfg, zsetWorkload, clientID, j, isRead)
				} else if cfg.bitmapMode {
					packet = buildBitmapPacket(cfg, clientID, j, isRead)
				} else if cfg.aiMode {
					packet = buildAIPacket(cfg, aiWorkload, clientID, j, isRead, dummyValue)
				} else if cfg.graphMode {
					packet = buildGraphPacket(cfg, graphWorkload, clientID, j, isRead)
				} else if cfg.timeMode {
					packet = buildTimePacket(cfg, timeWorkload, clientID, j, isRead)
				} else if cfg.hashMode {
					packet = buildHashPacket(cfg, clientID, j, isRead, dummyValue)
				} else if cfg.picMode {
					packet = buildPICPacket(cfg, picWorkload, clientID, j, isRead, dummyValue)
				} else {
					packet = buildStandardPacket(cfg, clientID, j, isRead, dummyValue)
				}

				pipelineBuffer = append(pipelineBuffer, packet)

				if len(pipelineBuffer) == batchSize {
					reqStart := time.Now()
					err := client.PipelineFast(pipelineBuffer, responseBuffer)
					duration := time.Since(reqStart)
					if err != nil {
						atomic.AddUint64(&totalErrors, uint64(len(pipelineBuffer)))
					} else {
						avgLat := duration / time.Duration(batchSize)
						for k := 0; k < batchSize; k++ {
							idx := atomic.AddInt64(&latencyIdx, 1)
							if idx < int64(len(latencies)) {
								latencies[idx] = avgLat
							}
							atomic.AddUint64(&totalBytes, uint64(len(pipelineBuffer[k].Value)+len(responseBuffer[k].Value)))
						}
					}
					pipelineBuffer = pipelineBuffer[:0]
				}
			}
			if len(pipelineBuffer) > 0 {
				client.PipelineFast(pipelineBuffer, responseBuffer[:len(pipelineBuffer)])
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	validLatencies := make([]time.Duration, 0, totalReqs)
	var minLat, maxLat time.Duration
	var sumLat time.Duration
	minLat = time.Hour

	count := int(atomic.LoadInt64(&latencyIdx)) + 1
	if count > totalReqs {
		count = totalReqs
	}

	for i := 0; i < count; i++ {
		l := latencies[i]
		validLatencies = append(validLatencies, l)
		sumLat += l
		if l < minLat {
			minLat = l
		}
		if l > maxLat {
			maxLat = l
		}
	}
	sort.Slice(validLatencies, func(i, j int) bool {
		return validLatencies[i] < validLatencies[j]
	})

	avgLat := 0.0
	if len(validLatencies) > 0 {
		avgLat = float64(sumLat.Microseconds()) / float64(len(validLatencies)) / 1000.0
	}

	throughput := float64(totalReqs) / duration.Seconds()
	bandwidth := float64(totalBytes) / duration.Seconds() / 1024 / 1024

	return BenchmarkResult{
		Duration:   duration,
		TotalOps:   uint64(totalReqs),
		Throughput: throughput,
		Bandwidth:  bandwidth,
		AvgLatency: avgLat,
		MinLatency: float64(minLat.Microseconds()) / 1000.0,
		MaxLatency: float64(maxLat.Microseconds()) / 1000.0,
		Latencies:  validLatencies,
		Errors:     totalErrors,
	}
}

func buildStandardPacket(cfg *Config, clientID, reqIdx int, isRead bool, value []byte) tcp.Packet {
	var keyIndex int
	if rand.Float64() < cfg.hotRatio {
		limit := cfg.requests / 10
		if limit == 0 {
			limit = 1
		}
		keyIndex = rand.Intn(limit)
	} else {
		keyIndex = rand.Intn(cfg.requests)
	}

	key := fmt.Sprintf("key_%d", keyIndex)

	if isRead {
		return tcp.Packet{Opcode: tcp.OpGet, Key: key}
	}
	return tcp.Packet{Opcode: tcp.OpSet, Key: key, Value: value}
}

func buildHashPacket(cfg *Config, clientID, reqIdx int, isRead bool, value []byte) tcp.Packet {
	key := fmt.Sprintf("hkey_%d_%d", clientID, reqIdx%1000)
	field := fmt.Sprintf("f_%d", rand.Intn(cfg.hashFields))
	if isRead {
		if rand.Float64() < 0.8 {
			return tcp.Packet{Opcode: tcp.OpHGet, Key: key, Value: []byte(field)}
		} else {
			return tcp.Packet{Opcode: tcp.OpHGetAll, Key: key}
		}
	} else {
		fieldBytes := []byte(field)
		payload := make([]byte, 2+len(fieldBytes)+len(value))
		binary.BigEndian.PutUint16(payload[0:2], uint16(len(fieldBytes)))
		copy(payload[2:], fieldBytes)
		copy(payload[2+len(fieldBytes):], value)
		return tcp.Packet{Opcode: tcp.OpHSet, Key: key, Value: payload}
	}
}

func buildZSetPacket(cfg *Config, zw *ZSetWorkload, clientID, reqIdx int, isRead bool) tcp.Packet {
	key := zw.keys[rand.Intn(len(zw.keys))]
	if isRead {
		r := rand.Float64()
		if r < 0.5 {
			start := rand.Intn(100)
			stop := start + rand.Intn(20)
			req := ZRangeReq{Start: start, Stop: stop}
			payload, _ := json.Marshal(req)
			return tcp.Packet{Opcode: tcp.OpZRange, Key: key, Value: payload}
		} else if r < 0.75 {
			member := fmt.Sprintf("user:%d", rand.Intn(cfg.zsetMembers))
			return tcp.Packet{Opcode: tcp.OpZScore, Key: key, Value: []byte(member)}
		} else {
			member := fmt.Sprintf("user:%d", rand.Intn(cfg.zsetMembers))
			return tcp.Packet{Opcode: tcp.OpZRank, Key: key, Value: []byte(member)}
		}
	} else {
		member := fmt.Sprintf("user:%d", rand.Intn(cfg.zsetMembers))
		score := rand.Float64() * 10000
		req := ZAddReq{Member: member, Score: score}
		payload, _ := json.Marshal(req)
		return tcp.Packet{Opcode: tcp.OpZAdd, Key: key, Value: payload}
	}
}

func buildBitmapPacket(cfg *Config, clientID, reqIdx int, isRead bool) tcp.Packet {
	key := fmt.Sprintf("bench_bitmap_%d", clientID%10)
	if isRead {
		if rand.Float64() < 0.5 {
			offset := rand.Uint64() % cfg.bitmapMaxOffset
			req := BitGetReq{Offset: offset}
			payload, _ := json.Marshal(req)
			return tcp.Packet{Opcode: tcp.OpBitGet, Key: key, Value: payload}
		} else {
			req := BitCountReq{Start: 0, End: -1}
			payload, _ := json.Marshal(req)
			return tcp.Packet{Opcode: tcp.OpBitCount, Key: key, Value: payload}
		}
	} else {
		offset := rand.Uint64() % cfg.bitmapMaxOffset
		val := rand.Intn(2)
		req := BitSetReq{Offset: offset, Value: val}
		payload, _ := json.Marshal(req)
		return tcp.Packet{Opcode: tcp.OpBitSet, Key: key, Value: payload}
	}
}

func buildAIPacket(cfg *Config, aiWorkload *AIWorkload, clientID, reqIdx int, isRead bool, value []byte) tcp.Packet {
	toFloat32 := func(in []float64) []float32 {
		out := make([]float32, len(in))
		for i, v := range in {
			out[i] = float32(v)
		}
		return out
	}
	if isRead {
		targetIdx := (clientID*cfg.requests + reqIdx) % len(aiWorkload.keys)
		targetKey := aiWorkload.keys[targetIdx]
		aiWorkload.mu.RLock()
		targetVec := aiWorkload.embeddings[targetKey]
		aiWorkload.mu.RUnlock()
		queryVec := perturbVector(targetVec, cfg.semanticNoise)
		req := VectorSearchReq{Vector: toFloat32(queryVec), K: 5}
		payload, _ := json.Marshal(req)
		return tcp.Packet{Opcode: tcp.OpVectorSearch, Value: payload}
	}
	keyIdx := (clientID*cfg.requests + reqIdx) % len(aiWorkload.keys)
	key := aiWorkload.keys[keyIdx]
	aiWorkload.mu.RLock()
	embedding := aiWorkload.embeddings[key]
	aiWorkload.mu.RUnlock()
	req := VectorPutReq{Data: vectorToBytes(embedding), Vector: toFloat32(embedding), TTL: "1h"}
	payload, _ := json.Marshal(req)
	return tcp.Packet{Opcode: tcp.OpVectorPut, Key: key, Value: payload}
}

func buildGraphPacket(cfg *Config, gw *GraphWorkload, clientID, reqIdx int, isRead bool) tcp.Packet {
	if isRead {
		if cfg.graphCompute && rand.Float64() < 0.5 {
			if rand.Float64() < 0.5 {
				// PageRank
				payload := make([]byte, 4)
				binary.BigEndian.PutUint32(payload, uint32(cfg.prIters))
				return tcp.Packet{Opcode: OpGraphPageRank, Key: gw.graphName, Value: payload}
			} else {
				// ShortestPath
				node1 := gw.nodeIDs[rand.Intn(len(gw.nodeIDs))]
				node2 := gw.nodeIDs[rand.Intn(len(gw.nodeIDs))]
				n1Len := len(node1)
				n2Len := len(node2)
				payload := make([]byte, 4+2+n1Len+2+n2Len)
				binary.BigEndian.PutUint32(payload[0:4], uint32(cfg.pathDepth))
				binary.BigEndian.PutUint16(payload[4:6], uint16(n1Len))
				copy(payload[6:], node1)
				binary.BigEndian.PutUint16(payload[6+n1Len:], uint16(n2Len))
				copy(payload[8+n1Len:], node2)
				return tcp.Packet{Opcode: OpGraphShortestPath, Key: gw.graphName, Value: payload}
			}
		} else {
			// Cache Properties / Neighbors
			node := gw.nodeIDs[rand.Intn(len(gw.nodeIDs))]
			if rand.Float64() < 0.5 {
				return tcp.Packet{Opcode: OpGraphGetNode, Key: gw.graphName, Value: []byte(node)}
			}
			return tcp.Packet{Opcode: OpGraphNeighbors, Key: gw.graphName, Value: []byte(node)}
		}
	} else {
		// AddEdge
		from := gw.nodeIDs[rand.Intn(len(gw.nodeIDs))]
		to := gw.nodeIDs[rand.Intn(len(gw.nodeIDs))]
		weight := rand.Float64()
		// Payload: [Weight:8][FromLen:2][From][To]
		// Key: graphName
		fromLen := len(from)
		toLen := len(to)
		payload := make([]byte, 8+2+fromLen+toLen)
		binary.BigEndian.PutUint64(payload[0:8], math.Float64bits(weight))
		binary.BigEndian.PutUint16(payload[8:10], uint16(fromLen))
		copy(payload[10:], from)
		copy(payload[10+fromLen:], to)
		return tcp.Packet{Opcode: OpGraphAddEdge, Key: gw.graphName, Value: payload}
	}
}

func buildTimePacket(cfg *Config, tw *TimeWorkload, clientID, reqIdx int, isRead bool) tcp.Packet {
	stream := tw.streamKeys[rand.Intn(len(tw.streamKeys))]
	if isRead {
		return tcp.Packet{Opcode: tcp.OpStreamRange, Key: stream, Value: []byte(`{}`)}
	}
	payload := []byte(fmt.Sprintf(`{"val":%f}`, rand.Float64()))
	return tcp.Packet{Opcode: tcp.OpStreamAppend, Key: stream, Value: payload}
}

func buildPICPacket(cfg *Config, pw *PICWorkload, clientID, reqIdx int, isRead bool, value []byte) tcp.Packet {
	chainID := pw.chainIDs[rand.Intn(len(pw.chainIDs))]
	if isRead {
		idx := -1
		if rand.Float64() < 0.2 {
			idx = rand.Intn(10)
		}
		payload := make([]byte, 4)
		binary.BigEndian.PutUint32(payload, uint32(int32(idx)))
		return tcp.Packet{Opcode: tcp.OpPICGet, Key: chainID, Value: payload}
	} else {
		step := reqIdx % 20
		prompt := fmt.Sprintf("Explain step %d of caching", step)
		promptBytes := []byte(prompt)
		respBytes := value
		meta := fmt.Sprintf(`{"model":"gpt-4","step":%d}`, step)
		metaBytes := []byte(meta)
		pLen := len(promptBytes)
		rLen := len(respBytes)
		mLen := len(metaBytes)
		totalLen := 4 + pLen + 4 + rLen + mLen
		payload := make([]byte, totalLen)
		offset := 0
		binary.BigEndian.PutUint32(payload[offset:], uint32(pLen))
		offset += 4
		copy(payload[offset:], promptBytes)
		offset += pLen
		binary.BigEndian.PutUint32(payload[offset:], uint32(rLen))
		offset += 4
		copy(payload[offset:], respBytes)
		offset += rLen
		copy(payload[offset:], metaBytes)
		return tcp.Packet{Opcode: tcp.OpPICAppend, Key: chainID, Value: payload}
	}
}

func buildMatrixPacket(cfg *Config, clientID, reqIdx int, isRead bool) tcp.Packet {
	key1 := fmt.Sprintf("mat_%d_%d", clientID, reqIdx%100)
	key2 := fmt.Sprintf("mat_%d_%d", clientID, (reqIdx+1)%100)
	if isRead {
		if rand.Float64() < 0.5 {
			return tcp.Packet{Opcode: tcp.OpMatrixGet, Key: key1}
		} else {
			return tcp.Packet{Opcode: tcp.OpMatrixAdd, Key: key1, Value: []byte(key2)}
		}
	} else {
		rows, cols := cfg.matrixDim, cfg.matrixDim
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = rand.Float32()
		}
		buf := make([]byte, 8+len(data)*4)
		binary.BigEndian.PutUint32(buf[0:4], uint32(rows))
		binary.BigEndian.PutUint32(buf[4:8], uint32(cols))
		for i, v := range data {
			binary.BigEndian.PutUint32(buf[8+i*4:], math.Float32bits(v))
		}
		return tcp.Packet{Opcode: tcp.OpMatrixSet, Key: key1, Value: buf}
	}
}

func buildPLGPacket(cfg *Config, pw *PLGWorkload, clientID, reqIdx int, isRead bool) tcp.Packet {
	node1 := pw.nodes[rand.Intn(len(pw.nodes))]

	if isRead {
		minDensity := rand.Float64() * 0.8
		payload := make([]byte, 8)
		binary.BigEndian.PutUint64(payload, math.Float64bits(minDensity))
		return tcp.Packet{Opcode: tcp.OpPLGExtract, Key: node1, Value: payload}
	} else {
		node2 := pw.nodes[rand.Intn(len(pw.nodes))]
		weight := rand.Float64()
		node2Bytes := []byte(node2)
		payload := make([]byte, 8+len(node2Bytes))
		binary.BigEndian.PutUint64(payload[:8], math.Float64bits(weight))
		copy(payload[8:], node2Bytes)
		return tcp.Packet{Opcode: tcp.OpPLGAddEdge, Key: node1, Value: payload}
	}
}

func generateRandomVector(dim int) []float64 {
	vec := make([]float64, dim)
	var norm float64
	for i := 0; i < dim; i++ {
		vec[i] = rand.NormFloat64()
		norm += vec[i] * vec[i]
	}
	norm = math.Sqrt(norm)
	for i := 0; i < dim; i++ {
		vec[i] /= norm
	}
	return vec
}

func perturbVector(vec []float64, noise float64) []float64 {
	res := make([]float64, len(vec))
	var norm float64
	for i, v := range vec {
		res[i] = v + (rand.Float64()-0.5)*noise
		norm += res[i] * res[i]
	}
	norm = math.Sqrt(norm)
	for i := range res {
		res[i] /= norm
	}
	return res
}

func vectorToBytes(vec []float64) []byte {
	return []byte(fmt.Sprintf("%v", vec))
}

func printBanner(cfg *Config) {
	fmt.Println("========================================")
	fmt.Println("   POMAI CACHE BENCHMARK TOOL")
	fmt.Println("========================================")
	fmt.Printf("Server:         %s\n", cfg.addr)
	fmt.Printf("Clients:        %d\n", cfg.clients)
	fmt.Printf("Requests:       %d\n", cfg.requests)
	fmt.Printf("Pipeline:       %d\n", cfg.pipeline)
	fmt.Printf("Data Size:      %d bytes\n", cfg.dataSize)
	mode := "Standard (KV)"
	if cfg.Mode != "" {
		mode = strings.ToUpper(cfg.Mode)
	} else {
		if cfg.zsetMode {
			mode = "Sorted Set (ZSet)"
		} else if cfg.bitmapMode {
			mode = fmt.Sprintf("Bitmap (Offset: %d)", cfg.bitmapMaxOffset)
		} else if cfg.aiMode {
			mode = fmt.Sprintf("AI Semantic (Dim: %d)", cfg.vectorDim)
		} else if cfg.graphMode {
			mode = "Graph"
			if cfg.graphCompute {
				mode += " + Compute"
			}
		} else if cfg.timeMode {
			mode = "TimeStream"
		} else if cfg.hashMode {
			mode = fmt.Sprintf("Hash (Fields/Key: %d)", cfg.hashFields)
		} else if cfg.picMode {
			mode = fmt.Sprintf("PIC/Inference (Chains: %d)", cfg.picChains)
		} else if cfg.matrixMode {
			mode = fmt.Sprintf("Matrix (Dim: %dx%d)", cfg.matrixDim, cfg.matrixDim)
		} else if cfg.plgMode {
			mode = fmt.Sprintf("PLG/Luu (Nodes: %d)", cfg.plgNodes)
		}
	}
	fmt.Printf("Mode:           %s\n", mode)
	fmt.Println("========================================")
}

func printResult(run int, result BenchmarkResult) {
	fmt.Printf("[Run %d] Ops:%d TPS:%.0f BW:%.2f MB/s Lat:%.3fms Err:%d\n",
		run, result.TotalOps, result.Throughput, result.Bandwidth, result.AvgLatency, result.Errors)
	if len(result.Latencies) > 0 {
		fmt.Printf("P50:%.3f P99:%.3f Max:%.3f\n",
			percentile(result.Latencies, 0.50),
			percentile(result.Latencies, 0.99),
			result.MaxLatency)
	}
}

func percentile(latencies []time.Duration, p float64) float64 {
	if len(latencies) == 0 {
		return 0
	}
	idx := int(float64(len(latencies)) * p)
	if idx >= len(latencies) {
		idx = len(latencies) - 1
	}
	return float64(latencies[idx]) / 1e6
}

func printAverageResults(results []BenchmarkResult) {
	if len(results) == 0 {
		return
	}
	var avgThroughput, avgBandwidth, avgLatency, avgErrors float64
	for _, r := range results {
		avgThroughput += r.Throughput
		avgBandwidth += r.Bandwidth
		avgLatency += r.AvgLatency
		avgErrors += float64(r.Errors)
	}
	n := float64(len(results))
	fmt.Println("========================================")
	fmt.Println("AVERAGE RESULTS")
	fmt.Println("========================================")
	fmt.Printf("Throughput:     %.0f req/s\n", avgThroughput/n)
	fmt.Printf("Bandwidth:      %.2f MB/s\n", avgBandwidth/n)
	fmt.Printf("Avg Latency:    %.3f ms\n", avgLatency/n)
	fmt.Printf("Total Errors:   %.0f\n", avgErrors)
	fmt.Println("========================================")
}
