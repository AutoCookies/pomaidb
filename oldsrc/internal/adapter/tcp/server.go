package tcp

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core"
	"github.com/AutoCookies/pomai-cache/internal/engine/core/data_types"
	"github.com/AutoCookies/pomai-cache/internal/engine/replication"
	"github.com/AutoCookies/pomai-cache/internal/engine/tenants"
	"github.com/AutoCookies/pomai-cache/shared/ds/timestream"
	"github.com/panjf2000/gnet/v2"
)

const (
	MaxValueSize = 10 * 1024 * 1024
)

type ServerMetrics struct {
	ActiveConnections atomic.Int64
	TotalRequests     atomic.Uint64
	TotalErrors       atomic.Uint64
	TotalBytes        atomic.Uint64
	StartTime         time.Time
}

func (m *ServerMetrics) GetStats() map[string]interface{} {
	uptime := time.Since(m.StartTime)
	totalReqs := m.TotalRequests.Load()
	totalBytes := m.TotalBytes.Load()
	totalErrors := m.TotalErrors.Load()

	var errorRate float64
	if totalReqs > 0 {
		errorRate = float64(totalErrors) / float64(totalReqs)
	}

	return map[string]interface{}{
		"active_connections": m.ActiveConnections.Load(),
		"total_requests":     totalReqs,
		"total_errors":       totalErrors,
		"total_bytes":        totalBytes,
		"uptime_seconds":     uptime.Seconds(),
		"requests_per_sec":   float64(totalReqs) / uptime.Seconds(),
		"bytes_per_sec":      float64(totalBytes) / uptime.Seconds(),
		"error_rate":         errorRate,
	}
}

type PomaiServer struct {
	gnet.BuiltinEventEngine

	tenants        *tenants.Manager
	repl           *replication.Manager
	metrics        *ServerMetrics
	addr           string
	eng            gnet.Engine
	connections    atomic.Int64
	maxConnections int64
	totalRequests  atomic.Uint64
	totalErrors    atomic.Uint64
	totalBytes     atomic.Uint64
	started        atomic.Bool
	startTime      time.Time
	multicore      bool
	numEventLoop   int
	reusePort      bool
	autoTuner      *core.AutoTuner
	ctx            context.Context
	cancel         context.CancelFunc
	mu             sync.Mutex
	timestreams    sync.Map
	plgStore       *data_types.PLGStore
}

type connCtx struct {
	tenantID string
	reqCount uint64
	created  int64
}

func NewPomaiServer(tm *tenants.Manager, rm *replication.Manager, maxConnections int) *PomaiServer {
	numLoops := runtime.NumCPU()
	if numLoops < 2 {
		numLoops = 2
	}
	if numLoops > 16 {
		numLoops = 16
	}

	ctx, cancel := context.WithCancel(context.Background())
	defaultStore := tm.GetStore("default")
	autoTuner := core.NewAutoTuner(defaultStore)

	s := &PomaiServer{
		tenants:        tm,
		repl:           rm,
		metrics:        &ServerMetrics{StartTime: time.Now()},
		multicore:      true,
		numEventLoop:   numLoops,
		reusePort:      true,
		ctx:            ctx,
		cancel:         cancel,
		autoTuner:      autoTuner,
		maxConnections: int64(maxConnections),
	}

	autoTuner.Start()
	return s
}

func (s *PomaiServer) ListenAndServe(addr string) error {
	s.addr = addr
	s.startTime = time.Now()
	s.started.Store(true)

	return gnet.Run(s, "tcp://"+addr,
		gnet.WithMulticore(s.multicore),
		gnet.WithReusePort(s.reusePort),
		gnet.WithTCPKeepAlive(time.Minute),
		gnet.WithTCPNoDelay(gnet.TCPNoDelay),
		gnet.WithReadBufferCap(256*1024),
		gnet.WithWriteBufferCap(256*1024),
		gnet.WithNumEventLoop(s.numEventLoop),
		gnet.WithTicker(true),
		gnet.WithSocketRecvBuffer(512*1024),
		gnet.WithSocketSendBuffer(512*1024),
		gnet.WithLoadBalancing(gnet.LeastConnections),
	)
}

func (s *PomaiServer) OnBoot(eng gnet.Engine) gnet.Action {
	s.eng = eng
	return gnet.None
}

func (s *PomaiServer) OnOpen(c gnet.Conn) ([]byte, gnet.Action) {
	if s.maxConnections > 0 && s.connections.Load() >= s.maxConnections {
		return nil, gnet.Close
	}

	s.connections.Add(1)
	ctx := &connCtx{
		tenantID: "default",
		created:  time.Now().UnixNano(),
	}
	c.SetContext(ctx)

	return nil, gnet.None
}

func (s *PomaiServer) OnClose(c gnet.Conn, err error) gnet.Action {
	s.connections.Add(-1)
	if err != nil {
		s.totalErrors.Add(1)
	}
	return gnet.None
}

func (s *PomaiServer) OnTraffic(c gnet.Conn) gnet.Action {
	buf, err := c.Peek(-1)
	if err != nil {
		s.totalErrors.Add(1)
		return gnet.Close
	}

	if len(buf) < HeaderSize {
		return gnet.None
	}

	processed := 0

	for len(buf) >= HeaderSize {
		if buf[0] != MagicByte {
			s.totalErrors.Add(1)
			return gnet.Close
		}

		opcode := buf[1]
		keyLen := binary.BigEndian.Uint16(buf[2:4])
		valLen := binary.BigEndian.Uint32(buf[4:8])

		packetSize := HeaderSize + int(keyLen) + int(valLen)

		if len(buf) < packetSize {
			break
		}

		packet := buf[:packetSize]
		s.handlePacketFast(c, packet, opcode, keyLen, valLen)

		buf = buf[packetSize:]
		processed += packetSize
	}

	if processed > 0 {
		c.Discard(processed)
	}

	return gnet.None
}

func (s *PomaiServer) handlePacketFast(c gnet.Conn, packet []byte, opcode uint8, keyLen uint16, valLen uint32) {
	if valLen > MaxValueSize {
		s.sendError(c, StatusInvalidRequest, "value too large")
		return
	}

	s.totalRequests.Add(1)
	s.totalBytes.Add(uint64(len(packet)))
	s.autoTuner.RecordRequest()

	ctx := c.Context().(*connCtx)
	ctx.reqCount++

	store := s.tenants.GetStore(ctx.tenantID)

	keyStart := HeaderSize
	keyEnd := keyStart + int(keyLen)
	valueStart := keyEnd
	valueEnd := valueStart + int(valLen)

	key := string(packet[keyStart:keyEnd])
	var value []byte
	if valLen > 0 {
		value = make([]byte, valLen)
		copy(value, packet[valueStart:valueEnd])
	}

	switch opcode {
	case OpGet:
		s.handleGetFast(c, store, key)
	case OpSet:
		s.handleSetFast(c, store, key, value)
	case OpDel:
		s.handleDelFast(c, store, key)
	case OpExists:
		s.handleExistsFast(c, store, key)
	case OpIncr:
		s.handleIncrFast(c, store, key, value)
	case OpStats:
		s.handleStatsFast(c, store)
	case OpMGet:
		s.handleMGetFast(c, store, value)
	case OpMSet:
		s.handleMSetFast(c, store, value)
	case OpStreamAppend:
		s.handleStreamAppendFast(c, ctx.tenantID, key, value)
	case OpStreamRange:
		s.handleStreamRangeFast(c, ctx.tenantID, key, value)
	case OpStreamWindow:
		s.handleStreamWindowFast(c, ctx.tenantID, key, value)
	case OpStreamAnomaly:
		s.handleStreamAnomalyFast(c, ctx.tenantID, key, value)
	case OpStreamForecast:
		s.handleStreamForecastFast(c, ctx.tenantID, key, value)
	case OpStreamPattern:
		s.handleStreamPatternFast(c, ctx.tenantID, key, value)
	case OpVectorPut:
		s.handleVectorPutFast(c, store, key, value)
	case OpVectorSearch:
		s.handleVectorSearchFast(c, store, value)
	case OpBitSet:
		s.handleBitSetFast(c, store, key, value)
	case OpBitGet:
		s.handleBitGetFast(c, store, key, value)
	case OpBitCount:
		s.handleBitCountFast(c, store, key, value)
	case OpStreamReadGroup:
		s.handleStreamReadGroupFast(c, ctx.tenantID, value)
	case OpCDCEnable:
		s.handleCDCEnableFast(c, store, value)
	case OpCDCGet:
		s.handleCDCGetFast(c, store, value)
	case OpZAdd:
		s.handleZAddFast(c, store, key, value)
	case OpZRem:
		s.handleZRemFast(c, store, key, value)
	case OpZScore:
		s.handleZScoreFast(c, store, key, value)
	case OpZRank:
		s.handleZRankFast(c, store, key, value)
	case OpZRange:
		s.handleZRangeFast(c, store, key, value)
	case OpZCard:
		s.handleZCardFast(c, store, key)
	case OpHSet:
		s.handleHSetFast(c, store, key, value)
	case OpHGet:
		s.handleHGetFast(c, store, key, value)
	case OpHDel:
		s.handleHDelFast(c, store, key, value)
	case OpHExists:
		s.handleHExistsFast(c, store, key, value)
	case OpHGetAll:
		s.handleHGetAllFast(c, store, key)
	case OpPICAppend:
		s.handlePICAppendFast(c, store, key, value)
	case OpPICGet:
		s.handlePICGetFast(c, store, key, value)
	case OpMatrixSet:
		s.handleMatrixSetFast(c, store, key, value)
	case OpMatrixGet:
		s.handleMatrixGetFast(c, store, key)
	case OpMatrixAdd:
		s.handleMatrixOpFast(c, store, key, value, 1)
	case OpMatrixMult:
		s.handleMatrixOpFast(c, store, key, value, 2)
	case OpClusterNodes:
		s.handleClusterNodesFast(c)
	case OpPLGAddEdge:
		s.handlePLGAddEdgeFast(c, store, key, value)
	case OpPLGExtract:
		s.handlePLGExtractFast(c, store, key, value)
	default:
		s.sendError(c, StatusInvalidRequest, "unknown command")
	}
}

func (s *PomaiServer) handleGetFast(c gnet.Conn, store *core.Store, key string) {
	value, ok := store.Get(key)
	if !ok {
		s.sendResponse(c, StatusKeyNotFound, nil)
		return
	}
	s.sendResponse(c, StatusOK, value)

	if s.repl != nil {
		ctx := c.Context().(*connCtx)
		go s.repl.BurstReplicate(ctx.tenantID, key, value, 0)
	}
}

func (s *PomaiServer) handleSetFast(c gnet.Conn, store *core.Store, key string, value []byte) {
	if key == "" {
		s.sendError(c, StatusInvalidRequest, "empty key")
		return
	}
	if err := store.Put(key, value, 0); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, nil)
}

func (s *PomaiServer) handleDelFast(c gnet.Conn, store *core.Store, key string) {
	if key != "" {
		store.Delete(key)
	}
	s.sendResponse(c, StatusOK, nil)
}

func (s *PomaiServer) handleExistsFast(c gnet.Conn, store *core.Store, key string) {
	if store.Exists(key) {
		s.sendResponse(c, StatusOK, []byte("1"))
	} else {
		s.sendResponse(c, StatusOK, []byte("0"))
	}
}

func (s *PomaiServer) handleIncrFast(c gnet.Conn, store *core.Store, key string, deltaBytes []byte) {
	delta := int64(1)
	if len(deltaBytes) > 0 {
		var err error
		delta, err = strconv.ParseInt(string(deltaBytes), 10, 64)
		if err != nil {
			s.sendError(c, StatusInvalidRequest, "invalid delta")
			return
		}
	}
	newVal, err := store.Incr(key, delta)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte(strconv.FormatInt(newVal, 10)))
}

func (s *PomaiServer) handleStatsFast(c gnet.Conn, store *core.Store) {
	stats := store.Stats()
	statsStr := fmt.Sprintf("items:%d\nbytes:%d\n", stats.Items, stats.Bytes)
	s.sendResponse(c, StatusOK, []byte(statsStr))
}

func (s *PomaiServer) handleMGetFast(c gnet.Conn, store *core.Store, data []byte) {
	keys := parseKeysFast(data)
	if len(keys) == 0 {
		s.sendError(c, StatusInvalidRequest, "no keys")
		return
	}
	results := store.MGet(keys)
	responseData := encodeMGetResponseFast(results)
	s.sendResponse(c, StatusOK, responseData)
}

func (s *PomaiServer) handleMSetFast(c gnet.Conn, store *core.Store, data []byte) {
	items, err := parseMSetRequestFast(data)
	if err != nil {
		s.sendError(c, StatusInvalidRequest, err.Error())
		return
	}
	if len(items) == 0 {
		s.sendError(c, StatusInvalidRequest, "no items")
		return
	}
	if err := store.MSet(items, 0); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, nil)
}

// handleStreamAppendFast now supports both JSON (single/object or array) and the binary format:
// [StreamLen:2][StreamName][IDLen:2][ID][Val:8 float64 BE][MetaLen:4][MetaBytes(JSON)]
// If the header key (from packet) is provided it will be used as streamName; otherwise streamName is read from payload.
func (s *PomaiServer) handleStreamAppendFast(c gnet.Conn, tenantID, headerStream string, payload []byte) {
	// Use tenant store (multi-tenant fix) rather than server-local timestreams
	store := s.tenants.GetStore(tenantID)
	if store == nil {
		s.sendError(c, StatusServerError, "tenant store not found")
		return
	}

	// JSON batch or single
	if len(payload) == 0 {
		s.sendError(c, StatusInvalidRequest, "empty event payload")
		return
	}
	first := payload[0]
	if first == '[' || first == '{' {
		// try to parse as JSON array or object
		// handle both array of events and single event object
		if first == '[' {
			var events []*timestream.Event
			if err := json.Unmarshal(payload, &events); err != nil {
				s.sendError(c, StatusInvalidRequest, "invalid batch event payload")
				return
			}
			if len(events) == 0 {
				s.sendError(c, StatusInvalidRequest, "empty event batch")
				return
			}
			// append via Store wrapper
			if err := store.StreamAppendBatch(headerStream, events); err != nil {
				s.sendError(c, StatusServerError, err.Error())
				return
			}
			s.sendResponse(c, StatusOK, nil)
			return
		} else {
			var evt timestream.Event
			if err := json.Unmarshal(payload, &evt); err != nil {
				s.sendError(c, StatusInvalidRequest, "invalid event payload")
				return
			}
			// if headerStream provided, use it; otherwise use evt.Type? evt.Type is not stream name.
			streamName := headerStream
			if streamName == "" {
				// If header empty, require evt.Metadata["stream"] maybe? For backward compatibility, require headerStream.
				s.sendError(c, StatusInvalidRequest, "empty stream name")
				return
			}
			if err := store.StreamAppend(streamName, evt.ID, evt.Value, evt.Metadata); err != nil {
				s.sendError(c, StatusServerError, err.Error())
				return
			}
			s.sendResponse(c, StatusOK, nil)
			return
		}
	}

	// Binary payload parsing
	// We'll accept either: payload contains streamName first OR headerStream provided in packet header.
	offset := 0
	payloadLen := len(payload)

	// Try read StreamLen and StreamName if headerStream is empty or payload starts with stream name
	var streamName string
	if payloadLen >= 2 {
		streamLen := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
		if headerStream == "" {
			// if we have enough bytes for stream name, read it
			if payloadLen >= 2+streamLen+2+8+4 {
				streamName = string(payload[offset+2 : offset+2+streamLen])
				offset += 2 + streamLen
			} else {
				s.sendError(c, StatusInvalidRequest, "payload too short for stream name")
				return
			}
		} else {
			// headerStream provided; detect if payload contains redundant StreamLen+StreamName matching headerStream => skip it
			if payloadLen >= 2+streamLen+2+8+4 {
				maybe := string(payload[offset+2 : offset+2+streamLen])
				if maybe == headerStream {
					offset += 2 + streamLen
				}
			}
			// use headerStream
			streamName = headerStream
		}
	} else if headerStream != "" {
		streamName = headerStream
	} else {
		s.sendError(c, StatusInvalidRequest, "payload too short")
		return
	}

	// Now expect [IDLen:2][ID][Val:8][MetaLen:4][MetaBytes]
	if payloadLen < offset+2+8+4 {
		s.sendError(c, StatusInvalidRequest, "payload too short for event fields")
		return
	}
	idLen := int(binary.BigEndian.Uint16(payload[offset : offset+2]))
	offset += 2
	if payloadLen < offset+idLen+8+4 {
		s.sendError(c, StatusInvalidRequest, "payload truncated for id/value/meta")
		return
	}
	id := string(payload[offset : offset+idLen])
	offset += idLen

	// read float64 big-endian
	valBits := binary.BigEndian.Uint64(payload[offset : offset+8])
	val := math.Float64frombits(valBits)
	offset += 8

	// meta len
	metaLen := int(binary.BigEndian.Uint32(payload[offset : offset+4]))
	offset += 4
	if metaLen < 0 || payloadLen < offset+metaLen {
		s.sendError(c, StatusInvalidRequest, "payload truncated for metadata")
		return
	}
	var meta map[string]interface{}
	if metaLen > 0 {
		metaBytes := payload[offset : offset+metaLen]
		if err := json.Unmarshal(metaBytes, &meta); err != nil {
			s.sendError(c, StatusInvalidRequest, "invalid metadata json")
			return
		}
	} else {
		meta = nil
	}

	// append to tenant store
	if err := store.StreamAppend(streamName, id, val, meta); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, nil)
}

func (s *PomaiServer) handleStreamRangeFast(c gnet.Conn, tenantID, streamName string, payload []byte) {
	if streamName == "" {
		s.sendError(c, StatusInvalidRequest, "empty stream name")
		return
	}
	req := struct {
		Start int64 `json:"start"`
		End   int64 `json:"end"`
	}{}
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &req); err != nil {
			s.sendError(c, StatusInvalidRequest, "invalid range payload")
			return
		}
	} else {
		now := time.Now().UnixNano()
		req.End = now
		req.Start = now - int64(time.Hour)
	}
	store := s.tenants.GetStore(tenantID)
	events, err := store.StreamRange(streamName, req.Start, req.End)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(events)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleStreamWindowFast(c gnet.Conn, tenantID, streamName string, payload []byte) {
	if streamName == "" {
		s.sendError(c, StatusInvalidRequest, "empty stream name")
		return
	}
	req := struct {
		WindowMs int64  `json:"window_ms"`
		Agg      string `json:"agg"`
	}{}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid window payload")
		return
	}
	if req.WindowMs <= 0 || req.Agg == "" {
		s.sendError(c, StatusInvalidRequest, "window_ms and agg required")
		return
	}
	store := s.tenants.GetStore(tenantID)
	result, err := store.StreamWindow(streamName, fmt.Sprintf("%dms", req.WindowMs), req.Agg)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(result)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleStreamAnomalyFast(c gnet.Conn, tenantID, streamName string, payload []byte) {
	if streamName == "" {
		s.sendError(c, StatusInvalidRequest, "empty stream name")
		return
	}
	req := struct {
		Threshold float64 `json:"threshold"`
	}{Threshold: 3.0}
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &req); err != nil {
			s.sendError(c, StatusInvalidRequest, "invalid anomaly payload")
			return
		}
	}
	store := s.tenants.GetStore(tenantID)
	events, err := store.StreamDetectAnomaly(streamName, req.Threshold)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(events)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleStreamForecastFast(c gnet.Conn, tenantID, streamName string, payload []byte) {
	if streamName == "" {
		s.sendError(c, StatusInvalidRequest, "empty stream name")
		return
	}
	req := struct {
		HorizonMs int64 `json:"horizon_ms"`
	}{HorizonMs: int64(time.Minute.Milliseconds())}
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &req); err != nil {
			s.sendError(c, StatusInvalidRequest, "invalid forecast payload")
			return
		}
	}
	store := s.tenants.GetStore(tenantID)
	pred, err := store.StreamForecast(streamName, fmt.Sprintf("%dms", req.HorizonMs))
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(map[string]float64{"predicted": pred})
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleStreamPatternFast(c gnet.Conn, tenantID, streamName string, payload []byte) {
	if streamName == "" {
		s.sendError(c, StatusInvalidRequest, "empty stream name")
		return
	}
	req := struct {
		Pattern  []string `json:"pattern"`
		WithinMs int64    `json:"within_ms"`
	}{}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid pattern payload")
		return
	}
	if len(req.Pattern) < 2 {
		s.sendError(c, StatusInvalidRequest, "pattern must have at least two types")
		return
	}
	if req.WithinMs <= 0 {
		req.WithinMs = int64(time.Minute.Milliseconds())
	}
	store := s.tenants.GetStore(tenantID)
	matches, err := store.StreamDetectPattern(streamName, req.Pattern, fmt.Sprintf("%dms", req.WithinMs))
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(matches)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleVectorPutFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	if key == "" {
		s.sendError(c, StatusInvalidRequest, "empty key")
		return
	}
	var req struct {
		Data   []byte    `json:"data"`
		Vector []float32 `json:"vector"`
		TTL    string    `json:"ttl"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json payload")
		return
	}
	if len(req.Vector) == 0 {
		s.sendError(c, StatusInvalidRequest, "empty vector")
		return
	}
	var ttl time.Duration
	if req.TTL != "" {
		var err error
		ttl, err = time.ParseDuration(req.TTL)
		if err != nil {
			s.sendError(c, StatusInvalidRequest, "invalid ttl format")
			return
		}
	}
	if err := store.PutWithEmbedding(key, req.Data, req.Vector, ttl); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, nil)
}

func (s *PomaiServer) handleVectorSearchFast(c gnet.Conn, store *core.Store, payload []byte) {
	var req struct {
		Vector []float32 `json:"vector"`
		K      int       `json:"k"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json payload")
		return
	}
	if req.K <= 0 {
		req.K = 5
	}
	results, err := store.SemanticSearch(req.Vector, req.K)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	respData, err := json.Marshal(results)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, respData)
}

func (s *PomaiServer) handleBitSetFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	if key == "" {
		s.sendError(c, StatusInvalidRequest, "empty key")
		return
	}
	var req struct {
		Offset uint64 `json:"offset"`
		Value  int    `json:"value"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json")
		return
	}
	orig, err := store.SetBit(key, req.Offset, req.Value)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte(strconv.Itoa(orig)))
}

func (s *PomaiServer) handleBitGetFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	if key == "" {
		s.sendError(c, StatusInvalidRequest, "empty key")
		return
	}
	var req struct {
		Offset uint64 `json:"offset"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json")
		return
	}
	val, err := store.GetBit(key, req.Offset)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte(strconv.Itoa(val)))
}

func (s *PomaiServer) handleBitCountFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	if key == "" {
		s.sendError(c, StatusInvalidRequest, "empty key")
		return
	}
	start := int64(0)
	end := int64(-1)
	if len(payload) > 0 {
		var req struct {
			Start int64 `json:"start"`
			End   int64 `json:"end"`
		}
		if err := json.Unmarshal(payload, &req); err == nil {
			start = req.Start
			end = req.End
		}
	}
	count, err := store.BitCount(key, start, end)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte(strconv.FormatInt(count, 10)))
}

func (s *PomaiServer) handleStreamReadGroupFast(c gnet.Conn, tenantID string, payload []byte) {
	var req struct {
		Stream string `json:"stream"`
		Group  string `json:"group"`
		Count  int    `json:"count"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json payload")
		return
	}
	if req.Stream == "" || req.Group == "" {
		s.sendError(c, StatusInvalidRequest, "stream and group are required")
		return
	}
	if req.Count <= 0 {
		req.Count = 10
	}
	store := s.tenants.GetStore(tenantID)
	events, err := store.ReadGroup(req.Stream, req.Group, req.Count)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(events)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleCDCEnableFast(c gnet.Conn, store *core.Store, payload []byte) {
	val := string(payload)
	enabled := val == "1" || val == "true"
	store.EnableCDC(enabled)
	s.sendResponse(c, StatusOK, []byte("OK"))
}

func (s *PomaiServer) handleCDCGetFast(c gnet.Conn, store *core.Store, payload []byte) {
	group := string(payload)
	if group == "" {
		group = "default_cdc_reader"
	}
	events, err := store.GetChanges(group, 100)
	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	resp, err := json.Marshal(events)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleZAddFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	if key == "" {
		s.sendError(c, StatusInvalidRequest, "empty key")
		return
	}
	var req struct {
		Score  float64 `json:"score"`
		Member string  `json:"member"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json")
		return
	}
	store.ZAdd(key, req.Score, req.Member)
	s.sendResponse(c, StatusOK, []byte("OK"))
}

func (s *PomaiServer) handleZRemFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	member := string(payload)
	removed := store.ZRem(key, member)
	if removed {
		s.sendResponse(c, StatusOK, []byte("1"))
	} else {
		s.sendResponse(c, StatusOK, []byte("0"))
	}
}

func (s *PomaiServer) handleZScoreFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	member := string(payload)
	score, ok := store.ZScore(key, member)
	if !ok {
		s.sendResponse(c, StatusKeyNotFound, nil)
		return
	}
	s.sendResponse(c, StatusOK, []byte(fmt.Sprintf("%f", score)))
}

func (s *PomaiServer) handleZRankFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	member := string(payload)
	rank := store.ZRank(key, member)
	if rank == -1 {
		s.sendResponse(c, StatusKeyNotFound, nil)
		return
	}
	s.sendResponse(c, StatusOK, []byte(strconv.Itoa(rank)))
}

func (s *PomaiServer) handleZRangeFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	var req struct {
		Start int `json:"start"`
		Stop  int `json:"stop"`
	}
	if err := json.Unmarshal(payload, &req); err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid json")
		return
	}
	items := store.ZRange(key, req.Start, req.Stop)
	resp, _ := json.Marshal(items)
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handleZCardFast(c gnet.Conn, store *core.Store, key string) {
	count := store.ZCard(key)
	s.sendResponse(c, StatusOK, []byte(strconv.Itoa(count)))
}

func (s *PomaiServer) handleHSetFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	if len(payload) < 2 {
		s.sendError(c, StatusInvalidRequest, "invalid hash payload")
		return
	}
	fieldLen := int(binary.BigEndian.Uint16(payload[0:2]))
	if len(payload) < 2+fieldLen {
		s.sendError(c, StatusInvalidRequest, "invalid field length")
		return
	}
	field := string(payload[2 : 2+fieldLen])
	value := payload[2+fieldLen:]
	if err := store.HSet(key, field, value); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte("OK"))
}

func (s *PomaiServer) handleHGetFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	field := string(payload)
	if field == "" {
		s.sendError(c, StatusInvalidRequest, "empty field")
		return
	}
	val, ok := store.HGet(key, field)
	if !ok {
		s.sendResponse(c, StatusKeyNotFound, nil)
		return
	}
	s.sendResponse(c, StatusOK, val)
}

func (s *PomaiServer) handleHDelFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	field := string(payload)
	if field == "" {
		s.sendError(c, StatusInvalidRequest, "empty field")
		return
	}
	if store.HDel(key, field) {
		s.sendResponse(c, StatusOK, []byte("1"))
	} else {
		s.sendResponse(c, StatusOK, []byte("0"))
	}
}

func (s *PomaiServer) handleHExistsFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	field := string(payload)
	if store.HExists(key, field) {
		s.sendResponse(c, StatusOK, []byte("1"))
	} else {
		s.sendResponse(c, StatusOK, []byte("0"))
	}
}

func (s *PomaiServer) handleHGetAllFast(c gnet.Conn, store *core.Store, key string) {
	results := store.HGetAll(key)
	if results == nil {
		s.sendResponse(c, StatusOK, []byte{0, 0})
		return
	}
	responseData := encodeMGetResponseFast(results)
	s.sendResponse(c, StatusOK, responseData)
}

func (s *PomaiServer) sendResponse(c gnet.Conn, status uint8, value []byte) {
	keyLen := uint16(0)
	valLen := uint32(len(value))
	packetLen := HeaderSize + len(value)
	buf := make([]byte, packetLen)
	buf[0] = MagicByte
	buf[1] = status
	binary.BigEndian.PutUint16(buf[2:4], keyLen)
	binary.BigEndian.PutUint32(buf[4:8], valLen)
	if len(value) > 0 {
		copy(buf[HeaderSize:], value)
	}
	c.AsyncWrite(buf, nil)
}

func (s *PomaiServer) sendError(c gnet.Conn, status uint8, message string) {
	s.totalErrors.Add(1)
	s.sendResponse(c, status, []byte(message))
}

func (s *PomaiServer) OnTick() (time.Duration, gnet.Action) {
	select {
	case <-s.ctx.Done():
		return 0, gnet.Shutdown
	default:
	}
	if !s.started.Load() {
		return time.Minute, gnet.None
	}
	uptime := time.Since(s.startTime)
	conns := s.connections.Load()
	reqs := s.totalRequests.Load()
	errs := s.totalErrors.Load()
	bytes := s.totalBytes.Load()
	rps := float64(0)
	bps := float64(0)
	errorRate := float64(0)
	if uptime.Seconds() > 0 {
		rps = float64(reqs) / uptime.Seconds()
		bps = float64(bytes) / uptime.Seconds() / 1024 / 1024
	}
	if reqs > 0 {
		errorRate = float64(errs) / float64(reqs) * 100
	}
	log.Printf("[TCP] Conns: %d | Reqs: %d | RPS: %.0f | BW: %.2f MB/s | Errors: %.2f%%",
		conns, reqs, rps, bps, errorRate)
	return 30 * time.Second, gnet.None
}

func (s *PomaiServer) Shutdown(timeout time.Duration) error {
	log.Println("[TCP] Shutting down...")
	if !s.started.Swap(false) {
		return nil
	}
	if s.cancel != nil {
		s.cancel()
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	s.mu.Lock()
	eng := s.eng
	s.mu.Unlock()
	if err := eng.Stop(ctx); err != nil {
		log.Printf("[TCP] Error stopping engine: %v", err)
	}
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for time.Now().Before(deadline) {
		if s.connections.Load() == 0 {
			break
		}
		<-ticker.C
	}
	return nil
}

func (s *PomaiServer) Stats() map[string]interface{} {
	uptime := time.Since(s.startTime)
	reqs := s.totalRequests.Load()
	bytes := s.totalBytes.Load()
	errs := s.totalErrors.Load()
	rps := float64(0)
	bps := float64(0)
	errorRate := float64(0)
	if uptime.Seconds() > 0 {
		rps = float64(reqs) / uptime.Seconds()
		bps = float64(bytes) / uptime.Seconds()
	}
	if reqs > 0 {
		errorRate = float64(errs) / float64(reqs) * 100
	}
	return map[string]interface{}{
		"server_type":        "gnet",
		"connections":        s.connections.Load(),
		"total_requests":     reqs,
		"total_errors":       errs,
		"total_bytes":        bytes,
		"uptime_seconds":     uptime.Seconds(),
		"requests_per_sec":   rps,
		"bytes_per_sec":      bps,
		"error_rate_percent": errorRate,
		"multicore":          s.multicore,
		"event_loops":        s.numEventLoop,
	}
}

func parseKeysFast(data []byte) []string {
	if len(data) < 2 {
		return nil
	}
	keys := make([]string, 0, 16)
	offset := 0
	dataLen := len(data)
	for offset+2 <= dataLen {
		keyLen := int(data[offset])<<8 | int(data[offset+1])
		offset += 2
		if offset+keyLen > dataLen {
			break
		}
		keys = append(keys, string(data[offset:offset+keyLen]))
		offset += keyLen
	}
	return keys
}

func encodeMGetResponseFast(results map[string][]byte) []byte {
	totalSize := 2
	for key, val := range results {
		totalSize += 2 + len(key) + 4 + len(val)
	}
	buf := make([]byte, totalSize)
	offset := 0
	count := len(results)
	buf[offset] = byte(count >> 8)
	buf[offset+1] = byte(count)
	offset += 2
	for key, val := range results {
		keyLen := len(key)
		buf[offset] = byte(keyLen >> 8)
		buf[offset+1] = byte(keyLen)
		offset += 2
		copy(buf[offset:], key)
		offset += keyLen
		valLen := len(val)
		buf[offset] = byte(valLen >> 24)
		buf[offset+1] = byte(valLen >> 16)
		buf[offset+2] = byte(valLen >> 8)
		buf[offset+3] = byte(valLen)
		offset += 4
		copy(buf[offset:], val)
		offset += valLen
	}
	return buf
}

func parseMSetRequestFast(data []byte) (map[string][]byte, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("invalid data: too short")
	}
	count := int(data[0])<<8 | int(data[1])
	if count <= 0 {
		return nil, fmt.Errorf("invalid count: %d", count)
	}
	items := make(map[string][]byte, count)
	offset := 2
	dataLen := len(data)
	for i := 0; i < count; i++ {
		if offset+2 > dataLen {
			return nil, fmt.Errorf("truncated at key %d length", i)
		}
		keyLen := int(data[offset])<<8 | int(data[offset+1])
		offset += 2
		if offset+keyLen > dataLen {
			return nil, fmt.Errorf("truncated at key %d data", i)
		}
		key := string(data[offset : offset+keyLen])
		offset += keyLen
		if offset+4 > dataLen {
			return nil, fmt.Errorf("truncated at value %d length", i)
		}
		valLen := int(data[offset])<<24 | int(data[offset+1])<<16 | int(data[offset+2])<<8 | int(data[offset+3])
		offset += 4
		if offset+valLen > dataLen {
			return nil, fmt.Errorf("truncated at value %d data", i)
		}
		val := make([]byte, valLen)
		copy(val, data[offset:offset+valLen])
		items[key] = val
		offset += valLen
	}
	return items, nil
}

func (s *PomaiServer) handlePICAppendFast(c gnet.Conn, store *core.Store, chainID string, payload []byte) {
	offset := 0
	if len(payload) < 4 {
		s.sendError(c, StatusInvalidRequest, "payload too short")
		return
	}

	promptLen := int(binary.BigEndian.Uint32(payload[offset : offset+4]))
	offset += 4
	if len(payload) < offset+promptLen {
		s.sendError(c, StatusInvalidRequest, "invalid prompt len")
		return
	}
	prompt := string(payload[offset : offset+promptLen])
	offset += promptLen

	if len(payload) < offset+4 {
		s.sendError(c, StatusInvalidRequest, "invalid resp header")
		return
	}
	respLen := int(binary.BigEndian.Uint32(payload[offset : offset+4]))
	offset += 4
	if len(payload) < offset+respLen {
		s.sendError(c, StatusInvalidRequest, "invalid resp len")
		return
	}
	response := payload[offset : offset+respLen]
	offset += respLen

	var metadata map[string]string
	if offset < len(payload) {
		metaBytes := payload[offset:]
		json.Unmarshal(metaBytes, &metadata)
	}

	if err := store.PICAppend(chainID, prompt, response, metadata); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte("OK"))
}

func (s *PomaiServer) handlePICGetFast(c gnet.Conn, store *core.Store, chainID string, payload []byte) {
	idx := -1
	if len(payload) >= 4 {
		idx = int(int32(binary.BigEndian.Uint32(payload)))
	}

	chainItem, err := store.PICGet(chainID, idx)
	if err != nil {
		s.sendError(c, StatusKeyNotFound, err.Error())
		return
	}

	respBytes, err := json.Marshal(chainItem)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, respBytes)
}

func (s *PomaiServer) handleMatrixSetFast(c gnet.Conn, store *core.Store, key string, payload []byte) {
	// Payload: [Rows:4][Cols:4][Float32 Data...]
	mat, err := data_types.DecodeMatrix(payload)
	if err != nil {
		s.sendError(c, StatusInvalidRequest, "invalid matrix format")
		return
	}
	if err := store.MatrixSet(key, mat.Rows, mat.Cols, mat.Data); err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, []byte("OK"))
}

func (s *PomaiServer) handleMatrixGetFast(c gnet.Conn, store *core.Store, key string) {
	mat, err := store.MatrixGet(key)
	if err != nil {
		s.sendError(c, StatusKeyNotFound, "matrix not found")
		return
	}
	data := data_types.EncodeMatrix(mat)
	s.sendResponse(c, StatusOK, data)
}

func (s *PomaiServer) handleMatrixOpFast(c gnet.Conn, store *core.Store, key1 string, payload []byte, op int) {
	key2 := string(payload)
	if key1 == "" || key2 == "" {
		s.sendError(c, StatusInvalidRequest, "keys required")
		return
	}

	var result data_types.Matrix
	var err error

	if op == 1 {
		result, err = store.MatrixAdd(key1, key2)
	} else {
		result, err = store.MatrixMultiply(key1, key2)
	}

	if err != nil {
		s.sendError(c, StatusServerError, err.Error())
		return
	}
	s.sendResponse(c, StatusOK, data_types.EncodeMatrix(result))
}

func (s *PomaiServer) handleClusterNodesFast(c gnet.Conn) {
	if s.repl == nil {
		s.sendError(c, StatusServerError, "cluster mode disabled")
		return
	}

	stats := s.repl.GetStats()
	resp, err := json.Marshal(map[string]interface{}{
		"active_peers":  stats.ActivePeers,
		"healthy_peers": stats.HealthyPeers,
		"total_ops":     stats.TotalOps,
		"lag_ms":        stats.CurrentLag,
		"role":          "master",
	})

	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}

func (s *PomaiServer) handlePLGAddEdgeFast(c gnet.Conn, store *core.Store, node1 string, payload []byte) {
	if len(payload) < 8 {
		s.sendError(c, StatusInvalidRequest, "invalid payload length")
		return
	}

	weightBits := binary.BigEndian.Uint64(payload[:8])
	weight := math.Float64frombits(weightBits)
	node2 := string(payload[8:])

	if node1 == "" || node2 == "" {
		s.sendError(c, StatusInvalidRequest, "empty nodes")
		return
	}

	store.PLGAddEdge(node1, node2, weight)
	s.sendResponse(c, StatusOK, []byte("OK"))
}

func (s *PomaiServer) handlePLGExtractFast(c gnet.Conn, store *core.Store, startNode string, payload []byte) {
	if len(payload) < 8 {
		s.sendError(c, StatusInvalidRequest, "invalid payload length")
		return
	}

	densityBits := binary.BigEndian.Uint64(payload[:8])
	minDensity := math.Float64frombits(densityBits)

	if startNode == "" {
		s.sendError(c, StatusInvalidRequest, "empty start node")
		return
	}

	cluster := store.PLGExtractCluster(startNode, minDensity)

	resp, err := json.Marshal(cluster)
	if err != nil {
		s.sendError(c, StatusServerError, "marshal error")
		return
	}
	s.sendResponse(c, StatusOK, resp)
}
