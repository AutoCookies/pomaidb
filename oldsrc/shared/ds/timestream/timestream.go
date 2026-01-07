package timestream

import (
	"encoding/json"
	"sync"
	"time"
)

const (
	BlockCapacity = 1024
)

// Event: Cấu trúc dùng để trao đổi dữ liệu với bên ngoài (API Layer)
type Event struct {
	ID        string                 `json:"id"`
	Timestamp int64                  `json:"timestamp"`
	Type      string                 `json:"type"`
	Value     float64                `json:"value"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// ---------------------------------------------------------
// TYPE REGISTRY (String Interning)
// ---------------------------------------------------------
type TypeRegistry struct {
	mu      sync.RWMutex
	strToId map[string]uint16
	idToStr []string
}

// Singleton Registry cho toàn bộ hệ thống TimeStream
var GlobalTypeRegistry = &TypeRegistry{
	strToId: make(map[string]uint16),
	idToStr: make([]string, 0),
}

func (r *TypeRegistry) GetID(s string) uint16 {
	if s == "" {
		return 0
	}
	r.mu.RLock()
	id, ok := r.strToId[s]
	r.mu.RUnlock()
	if ok {
		return id
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if id, ok = r.strToId[s]; ok {
		return id
	}
	newId := uint16(len(r.idToStr) + 1)
	r.idToStr = append(r.idToStr, s)
	r.strToId[s] = newId
	return newId
}

func (r *TypeRegistry) GetString(id uint16) string {
	if id == 0 {
		return ""
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	idx := int(id) - 1
	if idx < 0 || idx >= len(r.idToStr) {
		return ""
	}
	return r.idToStr[idx]
}

// ---------------------------------------------------------
// COLUMNAR BLOCK (SoA) - Exported Fields for AL usage
// ---------------------------------------------------------
type StreamBlock struct {
	// Các mảng này phải Public để package AL truy cập trực tiếp (Zero-overhead)
	Timestamps [BlockCapacity]int64
	Values     [BlockCapacity]float64
	TypeIDs    [BlockCapacity]uint16
	Metadatas  [BlockCapacity][]byte
	IDs        [BlockCapacity]string

	Count     int
	StartTime int64
	EndTime   int64
	Next      *StreamBlock
}

// Pool để tái sử dụng block, giảm GC pressure
var blockPool = sync.Pool{
	New: func() interface{} {
		return &StreamBlock{}
	},
}

func GetBlock() *StreamBlock {
	b := blockPool.Get().(*StreamBlock)
	b.Count = 0
	b.Next = nil
	b.StartTime = 0
	b.EndTime = 0
	return b
}

func PutBlock(b *StreamBlock) {
	// Clear pointers
	for i := 0; i < b.Count; i++ {
		b.Metadatas[i] = nil
		b.IDs[i] = ""
	}
	b.Count = 0
	blockPool.Put(b)
}

// ---------------------------------------------------------
// STREAM CONTAINER
// ---------------------------------------------------------
type Stream struct {
	Mu        sync.RWMutex
	Head      *StreamBlock
	Tail      *StreamBlock
	TotalSize int64

	// Thống kê nhanh (Cached Statistics)
	Sum       float64
	SqSum     float64
	Retention time.Duration
}

func NewStream(retention time.Duration) *Stream {
	if retention == 0 {
		retention = 24 * time.Hour
	}
	firstBlock := GetBlock()
	return &Stream{
		Head:      firstBlock,
		Tail:      firstBlock,
		Retention: retention,
	}
}

// Append thêm dữ liệu vào cuối Stream (Write path)
func (s *Stream) Append(events []*Event) {
	s.Mu.Lock()
	defer s.Mu.Unlock()

	now := time.Now().UnixNano()
	s.pruneLocked(now)

	for _, evt := range events {
		if evt.Timestamp == 0 {
			evt.Timestamp = now
		}

		if s.Tail.Count >= BlockCapacity {
			newBlock := GetBlock()
			s.Tail.Next = newBlock
			s.Tail = newBlock
		}

		idx := s.Tail.Count

		// Write to Columns
		s.Tail.Timestamps[idx] = evt.Timestamp
		s.Tail.Values[idx] = evt.Value
		s.Tail.TypeIDs[idx] = GlobalTypeRegistry.GetID(evt.Type)
		s.Tail.IDs[idx] = evt.ID

		if len(evt.Metadata) > 0 {
			if metaBytes, err := json.Marshal(evt.Metadata); err == nil {
				s.Tail.Metadatas[idx] = metaBytes
			}
		} else {
			s.Tail.Metadatas[idx] = nil
		}

		if s.Tail.Count == 0 {
			s.Tail.StartTime = evt.Timestamp
		}
		s.Tail.EndTime = evt.Timestamp
		s.Tail.Count++

		s.TotalSize++
		s.Sum += evt.Value
		s.SqSum += evt.Value * evt.Value
	}
}

func (s *Stream) pruneLocked(now int64) {
	cutoff := now - int64(s.Retention)
	for s.Head != s.Tail && s.Head.EndTime < cutoff {
		oldHead := s.Head
		// Update stats
		for i := 0; i < oldHead.Count; i++ {
			val := oldHead.Values[i]
			s.Sum -= val
			s.SqSum -= val * val
		}
		s.TotalSize -= int64(oldHead.Count)
		s.Head = s.Head.Next
		PutBlock(oldHead)
	}
}

// GetEvent hydrates Event object from columnar data at index i
func (b *StreamBlock) GetEvent(i int) *Event {
	e := &Event{
		ID:        b.IDs[i],
		Timestamp: b.Timestamps[i],
		Type:      GlobalTypeRegistry.GetString(b.TypeIDs[i]),
		Value:     b.Values[i],
	}
	if len(b.Metadatas[i]) > 0 {
		_ = json.Unmarshal(b.Metadatas[i], &e.Metadata)
	}
	return e
}
