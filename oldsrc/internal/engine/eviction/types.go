// File: internal/engine/eviction/types.go
package eviction

import (
	"errors"
	"sync/atomic"
)

var (
	ErrInsufficientStorage = errors.New("insufficient storage after eviction attempts")
	ErrTinyLFURejected     = errors.New("tiny-lfu admission control rejected")
)

// EvictionMetrics tracks eviction performance
type EvictionMetrics struct {
	TotalEvictions       atomic.Uint64
	AsyncEvictions       atomic.Uint64
	EmergencyEvictions   atomic.Uint64
	BytesFreed           atomic.Int64
	AvgEvictionTimeMs    atomic.Int64
	LastEvictionDuration atomic.Int64
}

// VictimCandidate represents a candidate for eviction
type VictimCandidate struct {
	Key  string
	Size int
	Freq uint32
}

// Candidate is legacy struct (kept for backward compatibility)
type Candidate struct {
	ShardIdx int
	Key      string
	Priority int64
	Size     int
}
