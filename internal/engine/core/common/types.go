// File: internal/engine/core/types.go
package common

import (
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/common"
)

// StoreConfig configures the cache store
type StoreConfig struct {
	ShardCount        int
	CapacityBytes     int64
	FreqBoost         int64
	EvictionThreshold float64
	EvictionTarget    float64
	TenantID          string
}

// DefaultStoreConfig returns default configuration
func DefaultStoreConfig() *StoreConfig {
	return &StoreConfig{
		ShardCount:        256,
		CapacityBytes:     0,
		FreqBoost:         1_000_000,
		EvictionThreshold: 0.80,
		EvictionTarget:    0.70,
		TenantID:          "default",
	}
}

// CacheOptions for Put operations
type CacheOptions struct {
	TTL         time.Duration
	IfNotExists bool
	IfExists    bool
	ReturnOld   bool
}

type OpType = common.OpType
type ReplicaOp = common.ReplicaOp

const (
	OpTypeSet    = common.OpTypeSet
	OpTypeDelete = common.OpTypeDelete
	OpTypeIncr   = common.OpTypeIncr
)

type BloomStats struct {
	Hits              uint64
	Misses            uint64
	Avoided           uint64
	FalsePositiveRate float64
}

type EvictionMetrics struct {
	TotalEvictions       uint64
	AsyncEvictions       uint64
	EmergencyEvictions   uint64
	BytesFreed           int64
	AvgEvictionTimeMs    int64
	LastEvictionDuration int64
}

type Stats struct {
	Hits         uint64  `json:"hits"`
	Misses       uint64  `json:"misses"`
	Items        int64   `json:"items"`
	Bytes        int64   `json:"bytes"`
	Capacity     int64   `json:"capacity"`
	Evictions    uint64  `json:"evictions"`
	ShardCount   int     `json:"shard_count"`
	HitRate      float64 `json:"hit_rate"`
	MissRate     float64 `json:"miss_rate"`
	UsagePercent float64 `json:"usage_percent"`
	AvgItemSize  int64   `json:"avg_item_size"`
	TenantID     string  `json:"tenant_id"`
}

type AdaptiveTTLManager = common.AdaptiveTTLManager
type MemoryController = common.MemoryController
type ReplicationManager = common.ReplicationManager
