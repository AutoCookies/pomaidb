// File: internal/engine/stats/types.go
package stats

// Stats represents cache statistics
type Stats struct {
	// Core Metrics
	Hits       uint64 `json:"hits"`
	Misses     uint64 `json:"misses"`
	Items      int64  `json:"items"`
	Bytes      int64  `json:"bytes"`
	Capacity   int64  `json:"capacity"`
	Evictions  uint64 `json:"evictions"`
	ShardCount int    `json:"shard_count"`

	// Performance
	HitRate      float64 `json:"hit_rate"`
	MissRate     float64 `json:"miss_rate"`
	UsagePercent float64 `json:"usage_percent"`
	AvgItemSize  int64   `json:"avg_item_size"`

	// Features
	AdaptiveTTLEnabled bool    `json:"adaptive_ttl_enabled"`
	BloomEnabled       bool    `json:"bloom_enabled"`
	BloomFPRate        float64 `json:"bloom_fp_rate"`
	BloomAvoided       uint64  `json:"bloom_avoided"`

	// Tenant
	TenantID string `json:"tenant_id"`
}

// EvictionStats tracks eviction metrics
type EvictionStats struct {
	TotalEvictions     uint64  `json:"total_evictions"`
	AsyncEvictions     uint64  `json:"async_evictions"`
	EmergencyEvictions uint64  `json:"emergency_evictions"`
	BytesFreed         int64   `json:"bytes_freed"`
	AvgTimeMs          int64   `json:"avg_time_ms"`
	InFlight           bool    `json:"in_flight"`
	ThresholdPercent   float64 `json:"threshold_percent"`
	TargetPercent      float64 `json:"target_percent"`
}

// HealthStatus represents health check result
type HealthStatus struct {
	Healthy        bool    `json:"healthy"`
	Status         string  `json:"status"`
	UsagePercent   float64 `json:"usage_percent"`
	EvictionActive bool    `json:"eviction_active"`
	Message        string  `json:"message,omitempty"`
}
