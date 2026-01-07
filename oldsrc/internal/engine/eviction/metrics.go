// File: internal/engine/eviction/metrics.go
package eviction

import (
	"fmt"
	"time"
)

// GetMetricsSnapshot returns current metrics snapshot
func (m *EvictionMetrics) GetSnapshot() EvictionMetricsSnapshot {
	return EvictionMetricsSnapshot{
		TotalEvictions:       m.TotalEvictions.Load(),
		AsyncEvictions:       m.AsyncEvictions.Load(),
		EmergencyEvictions:   m.EmergencyEvictions.Load(),
		BytesFreed:           m.BytesFreed.Load(),
		AvgEvictionTimeMs:    m.AvgEvictionTimeMs.Load(),
		LastEvictionDuration: m.LastEvictionDuration.Load(),
	}
}

// EvictionMetricsSnapshot is a point-in-time snapshot
type EvictionMetricsSnapshot struct {
	TotalEvictions       uint64
	AsyncEvictions       uint64
	EmergencyEvictions   uint64
	BytesFreed           int64
	AvgEvictionTimeMs    int64
	LastEvictionDuration int64
}

// Reset resets all metrics
func (m *EvictionMetrics) Reset() {
	m.TotalEvictions.Store(0)
	m.AsyncEvictions.Store(0)
	m.EmergencyEvictions.Store(0)
	m.BytesFreed.Store(0)
	m.AvgEvictionTimeMs.Store(0)
	m.LastEvictionDuration.Store(0)
}

// FormatBytes formats bytes into human-readable string
func FormatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}

	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}

	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// CalculateEvictionRate calculates evictions per second
func CalculateEvictionRate(totalEvictions uint64, startTime time.Time) float64 {
	if totalEvictions == 0 {
		return 0.0
	}

	duration := time.Since(startTime).Seconds()
	if duration <= 0 {
		return 0.0
	}

	return float64(totalEvictions) / duration
}

// CalculateAvgBytesPerEviction calculates average bytes freed per eviction
func CalculateAvgBytesPerEviction(bytesFreed int64, totalEvictions uint64) int64 {
	if totalEvictions == 0 {
		return 0
	}

	return bytesFreed / int64(totalEvictions)
}

// String returns formatted metrics string
func (s EvictionMetricsSnapshot) String() string {
	return fmt.Sprintf(
		"Evictions: %d (async: %d, emergency: %d), Freed: %s, Avg time: %dms",
		s.TotalEvictions,
		s.AsyncEvictions,
		s.EmergencyEvictions,
		FormatBytes(s.BytesFreed),
		s.AvgEvictionTimeMs,
	)
}
