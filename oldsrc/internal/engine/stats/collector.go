// File: internal/engine/stats/collector.go
package stats

// Collector collects statistics
type Collector struct {
	store StoreInterface
}

// StoreInterface defines required store methods
type StoreInterface interface {
	GetShards() []ShardInterface
	GetHits() uint64
	GetMisses() uint64
	GetEvictions() uint64
	GetTotalBytes() int64
	GetCapacityBytes() int64
	GetTenantID() string
}

// ShardInterface defines shard methods
type ShardInterface interface {
	GetItemCount() int
	GetBytes() int64
}

// NewCollector creates stats collector
func NewCollector(store StoreInterface) *Collector {
	return &Collector{store: store}
}

// Collect gathers current statistics
func (c *Collector) Collect() Stats {
	var totalItems int64
	var totalBytes int64

	shards := c.store.GetShards()
	for _, shard := range shards {
		totalItems += int64(shard.GetItemCount())
		totalBytes += shard.GetBytes()
	}

	hits := c.store.GetHits()
	misses := c.store.GetMisses()
	totalRequests := hits + misses

	hitRate := 0.0
	missRate := 0.0
	if totalRequests > 0 {
		hitRate = float64(hits) / float64(totalRequests) * 100
		missRate = float64(misses) / float64(totalRequests) * 100
	}

	capacity := c.store.GetCapacityBytes()
	usagePercent := 0.0
	if capacity > 0 {
		usagePercent = float64(totalBytes) / float64(capacity) * 100
	}

	avgItemSize := int64(0)
	if totalItems > 0 {
		avgItemSize = totalBytes / totalItems
	}

	return Stats{
		Hits:         hits,
		Misses:       misses,
		Items:        totalItems,
		Bytes:        totalBytes,
		Capacity:     capacity,
		Evictions:    c.store.GetEvictions(),
		ShardCount:   len(shards),
		HitRate:      hitRate,
		MissRate:     missRate,
		UsagePercent: usagePercent,
		AvgItemSize:  avgItemSize,
		TenantID:     c.store.GetTenantID(),
	}
}
