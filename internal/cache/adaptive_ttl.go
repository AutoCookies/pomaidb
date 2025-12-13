// File: internal/cache/adaptive_ttl.go
package cache

import (
	"math"
	"time"
)

type AdaptiveTTL struct {
	minTTL          time.Duration
	maxTTL          time.Duration
	baseAccessCount uint64
	decayFactor     float64
}

func NewAdaptiveTTL(minTTL, maxTTL time.Duration) *AdaptiveTTL {
	return &AdaptiveTTL{
		minTTL:      minTTL,
		maxTTL:      maxTTL,
		decayFactor: 0.9, // Hot items get longer TTL
	}
}

// ComputeTTL calculates adaptive TTL based on access patterns
func (a *AdaptiveTTL) ComputeTTL(accesses uint64, age time.Duration) time.Duration {
	if accesses == 0 {
		return a.minTTL
	}

	// Hotness score:  more accesses + recency = longer TTL
	hotnessScore := float64(accesses) * math.Exp(-age.Seconds()/3600) // Decay over 1 hour

	// Scale TTL between min and max
	ratio := math.Min(1.0, hotnessScore/100.0) // Cap at 100 accesses
	ttl := time.Duration(float64(a.minTTL) + ratio*float64(a.maxTTL-a.minTTL))

	return ttl
}
