// File: internal/engine/ttl/adaptive.go
package ttl

import (
	"math"
	"time"
)

// AdaptiveTTL computes TTL based on access patterns
type AdaptiveTTL struct {
	minTTL          time.Duration
	maxTTL          time.Duration
	baseAccessCount uint64
	decayFactor     float64
}

// NewAdaptiveTTL creates adaptive TTL calculator
func NewAdaptiveTTL(minTTL, maxTTL time.Duration) *AdaptiveTTL {
	return &AdaptiveTTL{
		minTTL:          minTTL,
		maxTTL:          maxTTL,
		baseAccessCount: 100, // Base for normalization
		decayFactor:     0.9, // Hot items get longer TTL
	}
}

// ComputeTTL calculates adaptive TTL based on access patterns
//
// Algorithm:
//  1. Calculate hotness score = accesses * e^(-age/3600)
//  2. Normalize score to ratio [0, 1]
//  3. Interpolate between minTTL and maxTTL
//
// Examples:
//   - 0 accesses → minTTL
//   - 100 accesses, age=0 → maxTTL
//   - 100 accesses, age=1h → ~25min (decayed)
func (a *AdaptiveTTL) ComputeTTL(accesses uint64, age time.Duration) time.Duration {
	if accesses == 0 {
		return a.minTTL
	}

	// Hotness score:  more accesses + recency = longer TTL
	// Decay over 1 hour (3600 seconds)
	hotnessScore := float64(accesses) * math.Exp(-age.Seconds()/3600.0)

	// Normalize to [0, 1] range
	// Cap at baseAccessCount (default 100)
	ratio := math.Min(1.0, hotnessScore/float64(a.baseAccessCount))

	// Interpolate between min and max TTL
	ttl := time.Duration(
		float64(a.minTTL) + ratio*float64(a.maxTTL-a.minTTL),
	)

	return ttl
}

// GetMinTTL returns minimum TTL
func (a *AdaptiveTTL) GetMinTTL() time.Duration {
	return a.minTTL
}

// GetMaxTTL returns maximum TTL
func (a *AdaptiveTTL) GetMaxTTL() time.Duration {
	return a.maxTTL
}

// SetDecayFactor sets decay factor (default 0.9)
func (a *AdaptiveTTL) SetDecayFactor(factor float64) {
	if factor > 0 && factor <= 1.0 {
		a.decayFactor = factor
	}
}

// SetBaseAccessCount sets base access count for normalization
func (a *AdaptiveTTL) SetBaseAccessCount(count uint64) {
	if count > 0 {
		a.baseAccessCount = count
	}
}

// ComputeTTLWithOptions computes TTL with custom parameters
type TTLOptions struct {
	Accesses    uint64
	Age         time.Duration
	Priority    float64 // Priority multiplier (0-2)
	MinOverride time.Duration
	MaxOverride time.Duration
}

// ComputeTTLWithOptions computes TTL with advanced options
func (a *AdaptiveTTL) ComputeTTLWithOptions(opts TTLOptions) time.Duration {
	minTTL := a.minTTL
	maxTTL := a.maxTTL

	// Apply overrides
	if opts.MinOverride > 0 {
		minTTL = opts.MinOverride
	}
	if opts.MaxOverride > 0 {
		maxTTL = opts.MaxOverride
	}

	if opts.Accesses == 0 {
		return minTTL
	}

	// Calculate base hotness score
	hotnessScore := float64(opts.Accesses) * math.Exp(-opts.Age.Seconds()/3600.0)

	// Apply priority multiplier
	if opts.Priority > 0 {
		hotnessScore *= opts.Priority
	}

	// Normalize
	ratio := math.Min(1.0, hotnessScore/float64(a.baseAccessCount))

	// Interpolate
	ttl := time.Duration(
		float64(minTTL) + ratio*float64(maxTTL-minTTL),
	)

	return ttl
}

// EstimateOptimalTTL estimates optimal TTL for given access pattern
func (a *AdaptiveTTL) EstimateOptimalTTL(
	accessesPerHour uint64,
	avgQueryLatency time.Duration,
) time.Duration {
	// High access rate + expensive query → longer TTL
	costFactor := float64(avgQueryLatency.Milliseconds()) / 100.0
	accessFactor := float64(accessesPerHour) / 1000.0

	combinedScore := costFactor * accessFactor

	// Map to TTL range
	ratio := math.Min(1.0, combinedScore)
	ttl := time.Duration(
		float64(a.minTTL) + ratio*float64(a.maxTTL-a.minTTL),
	)

	return ttl
}

// TTLStats provides statistics about TTL distribution
type TTLStats struct {
	MinTTL      time.Duration
	MaxTTL      time.Duration
	AvgTTL      time.Duration
	MedianTTL   time.Duration
	TotalKeys   int
	ExpiredKeys int
}

// AnalyzeTTLDistribution analyzes TTL distribution (placeholder)
func (a *AdaptiveTTL) AnalyzeTTLDistribution() TTLStats {
	// This would need store access to analyze actual data
	return TTLStats{
		MinTTL: a.minTTL,
		MaxTTL: a.maxTTL,
	}
}
