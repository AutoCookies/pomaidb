// File: internal/engine/eviction/tinylfu.go
package eviction

import (
	"math/rand"
	"time"
)

// AdmitOrRejectTinyLFU uses TinyLFU algorithm for admission control
func (m *Manager) AdmitOrRejectTinyLFU(key string, newSize int) bool {
	capacityBytes := m.store.GetCapacityBytes()

	// Unlimited capacity
	if capacityBytes == 0 {
		return true
	}

	// Room available
	currentBytes := m.store.GetTotalBytes()
	if currentBytes+int64(newSize) <= capacityBytes {
		return true
	}

	freqEstimator := m.store.GetFreqEstimator()
	if freqEstimator == nil {
		return true // No frequency tracking, admit
	}

	// Estimate frequency of new key
	newFreq := freqEstimator.Estimate(key) + 1

	// Try to find victim with lower frequency
	const maxAttempts = 3
	for i := 0; i < maxAttempts; i++ {
		victimKey, _, victimFreq := m.SampleVictim()

		if victimKey == "" {
			return true // No victim found, admit
		}

		// If new key is at least as frequent as victim, evict victim
		if newFreq >= victimFreq {
			freed := m.EvictKey(victimKey)
			if freed > 0 {
				// Check if we now have room
				if m.store.GetTotalBytes()+int64(newSize) <= capacityBytes {
					return true
				}
				// Try next victim
				continue
			}
			continue
		}

		// New key less frequent than victim, reject
		return false
	}

	// Final check after attempts
	return m.store.GetTotalBytes()+int64(newSize) <= capacityBytes
}

// SampleVictim randomly samples shards to find eviction victim
func (m *Manager) SampleVictim() (key string, size int, freq uint32) {
	shardCount := m.store.GetShardCount()
	if shardCount == 0 {
		return "", 0, 0
	}

	const (
		sampleShardLimit = 8
		perShardSamples  = 1
	)

	var bestKey string
	var bestSize int
	var bestFreq uint32 = ^uint32(0) // Max uint32

	freqEstimator := m.store.GetFreqEstimator()

	// Sample random shards
	for i := 0; i < sampleShardLimit; i++ {
		idx := rand.Intn(shardCount)
		sh := m.store.GetShardByIndex(idx)

		sh.RLock()

		elem := sh.GetLRUBack()
		count := 0

		for elem != nil && count < perShardSamples {
			ent := extractEntry(elem)

			// Skip expired
			if ent.ExpireAt != 0 && time.Now().UnixNano() > ent.ExpireAt {
				elem = getPrevElement(elem)
				continue
			}

			freq := uint32(0)
			if freqEstimator != nil {
				freq = freqEstimator.Estimate(ent.Key)
			}

			// Keep track of least frequent
			if freq < bestFreq {
				bestFreq = freq
				bestKey = ent.Key
				bestSize = ent.Size
			}

			elem = getPrevElement(elem)
			count++
		}

		sh.RUnlock()
	}

	if bestKey == "" {
		return "", 0, 0
	}

	return bestKey, bestSize, bestFreq
}

// ReserveSpace tries to reserve space, evicting if needed
func (m *Manager) ReserveSpace(newSize int, evictionThreshold float64) error {
	capacityBytes := m.store.GetCapacityBytes()

	// Check global memory controller first
	if memCtrl := m.store.GetGlobalMemCtrl(); memCtrl != nil && capacityBytes == 0 {
		// Global memory control only (no local capacity)
		// This would need Reserve() method on MemoryController
		// For now, return nil
		return nil
	}

	if capacityBytes == 0 {
		return nil // Unlimited
	}

	maxRetries := 5
	for retry := 0; retry < maxRetries; retry++ {
		currentBytes := m.store.GetTotalBytes()
		projectedBytes := currentBytes + int64(newSize)

		if projectedBytes <= capacityBytes {
			return nil // Room available
		}

		// Trigger eviction
		if err := m.EvictIfNeeded(evictionThreshold); err != nil {
			if retry == maxRetries-1 {
				return err
			}
			continue
		}
	}

	return ErrInsufficientStorage
}
