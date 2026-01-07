// File: internal/engine/eviction/legacy.go
package eviction

// Legacy functions kept for backward compatibility

// CollectCandidates is legacy function (deprecated)
func (m *Manager) CollectCandidates(start, shardLimit, samplesPerShard, maxCands int) []Candidate {
	cands := make([]Candidate, 0, 32)
	shardCount := m.store.GetShardCount()

	for i := 0; i < shardLimit && len(cands) < maxCands; i++ {
		idx := (start + i) % shardCount
		sh := m.store.GetShardByIndex(idx)

		sh.RLock()
		elem := sh.GetLRUBack()
		count := 0

		for elem != nil && count < samplesPerShard && len(cands) < maxCands {
			ent := extractEntry(elem)

			// Calculate priority (legacy algorithm)
			priority := ent.ExpireAt + int64(ent.Size)*1_000_000

			cands = append(cands, Candidate{
				ShardIdx: idx,
				Key:      ent.Key,
				Priority: priority,
				Size:     ent.Size,
			})

			elem = getPrevElement(elem)
			count++
		}

		sh.RUnlock()
	}

	return cands
}

// SelectVictim selects best victim from candidates (deprecated)
func (m *Manager) SelectVictim(cands []Candidate) Candidate {
	if len(cands) == 0 {
		return Candidate{}
	}

	best := cands[0]
	for _, c := range cands[1:] {
		if c.Priority < best.Priority {
			best = c
		}
	}

	return best
}

// TryReserveWithEviction is legacy function (deprecated)
func (m *Manager) TryReserveWithEviction(newSize int, threshold float64) error {
	if err := m.ReserveSpace(newSize, threshold); err != nil {
		return err
	}

	capacityBytes := m.store.GetCapacityBytes()
	newBytes := m.store.GetTotalBytes() + int64(newSize)

	if capacityBytes > 0 && newBytes > capacityBytes {
		// Rollback
		if memCtrl := m.store.GetGlobalMemCtrl(); memCtrl != nil {
			memCtrl.Release(int64(newSize))
		}
		return ErrInsufficientStorage
	}

	return nil
}
