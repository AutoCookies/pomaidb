// File: internal/engine/persistence/snapshot.go
package persistence

import (
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"log"
	"time"
)

// Manager handles snapshot/restore operations
type Manager struct {
	store StoreInterface
}

// NewManager creates snapshot manager
func NewManager(store StoreInterface) *Manager {
	return &Manager{store: store}
}

// SnapshotTo writes cache data (KV + ZSet) to writer
func (m *Manager) SnapshotTo(w io.Writer) error {
	enc := gob.NewEncoder(w)

	// 1. Write version
	if err := enc.Encode(SnapshotVersionV2); err != nil {
		return fmt.Errorf("failed to write version: %w", err)
	}

	// 2. Snapshot KV data
	if err := m.snapshotKV(enc); err != nil {
		return fmt.Errorf("failed to snapshot KV: %w", err)
	}

	// 3. Snapshot ZSet data
	if err := m.snapshotZSets(enc); err != nil {
		return fmt.Errorf("failed to snapshot ZSets: %w", err)
	}

	log.Printf("[SNAPSHOT] Snapshot completed")
	return nil
}

// snapshotKV writes KV data
func (m *Manager) snapshotKV(enc *gob.Encoder) error {
	shards := m.store.GetShards()
	itemCount := 0

	for _, shard := range shards {
		shard.RLock()
		items := shard.GetItems()

		for _, elem := range items {
			entry := extractEntry(elem)
			if entry == nil {
				continue
			}

			// Skip expired entries
			if entry.IsExpired() {
				continue
			}

			item := SnapshotItem{
				Type:       0, // KV
				Key:        entry.Key(),
				Value:      entry.Value(),
				ExpireAt:   entry.ExpireAt(),
				LastAccess: entry.LastAccess(),
				Accesses:   entry.Accesses(),
				CreatedAt:  entry.CreatedAt(),
			}

			if err := enc.Encode(&item); err != nil {
				shard.RUnlock()
				return fmt.Errorf("failed to encode item: %w", err)
			}
			itemCount++
		}

		shard.RUnlock()
	}

	log.Printf("[SNAPSHOT] KV:  %d items", itemCount)
	return nil
}

// snapshotZSets writes ZSet data
func (m *Manager) snapshotZSets(enc *gob.Encoder) error {
	zsets := m.store.GetZSets()
	zsetCount := 0

	for key, zset := range zsets {
		nodes := zset.Dump()

		members := make([]ZMember, len(nodes))
		for i, node := range nodes {
			members[i] = ZMember{
				Member: node.Member,
				Score:  node.Score,
			}
		}

		item := SnapshotItem{
			Type:     1, // ZSet
			Key:      key,
			ZMembers: members,
		}

		if err := enc.Encode(&item); err != nil {
			return fmt.Errorf("failed to encode zset:  %w", err)
		}
		zsetCount++
	}

	log.Printf("[SNAPSHOT] ZSets: %d sets", zsetCount)
	return nil
}

// RestoreFrom reads cache data from reader
func (m *Manager) RestoreFrom(r io.Reader) error {
	dec := gob.NewDecoder(r)

	// 1. Read version
	var version int
	if err := dec.Decode(&version); err != nil {
		return fmt.Errorf("failed to read version:  %w", err)
	}

	if version != SnapshotVersionV1 && version != SnapshotVersionV2 {
		return fmt.Errorf("unsupported snapshot version: %d", version)
	}

	// 2. Restore items
	kvCount := 0
	zsetCount := 0

	for {
		var item SnapshotItem
		err := dec.Decode(&item)

		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to decode item: %w", err)
		}

		if item.Type == 0 {
			// Restore KV
			if err := m.restoreKVItem(item); err != nil {
				log.Printf("[SNAPSHOT] Failed to restore KV item %s: %v", item.Key, err)
				continue
			}
			kvCount++
		} else if item.Type == 1 {
			// Restore ZSet
			if err := m.restoreZSetItem(item); err != nil {
				log.Printf("[SNAPSHOT] Failed to restore ZSet %s: %v", item.Key, err)
				continue
			}
			zsetCount++
		}
	}

	log.Printf("[SNAPSHOT] Restore completed: %d KV items, %d ZSets", kvCount, zsetCount)
	return nil
}

// restoreKVItem restores single KV item
func (m *Manager) restoreKVItem(item SnapshotItem) error {
	// Skip expired items
	if item.ExpireAt != 0 && time.Now().UnixNano() > item.ExpireAt {
		return nil
	}

	// This will be implemented by store
	// For now, return placeholder
	return errors.New("not implemented - store must implement restoration")
}

// restoreZSetItem restores single ZSet
func (m *Manager) restoreZSetItem(item SnapshotItem) error {
	// This will be implemented by store
	return errors.New("not implemented - store must implement restoration")
}

// Helper:  Extract entry from interface
func extractEntry(elem interface{}) EntryInterface {
	if entry, ok := elem.(EntryInterface); ok {
		return entry
	}
	return nil
}
