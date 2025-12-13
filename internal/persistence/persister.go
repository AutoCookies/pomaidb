// File: internal/persistence/persister.go
package persistence

import (
	"github.com/AutoCookies/pomai-cache/internal/cache"
)

// Persister interface for different persistence backends
type Persister interface {
	// Persist writes a single key-value pair
	Persist(key string, value []byte) error

	// PersistBatch writes multiple key-value pairs atomically (if supported)
	PersistBatch(ops []WriteOp) error

	// Load reads a value by key (optional, for recovery)
	Load(key string) ([]byte, error)

	// Snapshot creates a full snapshot of the store
	Snapshot(s *cache.Store) error

	// Restore restores data from snapshot
	Restore(s *cache.Store) error

	// Close closes the persister and flushes pending data
	Close() error
}
