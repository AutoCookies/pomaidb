// File: internal/persistence/noop_persister.go
package persistence

import (
	"github.com/AutoCookies/pomai-cache/internal/cache"
)

// NoOpPersister does nothing (for testing or when persistence is disabled)
type NoOpPersister struct{}

func NewNoOpPersister() *NoOpPersister {
	return &NoOpPersister{}
}

func (n *NoOpPersister) Persist(key string, value []byte) error {
	return nil
}

func (n *NoOpPersister) PersistBatch(ops []WriteOp) error {
	return nil
}

func (n *NoOpPersister) Load(key string) ([]byte, error) {
	return nil, nil
}

func (n *NoOpPersister) Snapshot(s *cache.Store) error {
	return nil
}

func (n *NoOpPersister) Restore(s *cache.Store) error {
	return nil
}

func (n *NoOpPersister) Close() error {
	return nil
}
