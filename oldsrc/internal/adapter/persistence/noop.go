package persistence

import (
	"github.com/AutoCookies/pomai-cache/internal/core/ports"
)

type NoOpPersister struct{}

func NewNoOpPersister() *NoOpPersister {
	return &NoOpPersister{}
}

func (n *NoOpPersister) Persist(key string, value []byte) error {
	return nil
}

func (n *NoOpPersister) PersistBatch(ops []ports.WriteOp) error {
	return nil
}

func (n *NoOpPersister) Load(key string) ([]byte, error) {
	return nil, nil
}

func (n *NoOpPersister) Snapshot(target ports.Serializable) error {
	return nil
}

func (n *NoOpPersister) Restore(target ports.Serializable) error {
	return nil
}

func (n *NoOpPersister) Close() error {
	return nil
}
