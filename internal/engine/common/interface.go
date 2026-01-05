// File: internal/engine/common/interfaces.go
package common

import (
	"errors"
	"time"
)

// MemoryController manages memory allocation
type MemoryController interface {
	Reserve(bytes int64) bool
	Release(bytes int64)
	Used() int64
	Capacity() int64
}

// AdaptiveTTLManager computes adaptive TTL
type AdaptiveTTLManager interface {
	ComputeTTL(accesses uint64, age time.Duration) time.Duration
	GetMinTTL() time.Duration
	GetMaxTTL() time.Duration
}

// ReplicationManager handles replication
type ReplicationManager interface {
	Replicate(op ReplicaOp) error
	GetStats() interface{}
	Shutdown()
}

// ReplicaOp represents a replication operation
type ReplicaOp struct {
	Type      OpType
	Key       string
	Value     []byte
	Delta     int64
	TTL       time.Duration
	TenantID  string
	Timestamp int64
	SeqNum    uint64
}

// OpType represents operation type
type OpType uint8

const (
	OpTypeSet OpType = iota + 1
	OpTypeDelete
	OpTypeIncr
)

func (t OpType) String() string {
	switch t {
	case OpTypeSet:
		return "SET"
	case OpTypeDelete:
		return "DELETE"
	case OpTypeIncr:
		return "INCR"
	default:
		return "UNKNOWN"
	}
}

var (
	ErrEmptyKey            = errors.New("empty key")
	ErrInsufficientStorage = errors.New("insufficient storage")
	ErrValueNotInteger     = errors.New("value is not an integer")
	ErrKeyNotFound         = errors.New("key not found")
	ErrCorruptData         = errors.New("corrupted chunk data")
)
