// File: internal/engine/replication/oplog.go
package replication

import (
	"sync"
)

// OpLog maintains operation history for replication
type OpLog struct {
	mu      sync.RWMutex
	entries []ReplicaOp
	maxSize int
	head    uint64
}

// NewOpLog creates operation log
func NewOpLog(maxSize int) *OpLog {
	if maxSize <= 0 {
		maxSize = 100000
	}
	return &OpLog{
		entries: make([]ReplicaOp, 0, maxSize),
		maxSize: maxSize,
		head:    0,
	}
}

// Append adds operation to log and returns sequence number
func (ol *OpLog) Append(op ReplicaOp) uint64 {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	ol.head++
	op.SeqNum = ol.head

	// Circular buffer behavior
	if len(ol.entries) >= ol.maxSize {
		ol.entries = ol.entries[1:]
	}

	ol.entries = append(ol.entries, op)
	return ol.head
}

// GetSince retrieves operations after given sequence number
func (ol *OpLog) GetSince(seqNum uint64) []ReplicaOp {
	ol.mu.RLock()
	defer ol.mu.RUnlock()

	result := make([]ReplicaOp, 0)
	for _, op := range ol.entries {
		if op.SeqNum > seqNum {
			result = append(result, op)
		}
	}
	return result
}

// GetHead returns current sequence number
func (ol *OpLog) GetHead() uint64 {
	ol.mu.RLock()
	defer ol.mu.RUnlock()
	return ol.head
}

// GetSize returns current log size
func (ol *OpLog) GetSize() int {
	ol.mu.RLock()
	defer ol.mu.RUnlock()
	return len(ol.entries)
}

// GetRange retrieves operations in sequence number range
func (ol *OpLog) GetRange(from, to uint64) []ReplicaOp {
	ol.mu.RLock()
	defer ol.mu.RUnlock()

	result := make([]ReplicaOp, 0)
	for _, op := range ol.entries {
		if op.SeqNum >= from && op.SeqNum <= to {
			result = append(result, op)
		}
	}
	return result
}

// Clear removes all entries (for testing)
func (ol *OpLog) Clear() {
	ol.mu.Lock()
	defer ol.mu.Unlock()
	ol.entries = make([]ReplicaOp, 0, ol.maxSize)
	ol.head = 0
}

// Trim removes operations older than given sequence number
func (ol *OpLog) Trim(seqNum uint64) int {
	ol.mu.Lock()
	defer ol.mu.Unlock()

	newEntries := make([]ReplicaOp, 0, len(ol.entries))
	for _, op := range ol.entries {
		if op.SeqNum >= seqNum {
			newEntries = append(newEntries, op)
		}
	}

	removed := len(ol.entries) - len(newEntries)
	ol.entries = newEntries
	return removed
}
