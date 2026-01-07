// File: internal/engine/replication/types.go
package replication

import (
	"encoding/gob"
	"errors"
	"net"
	"sync"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core"
)

// Errors
var (
	ErrNotLeader        = errors.New("not leader")
	ErrNoHealthyPeers   = errors.New("no healthy peers available")
	ErrReplicationLag   = errors.New("replication lag exceeded threshold")
	ErrPeerDisconnected = errors.New("peer disconnected")
	ErrPeerNotFound     = errors.New("peer not found")
)

// ReplicationMode defines replication strategy
type ReplicationMode int

const (
	ModeAsync    ReplicationMode = iota // Fire and forget
	ModeSync                            // Wait for all peers
	ModeSemiSync                        // Wait for quorum
)

func (m ReplicationMode) String() string {
	switch m {
	case ModeAsync:
		return "async"
	case ModeSync:
		return "sync"
	case ModeSemiSync:
		return "semi-sync"
	default:
		return "unknown"
	}
}

// OpType represents operation type
type OpType uint8

const (
	OpTypeSet OpType = iota + 1
	OpTypeDelete
	OpTypeIncr
	OpTypeExpire // For future TTL operations
)

func (t OpType) String() string {
	switch t {
	case OpTypeSet:
		return "SET"
	case OpTypeDelete:
		return "DELETE"
	case OpTypeIncr:
		return "INCR"
	case OpTypeExpire:
		return "EXPIRE"
	default:
		return "UNKNOWN"
	}
}

// ReplicaOp represents a replicated operation
type ReplicaOp struct {
	Type      OpType
	Key       string
	Value     []byte
	Delta     int64
	TTL       time.Duration
	Timestamp int64
	SeqNum    uint64
	TenantID  string
}

// PeerStatus represents peer health status
type PeerStatus int

const (
	PeerHealthy  PeerStatus = iota // Fully operational
	PeerDegraded                   // High latency/errors
	PeerDown                       // Unreachable
)

func (s PeerStatus) String() string {
	switch s {
	case PeerHealthy:
		return "healthy"
	case PeerDegraded:
		return "degraded"
	case PeerDown:
		return "down"
	default:
		return "unknown"
	}
}

// Peer represents a replication peer
type Peer struct {
	ID          string
	Addr        string
	conn        net.Conn
	encoder     *gob.Encoder
	decoder     *gob.Decoder
	mu          sync.RWMutex
	status      PeerStatus
	lastSeqAck  uint64
	lastContact time.Time
	lag         int64 // Milliseconds
	isMock      bool  // For testing
}

// GetStatus returns peer status (thread-safe)
func (p *Peer) GetStatus() PeerStatus {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.status
}

// SetStatus sets peer status (thread-safe)
func (p *Peer) SetStatus(status PeerStatus) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.status = status
}

// GetLag returns replication lag in milliseconds
func (p *Peer) GetLag() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lag
}

// GetLastContact returns last contact time
func (p *Peer) GetLastContact() time.Time {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.lastContact
}

// ReplicationStats tracks replication metrics
type ReplicationStats struct {
	TotalOps         uint64
	ReplicatedOps    uint64
	FailedReplicas   uint64
	AverageLatencyMs int64
	CurrentLag       int64
	ActivePeers      int
	HealthyPeers     int
	DegradedPeers    int
	DownPeers        int
}

// TenantManagerInterface defines tenant operations
// FIX: Trả về *core.Store cụ thể để khớp với implementation của tenants.Manager
type TenantManagerInterface interface {
	GetStore(tenantID string) *core.Store
}

// AckMessage represents acknowledgment from peer
type AckMessage struct {
	SeqNum uint64
	Status string
	Error  string
}
