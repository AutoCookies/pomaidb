// File: internal/engine/replication/peer.go
package replication

import (
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"net"
	"time"
)

// newPeer creates new peer instance
func newPeer(id, addr string, conn net.Conn, isMock bool) *Peer {
	return &Peer{
		ID:          id,
		Addr:        addr,
		conn:        conn,
		encoder:     gob.NewEncoder(conn),
		decoder:     gob.NewDecoder(conn),
		status:      PeerHealthy,
		lastContact: time.Now(),
		lastSeqAck:  0,
		isMock:      isMock,
	}
}

// connectToPeer establishes connection to peer
func connectToPeer(addr string, timeout time.Duration) (net.Conn, error) {
	return net.DialTimeout("tcp", addr, timeout)
}

// sendOpToPeer sends operation to peer and waits for acknowledgment
func (rm *Manager) sendOpToPeer(peer *Peer, op ReplicaOp) error {
	peer.mu.Lock()
	defer peer.mu.Unlock()

	if peer.status == PeerDown {
		return ErrPeerDisconnected
	}

	start := time.Now()

	// Send operation
	if err := peer.encoder.Encode(op); err != nil {
		peer.status = PeerDegraded
		return fmt.Errorf("encode error: %w", err)
	}

	// Wait for acknowledgment
	var ack AckMessage
	peer.conn.SetReadDeadline(time.Now().Add(50 * time.Millisecond))

	if err := peer.decoder.Decode(&ack); err != nil {
		if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
			peer.status = PeerDegraded
		}
		return fmt.Errorf("decode ack error: %w", err)
	}

	// Update peer state
	peer.lastSeqAck = ack.SeqNum
	peer.lastContact = time.Now()
	peer.lag = time.Since(start).Milliseconds()

	if peer.status != PeerHealthy {
		peer.status = PeerHealthy
	}

	// Check for error in ack
	if ack.Status != "ok" && ack.Error != "" {
		return fmt.Errorf("peer error: %s", ack.Error)
	}

	return nil
}

// handlePeer handles incoming replication from peer (follower mode)
func (rm *Manager) handlePeer(peer *Peer) {
	defer func() {
		peer.conn.Close()
		rm.RemovePeer(peer.ID)
	}()

	log.Printf("[REPLICATION] Started handling peer: %s", peer.ID)

	for {
		select {
		case <-rm.ctx.Done():
			return
		default:
		}

		var op ReplicaOp
		peer.conn.SetReadDeadline(time.Now().Add(30 * time.Second))

		// Receive operation
		if err := peer.decoder.Decode(&op); err != nil {
			if err == io.EOF {
				log.Printf("[REPLICATION] Peer %s disconnected", peer.ID)
				return
			}
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue
			}
			log.Printf("[REPLICATION] Decode error from peer %s: %v", peer.ID, err)
			return
		}

		// Apply operation
		ack := AckMessage{
			SeqNum: op.SeqNum,
			Status: "ok",
		}

		if err := rm.applyOp(op); err != nil {
			log.Printf("[REPLICATION] Failed to apply op from peer %s: %v", peer.ID, err)
			ack.Status = "error"
			ack.Error = err.Error()
		}

		// Send acknowledgment
		if err := peer.encoder.Encode(ack); err != nil {
			log.Printf("[REPLICATION] Failed to send ack to peer %s: %v", peer.ID, err)
			return
		}
	}
}

// applyOp applies replicated operation to local store
func (rm *Manager) applyOp(op ReplicaOp) error {
	store := rm.tenants.GetStore(op.TenantID)

	switch op.Type {
	case OpTypeSet:
		return store.Put(op.Key, op.Value, op.TTL)

	case OpTypeDelete:
		store.Delete(op.Key)
		return nil

	case OpTypeIncr:
		_, err := store.Incr(op.Key, op.Delta)
		return err

	default:
		return fmt.Errorf("unknown op type: %d", op.Type)
	}
}

// checkPeerHealth checks health of all peers
func (rm *Manager) checkPeerHealth() {
	now := time.Now()

	rm.peersMu.Lock()
	defer rm.peersMu.Unlock()

	for _, peer := range rm.peers {
		peer.mu.RLock()
		timeSinceContact := now.Sub(peer.lastContact)
		lag := peer.lag
		peer.mu.RUnlock()

		// Check if peer is down
		if timeSinceContact > 30*time.Second {
			peer.mu.Lock()
			if peer.status != PeerDown {
				peer.status = PeerDown
				log.Printf("[REPLICATION] Peer %s marked as DOWN (no contact for %v)", peer.ID, timeSinceContact)
			}
			peer.mu.Unlock()
			continue
		}

		// Check if peer is degraded
		if lag > rm.maxLagMillis {
			peer.mu.Lock()
			if peer.status != PeerDegraded {
				peer.status = PeerDegraded
				log.Printf("[REPLICATION] Peer %s marked as DEGRADED (lag:  %dms)", peer.ID, lag)
			}
			peer.mu.Unlock()
			continue
		}

		// Peer is healthy
		peer.mu.Lock()
		if peer.status != PeerHealthy {
			peer.status = PeerHealthy
			log.Printf("[REPLICATION] Peer %s marked as HEALTHY", peer.ID)
		}
		peer.mu.Unlock()
	}
}
