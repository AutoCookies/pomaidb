package replication

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

const (
	gossipInterval  = 1 * time.Second
	failTimeout     = 15 * time.Second
	maxGossipPacket = 1024
	burstProb       = 0.1
)

type NodeState int

const (
	NodeStateAlive NodeState = iota
	NodeStateSuspect
	NodeStateDead
)

type NodeInfo struct {
	ID        string    `json:"id"`
	TCPAddr   string    `json:"tcp"`
	UDPAddr   string    `json:"udp"`
	State     NodeState `json:"-"`
	LastSeen  int64     `json:"-"`
	ShardMask uint32    `json:"mask"`
}

type GossipMessage struct {
	SenderID string     `json:"sid"`
	Sender   NodeInfo   `json:"sinfo"`
	Members  []NodeInfo `json:"mem"`
}

type Manager struct {
	nodeID  string
	mode    ReplicationMode
	tenants TenantManagerInterface

	peers     map[string]*Peer
	peersList []string
	peersMu   sync.RWMutex

	clusterNodes map[string]*NodeInfo
	clusterMu    sync.RWMutex
	udpConn      *net.UDPConn
	gossipPort   string

	isLeader     atomic.Bool
	opLog        *OpLog
	ctx          context.Context
	cancel       context.CancelFunc
	writeQuorum  int
	maxLagMillis int64
	stats        struct {
		TotalOps         atomic.Uint64
		ReplicatedOps    atomic.Uint64
		FailedReplicas   atomic.Uint64
		AverageLatencyMs atomic.Int64
		CurrentLag       atomic.Int64
		BurstOps         atomic.Uint64
	}
}

type PomaiAgent struct {
	shardID   int
	state     string
	neighbors []string
	mu        sync.RWMutex
}

func NewManager(nodeID string, mode ReplicationMode, tenants TenantManagerInterface) *Manager {
	ctx, cancel := context.WithCancel(context.Background())

	rm := &Manager{
		nodeID:       nodeID,
		mode:         mode,
		tenants:      tenants,
		peers:        make(map[string]*Peer),
		peersList:    make([]string, 0),
		clusterNodes: make(map[string]*NodeInfo),
		opLog:        NewOpLog(100000),
		ctx:          ctx,
		cancel:       cancel,
		writeQuorum:  1,
		maxLagMillis: 5000,
	}

	rm.isLeader.Store(true)

	go rm.healthCheckLoop()
	go rm.metricsLoop()

	log.Printf("[REPLICATION] Manager started: node=%s, mode=%s", nodeID, mode)
	return rm
}

func (rm *Manager) StartAgents(shardCount int) {
	for i := 0; i < shardCount; i++ {
		agent := &PomaiAgent{shardID: i, state: "exploring"}
		go agent.run(rm)
	}
	log.Printf("[PMAC] Started %d Pomai Agents", shardCount)
}

func (a *PomaiAgent) run(m *Manager) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			a.explore(m)
			a.cluster(m)
			a.optimize(m)
		}
	}
}

func (a *PomaiAgent) explore(m *Manager) {
	m.peersMu.RLock()
	defer m.peersMu.RUnlock()
	a.mu.Lock()
	defer a.mu.Unlock()

	a.neighbors = make([]string, 0)
	for id, peer := range m.peers {
		if peer.GetLag() < 50 && peer.GetStatus() == PeerHealthy {
			a.neighbors = append(a.neighbors, id)
		}
	}
}

func (a *PomaiAgent) cluster(m *Manager) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.neighbors) >= 2 {
		a.state = "optimizing"
	} else {
		a.state = "exploring"
	}
}

func (a *PomaiAgent) optimize(m *Manager) {
	a.mu.RLock()
	state := a.state
	a.mu.RUnlock()

	if state == "optimizing" {
		// Optimization logic
	}
}

func (rm *Manager) BurstReplicate(tenantID, key string, value []byte, ttl time.Duration) {
	if rand.Float64() > burstProb {
		return
	}

	targets := rm.getRandomPeers(2)
	if len(targets) == 0 {
		return
	}

	op := ReplicaOp{
		Type:      OpTypeSet,
		Key:       key,
		Value:     value,
		TTL:       ttl,
		TenantID:  tenantID,
		Timestamp: time.Now().UnixNano(),
	}

	rm.stats.BurstOps.Add(1)

	for _, peer := range targets {
		go func(p *Peer) {
			_ = rm.sendOpToPeer(p, op)
		}(peer)
	}
}

func (rm *Manager) getRandomPeers(count int) []*Peer {
	rm.peersMu.RLock()
	defer rm.peersMu.RUnlock()

	n := len(rm.peersList)
	if n == 0 {
		return nil
	}

	result := make([]*Peer, 0, count)
	limit := count
	if n < limit {
		limit = n
	}

	indices := rand.Perm(n)
	for i := 0; i < limit; i++ {
		id := rm.peersList[indices[i]]
		if peer, ok := rm.peers[id]; ok && peer.GetStatus() == PeerHealthy {
			result = append(result, peer)
		}
	}
	return result
}

func (rm *Manager) EnableGossip(tcpPort, udpPort string, seeds []string) error {
	rm.gossipPort = udpPort

	addr, err := net.ResolveUDPAddr("udp", ":"+udpPort)
	if err != nil {
		return err
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		return err
	}
	rm.udpConn = conn

	myIP := getOutboundIP()

	me := &NodeInfo{
		ID:       rm.nodeID,
		TCPAddr:  fmt.Sprintf("%s:%s", myIP, tcpPort),
		UDPAddr:  fmt.Sprintf("%s:%s", myIP, udpPort),
		State:    NodeStateAlive,
		LastSeen: time.Now().Unix(),
	}
	rm.clusterMu.Lock()
	rm.clusterNodes[rm.nodeID] = me
	rm.clusterMu.Unlock()

	go rm.gossipListenLoop()
	go rm.gossipBroadcastLoop()
	go rm.gossipFailureLoop()

	for _, seed := range seeds {
		if seed != "" {
			go rm.PingDirect(seed)
		}
	}

	log.Printf("[CLUSTER] Gossip enabled on %s (Seeds: %d)", udpPort, len(seeds))
	return nil
}

func (rm *Manager) Stop() {
	rm.Shutdown()
	rm.StopGossip()
}

func (rm *Manager) StopGossip() {
	if rm.udpConn != nil {
		rm.udpConn.Close()
	}
}

func (rm *Manager) gossipListenLoop() {
	buf := make([]byte, maxGossipPacket)
	for {
		select {
		case <-rm.ctx.Done():
			return
		default:
			if rm.udpConn == nil {
				return
			}
			rm.udpConn.SetReadDeadline(time.Now().Add(2 * time.Second))
			n, _, err := rm.udpConn.ReadFromUDP(buf)
			if err != nil {
				continue
			}
			rm.handleGossipPacket(buf[:n])
		}
	}
}

func (rm *Manager) handleGossipPacket(data []byte) {
	var msg GossipMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}

	rm.clusterMu.Lock()
	defer rm.clusterMu.Unlock()

	rm.mergeNodeState(msg.Sender)
	for _, node := range msg.Members {
		rm.mergeNodeState(node)
	}
}

func (rm *Manager) mergeNodeState(node NodeInfo) {
	if node.ID == rm.nodeID {
		return
	}

	existing, exists := rm.clusterNodes[node.ID]

	if !exists {
		log.Printf("[CLUSTER] Discovery: Node %s at %s", node.ID, node.TCPAddr)
		newNode := node
		newNode.State = NodeStateAlive
		newNode.LastSeen = time.Now().Unix()
		rm.clusterNodes[node.ID] = &newNode
		go rm.AddPeer(node.ID, node.TCPAddr)
	} else {
		existing.LastSeen = time.Now().Unix()
		if existing.State != NodeStateAlive {
			log.Printf("[CLUSTER] Node %s recovered", node.ID)
			existing.State = NodeStateAlive
		}
		if existing.TCPAddr != node.TCPAddr {
			existing.TCPAddr = node.TCPAddr
		}
	}
}

func (rm *Manager) broadcastGossip() {
	rm.clusterMu.RLock()
	nodes := make([]NodeInfo, 0, len(rm.clusterNodes))
	targets := make([]string, 0)

	me := *rm.clusterNodes[rm.nodeID]

	for _, n := range rm.clusterNodes {
		if n.State == NodeStateAlive && n.ID != rm.nodeID {
			nodes = append(nodes, *n)
			targets = append(targets, n.UDPAddr)
		}
	}
	rm.clusterMu.RUnlock()

	subsetSize := 3
	if len(nodes) > subsetSize {
		rand.Shuffle(len(nodes), func(i, j int) { nodes[i], nodes[j] = nodes[j], nodes[i] })
		nodes = nodes[:subsetSize]
	}

	msg := GossipMessage{
		SenderID: rm.nodeID,
		Sender:   me,
		Members:  nodes,
	}
	data, _ := json.Marshal(msg)

	k := 3
	if len(targets) > 0 {
		rand.Shuffle(len(targets), func(i, j int) { targets[i], targets[j] = targets[j], targets[i] })
		if len(targets) < k {
			k = len(targets)
		}

		for i := 0; i < k; i++ {
			rm.sendUDP(targets[i], data)
		}
	}
}

func (rm *Manager) gossipBroadcastLoop() {
	ticker := time.NewTicker(gossipInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.broadcastGossip()
		}
	}
}

func (rm *Manager) gossipFailureLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.reapDeadNodes()
		}
	}
}

func (rm *Manager) reapDeadNodes() {
	rm.clusterMu.Lock()
	defer rm.clusterMu.Unlock()

	now := time.Now().Unix()
	timeout := int64(failTimeout.Seconds())

	for id, node := range rm.clusterNodes {
		if id == rm.nodeID {
			continue
		}

		if now-node.LastSeen > timeout {
			if node.State == NodeStateAlive {
				log.Printf("[CLUSTER] ⚠️ Node %s detected DEAD (timeout)", id)
				node.State = NodeStateDead
				go rm.RemovePeer(id)
			}
		}
	}
}

func (rm *Manager) PingDirect(addr string) {
	rm.clusterMu.RLock()
	me, ok := rm.clusterNodes[rm.nodeID]
	rm.clusterMu.RUnlock()
	if !ok {
		return
	}

	msg := GossipMessage{
		SenderID: rm.nodeID,
		Sender:   *me,
	}
	data, _ := json.Marshal(msg)
	rm.sendUDP(addr, data)
}

func (rm *Manager) sendUDP(addr string, data []byte) {
	if rm.udpConn == nil {
		return
	}
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return
	}
	rm.udpConn.WriteToUDP(data, udpAddr)
}

func getOutboundIP() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()
	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP.String()
}

func (rm *Manager) Replicate(op ReplicaOp) error {
	if !rm.isLeader.Load() {
		return ErrNotLeader
	}

	op.Timestamp = time.Now().UnixNano()
	seqNum := rm.opLog.Append(op)
	op.SeqNum = seqNum

	rm.stats.TotalOps.Add(1)

	switch rm.mode {
	case ModeAsync:
		go rm.replicateAsync(op)
		return nil
	case ModeSync:
		return rm.replicateSync(op)
	case ModeSemiSync:
		return rm.replicateSemiSync(op)
	default:
		return fmt.Errorf("unknown replication mode: %d", rm.mode)
	}
}

func (rm *Manager) replicateAsync(op ReplicaOp) {
	peers := rm.getHealthyPeers()
	for _, peer := range peers {
		go rm.sendOpToPeer(peer, op)
	}
}

func (rm *Manager) replicateSync(op ReplicaOp) error {
	peers := rm.getHealthyPeers()
	if len(peers) == 0 {
		return ErrNoHealthyPeers
	}

	errCh := make(chan error, len(peers))
	for _, peer := range peers {
		go func(p *Peer) {
			errCh <- rm.sendOpToPeer(p, op)
		}(peer)
	}

	successCount := 0
	for i := 0; i < len(peers); i++ {
		if err := <-errCh; err == nil {
			successCount++
		}
	}

	if successCount == len(peers) {
		rm.stats.ReplicatedOps.Add(1)
		return nil
	}
	rm.stats.FailedReplicas.Add(1)
	return fmt.Errorf("sync failed: %d/%d success", successCount, len(peers))
}

func (rm *Manager) replicateSemiSync(op ReplicaOp) error {
	peers := rm.getHealthyPeers()
	if len(peers) == 0 {
		return ErrNoHealthyPeers
	}

	quorum := rm.calculateQuorum(len(peers))
	errCh := make(chan error, len(peers))
	for _, peer := range peers {
		go func(p *Peer) {
			errCh <- rm.sendOpToPeer(p, op)
		}(peer)
	}

	successCount := 0
	timeout := time.After(100 * time.Millisecond)

	for i := 0; i < len(peers); i++ {
		select {
		case err := <-errCh:
			if err == nil {
				successCount++
				if successCount >= quorum {
					rm.stats.ReplicatedOps.Add(1)
					go func() {
						for j := i + 1; j < len(peers); j++ {
							<-errCh
						}
					}()
					return nil
				}
			}
		case <-timeout:
			if successCount >= quorum {
				rm.stats.ReplicatedOps.Add(1)
				return nil
			}
			rm.stats.FailedReplicas.Add(1)
			return fmt.Errorf("semi-sync timeout")
		}
	}
	if successCount >= quorum {
		rm.stats.ReplicatedOps.Add(1)
		return nil
	}
	rm.stats.FailedReplicas.Add(1)
	return fmt.Errorf("semi-sync failed")
}

func (rm *Manager) calculateQuorum(peerCount int) int {
	if rm.writeQuorum > 0 {
		return rm.writeQuorum
	}
	return (peerCount / 2) + 1
}

func (rm *Manager) getHealthyPeers() []*Peer {
	rm.peersMu.RLock()
	defer rm.peersMu.RUnlock()
	peers := make([]*Peer, 0, len(rm.peers))
	for _, peer := range rm.peers {
		if peer.GetStatus() == PeerHealthy {
			peers = append(peers, peer)
		}
	}
	return peers
}

func (rm *Manager) AddPeer(peerID, addr string) error {
	rm.peersMu.RLock()
	_, exists := rm.peers[peerID]
	rm.peersMu.RUnlock()
	if exists {
		return nil
	}

	conn, err := connectToPeer(addr, 5*time.Second)
	if err != nil {
		return fmt.Errorf("connect failed: %w", err)
	}

	peer := newPeer(peerID, addr, conn, false)
	rm.peersMu.Lock()
	rm.peers[peerID] = peer
	rm.peersList = append(rm.peersList, peerID)
	rm.peersMu.Unlock()

	go rm.handlePeer(peer)
	log.Printf("[REPLICATION] Peer added: %s", peerID)
	return nil
}

func (rm *Manager) RemovePeer(peerID string) {
	rm.peersMu.Lock()
	defer rm.peersMu.Unlock()

	peer, ok := rm.peers[peerID]
	if ok {
		peer.conn.Close()
		delete(rm.peers, peerID)

		for i, id := range rm.peersList {
			if id == peerID {
				lastIdx := len(rm.peersList) - 1
				rm.peersList[i] = rm.peersList[lastIdx]
				rm.peersList = rm.peersList[:lastIdx]
				break
			}
		}
	}
}

func (rm *Manager) Shutdown() {
	log.Println("[REPLICATION] Shutting down...")
	rm.cancel()
	rm.peersMu.Lock()
	for _, peer := range rm.peers {
		peer.conn.Close()
	}
	rm.peersMu.Unlock()
}

func (rm *Manager) healthCheckLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.checkPeerHealth()
		}
	}
}

func (rm *Manager) metricsLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.logMetrics()
		}
	}
}

func (rm *Manager) logMetrics() {
	stats := rm.GetStats()
	rm.clusterMu.RLock()
	nodeCount := len(rm.clusterNodes)
	rm.clusterMu.RUnlock()
	log.Printf("[REPLICATION] Nodes: %d | Ops: %d | Bursts: %d | Healthy: %d",
		nodeCount, stats.TotalOps, rm.stats.BurstOps.Load(), stats.HealthyPeers)
}

func (rm *Manager) GetStats() ReplicationStats {
	rm.peersMu.RLock()
	active := len(rm.peers)
	healthy := 0
	for _, p := range rm.peers {
		if p.GetStatus() == PeerHealthy {
			healthy++
		}
	}
	rm.peersMu.RUnlock()

	return ReplicationStats{
		TotalOps:       rm.stats.TotalOps.Load(),
		ReplicatedOps:  rm.stats.ReplicatedOps.Load(),
		FailedReplicas: rm.stats.FailedReplicas.Load(),
		ActivePeers:    active,
		HealthyPeers:   healthy,
		CurrentLag:     rm.stats.CurrentLag.Load(),
	}
}
