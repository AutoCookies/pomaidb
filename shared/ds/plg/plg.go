package plg

import (
	"hash/fnv"
	"sync"
)

const shardCount = 64

type LựuGraph struct {
	shards [shardCount]*graphShard
}

type graphShard struct {
	nodes map[string]map[string]float64
	mu    sync.RWMutex
}

func NewLựuGraph() *LựuGraph {
	lg := &LựuGraph{}
	for i := 0; i < shardCount; i++ {
		lg.shards[i] = &graphShard{
			nodes: make(map[string]map[string]float64),
		}
	}
	return lg
}

func (lg *LựuGraph) AddEdge(u, v string, w float64) {
	lg.addDirectedEdge(u, v, w)
	lg.addDirectedEdge(v, u, w)
}

func (lg *LựuGraph) addDirectedEdge(from, to string, w float64) {
	shard := lg.getShard(from)
	shard.mu.Lock()
	defer shard.mu.Unlock()

	if _, ok := shard.nodes[from]; !ok {
		shard.nodes[from] = make(map[string]float64)
	}
	shard.nodes[from][to] = w
}

func (lg *LựuGraph) ExtractCluster(startNode string, minDensity float64) []string {
	shard := lg.getShard(startNode)

	shard.mu.RLock()
	defer shard.mu.RUnlock()

	neighbors, ok := shard.nodes[startNode]
	if !ok {
		return []string{startNode}
	}

	cluster := make([]string, 0, len(neighbors)+1)
	cluster = append(cluster, startNode)

	for node, weight := range neighbors {
		if weight >= minDensity {
			cluster = append(cluster, node)
		}
	}

	return cluster
}

func (lg *LựuGraph) Clear() {
	for i := 0; i < shardCount; i++ {
		shard := lg.shards[i]
		shard.mu.Lock()
		shard.nodes = make(map[string]map[string]float64)
		shard.mu.Unlock()
	}
}

func (lg *LựuGraph) getShard(key string) *graphShard {
	h := fnv.New32a()
	h.Write([]byte(key))
	return lg.shards[h.Sum32()%shardCount]
}
