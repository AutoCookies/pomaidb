package pomegranate

import (
	"sync"
)

const (
	shards       = 64
	mask         = shards - 1
	maxNeighbors = 4
)

type Membrane struct {
	shards [shards]graphShard
}

type graphShard struct {
	links      map[string][]string
	lastAccess string
	mu         sync.RWMutex
	pad        [64]byte
}

func NewMembrane() *Membrane {
	m := &Membrane{}
	for i := 0; i < shards; i++ {
		m.shards[i].links = make(map[string][]string)
	}
	return m
}

func hash(key string) uint32 {
	var h uint32 = 2166136261
	for i := 0; i < len(key); i++ {
		h ^= uint32(key[i])
		h *= 16777619
	}
	return h
}

func (m *Membrane) Pulse(key string) {
	idx := hash(key) & mask
	s := &m.shards[idx]

	s.mu.Lock()
	last := s.lastAccess
	s.lastAccess = key

	if last != "" && last != key {
		neighbors := s.links[last]
		exists := false
		for _, n := range neighbors {
			if n == key {
				exists = true
				break
			}
		}
		if !exists {
			if len(neighbors) >= maxNeighbors {
				copy(neighbors, neighbors[1:])
				neighbors[maxNeighbors-1] = key
				s.links[last] = neighbors
			} else {
				s.links[last] = append(neighbors, key)
			}
		}
	}
	s.mu.Unlock()
}

func (m *Membrane) GetClusterStrength(key string) float64 {
	idx := hash(key) & mask
	s := &m.shards[idx]

	s.mu.RLock()
	degree := len(s.links[key])
	s.mu.RUnlock()

	if degree == 0 {
		return 0.1
	}
	return 1.0 + (float64(degree) * 0.25)
}

func (m *Membrane) Prune(key string) {
	idx := hash(key) & mask
	s := &m.shards[idx]

	s.mu.Lock()
	delete(s.links, key)
	if s.lastAccess == key {
		s.lastAccess = ""
	}
	s.mu.Unlock()
}
