package data_types

import (
	"github.com/AutoCookies/pomai-cache/shared/ds/plg"
)

type PLGStore struct {
	graph *plg.LựuGraph
}

func NewPLGStore() *PLGStore {
	return &PLGStore{
		graph: plg.NewLựuGraph(),
	}
}

func (p *PLGStore) AddEdge(node1, node2 string, weight float64) {
	p.graph.AddEdge(node1, node2, weight)
}

func (p *PLGStore) ExtractCluster(startNode string, minDensity float64) []string {
	return p.graph.ExtractCluster(startNode, minDensity)
}

func (p *PLGStore) Clear() {
	p.graph = plg.NewLựuGraph()
}
