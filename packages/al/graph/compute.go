package graph

import (
	"math"

	ds "github.com/AutoCookies/pomai-cache/packages/ds/graph"
)

// PageRank chạy trên cấu trúc CSR tĩnh
func PageRank(g *ds.CSRGraph, iterations int, damping float64, tolerance float64) []float64 {
	N := g.NumNodes
	if N == 0 {
		return nil
	}

	scores := make([]float64, N)
	newScores := make([]float64, N)
	initialScore := 1.0 / float64(N)

	// Init
	for i := 0; i < N; i++ {
		scores[i] = initialScore
	}

	baseScore := (1.0 - damping) / float64(N)

	for iter := 0; iter < iterations; iter++ {
		for i := 0; i < N; i++ {
			newScores[i] = 0
		}

		for u := 0; u < N; u++ {
			start := g.Indptr[u]
			end := g.Indptr[u+1]

			if start == end {
				continue
			}

			outDegree := float64(end - start)
			share := (scores[u] * damping) / outDegree

			// Sequential Memory Access -> 10/10 Prefetching
			for i := start; i < end; i++ {
				v := g.Indices[i]
				newScores[v] += share
			}
		}

		diff := 0.0
		for i := 0; i < N; i++ {
			newScores[i] += baseScore
			diff += math.Abs(newScores[i] - scores[i])
		}

		copy(scores, newScores)

		if diff < tolerance {
			break
		}
	}

	return scores
}

// ShortestPathBFS tìm đường ngắn nhất trên CSR
func ShortestPathBFS(g *ds.CSRGraph, startNode, endNode int32, maxDepth int) []int32 {
	if startNode == endNode {
		return []int32{startNode}
	}

	parent := make([]int32, g.NumNodes)
	dist := make([]int32, g.NumNodes)
	for i := range parent {
		parent[i] = -1
		dist[i] = -1
	}

	queue := make([]int32, 0, 1024)
	queue = append(queue, startNode)
	dist[startNode] = 0
	parent[startNode] = startNode

	found := false

	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]

		if u == endNode {
			found = true
			break
		}

		if dist[u] >= int32(maxDepth) {
			continue
		}

		start := g.Indptr[u]
		end := g.Indptr[u+1]
		for i := start; i < end; i++ {
			v := g.Indices[i]
			if dist[v] == -1 {
				dist[v] = dist[u] + 1
				parent[v] = u
				queue = append(queue, v)
			}
		}
	}

	if !found {
		return nil
	}

	// Reconstruct path
	path := make([]int32, 0, dist[endNode]+1)
	curr := endNode
	for curr != startNode {
		path = append(path, curr)
		curr = parent[curr]
	}
	path = append(path, startNode)

	// Reverse
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}
