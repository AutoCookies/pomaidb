package ttl

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultInterval = 100 * time.Millisecond
	clusterSize     = 16
	maxTimeBudget   = 10 * time.Millisecond
	sampleBatch     = 5
	predictBias     = 10 * time.Second
)

var (
	clusterPool = sync.Pool{
		New: func() interface{} { return make([]string, 0, clusterSize) },
	}
	visitedPool = sync.Pool{
		New: func() interface{} { return make(map[string]struct{}, clusterSize) },
	}
)

type PPPCleaner struct {
	manager  *Manager
	store    StoreInterface
	interval time.Duration
	running  atomic.Bool
}

func NewPPPCleaner(manager *Manager, store StoreInterface) *PPPCleaner {
	return &PPPCleaner{
		manager:  manager,
		store:    store,
		interval: defaultInterval,
	}
}

func (c *PPPCleaner) Start(ctx context.Context) {
	if !c.running.CompareAndSwap(false, true) {
		return
	}
	go c.loop(ctx)
}

func (c *PPPCleaner) loop(ctx context.Context) {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			c.running.Store(false)
			return
		case <-ticker.C:
			c.predictivePeel()
		}
	}
}

func (c *PPPCleaner) predictivePeel() {
	shards := c.store.GetShards()
	start := time.Now()
	now := start.UnixNano()

	for _, shard := range shards {
		if time.Since(start) > maxTimeBudget {
			break
		}

		seed := c.sampleSmartSeed(shard, now)
		if seed == "" {
			continue
		}

		cluster, expiredCandidates := c.expandCluster(shard, seed, now)

		if len(expiredCandidates) > 0 {
			c.deleteBatch(shard, expiredCandidates)
		} else {
			// proactive predictive prune
			items := shard.GetItems()
			avgPred := c.avgClusterPredict(cluster, items)
			if avgPred > 0 && avgPred < now+int64(predictBias) {
				cands := make([]string, 0, len(cluster))
				for _, k := range cluster {
					if elem, ok := items[k]; ok {
						if entry := extractEntry(elem); entry != nil {
							if entry.PredictNext() < now+int64(predictBias) {
								cands = append(cands, k)
							}
						}
					}
				}
				if len(cands) > 0 {
					c.deleteBatch(shard, cands)
				}
			}
		}

		c.recycleBuffers(cluster, nil)
	}
}

func (c *PPPCleaner) sampleSmartSeed(shard ShardInterface, now int64) string {
	shard.RLock()
	defer shard.RUnlock()

	items := shard.GetItems()
	if len(items) == 0 {
		return ""
	}

	keys := make([]string, 0, sampleBatch*2)
	count := 0
	for k := range items {
		keys = append(keys, k)
		count++
		if count >= sampleBatch*2 {
			break
		}
	}
	if len(keys) == 0 {
		return ""
	}

	n := len(keys)
	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	preds := make([]int64, n)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		i0 := w * chunk
		i1 := i0 + chunk
		if i1 > n {
			i1 = n
		}
		go func(a, b int) {
			for i := a; i < b; i++ {
				elem := items[keys[i]]
				entry := extractEntry(elem)
				if entry == nil {
					preds[i] = 1<<63 - 1
				} else {
					if exp := entry.ExpireAt(); exp > 0 && now > exp {
						preds[i] = now - 1
					} else {
						preds[i] = entry.PredictNext()
					}
				}
			}
			wg.Done()
		}(i0, i1)
	}
	wg.Wait()

	bestIdx := -1
	minPred := int64(1<<63 - 1)
	for i := 0; i < n && (i < sampleBatch); i++ {
		// select top sampleBatch minima across preds
		// naive selection: scan all and pick smallest unseen
		for j := 0; j < n; j++ {
			if preds[j] < minPred {
				minPred = preds[j]
				bestIdx = j
			}
		}
	}
	if bestIdx >= 0 && bestIdx < len(keys) {
		return keys[bestIdx]
	}
	// fallback: random-ish first
	return keys[0]
}

func (c *PPPCleaner) expandCluster(shard ShardInterface, seedKey string, now int64) ([]string, []string) {
	shard.RLock()
	defer shard.RUnlock()

	items := shard.GetItems()

	cluster := clusterPool.Get().([]string)[:0]
	visited := visitedPool.Get().(map[string]struct{})

	for k := range visited {
		delete(visited, k)
	}

	queue := make([]string, 0, clusterSize)
	queue = append(queue, seedKey)
	visited[seedKey] = struct{}{}
	cluster = append(cluster, seedKey)

	expiredCandidates := make([]string, 0, clusterSize)

	for len(queue) > 0 && len(cluster) < clusterSize {
		curr := queue[0]
		queue = queue[1:]

		elem, ok := items[curr]
		if !ok {
			continue
		}
		entry := extractEntry(elem)
		if entry == nil {
			continue
		}

		if exp := entry.ExpireAt(); exp > 0 && now > exp {
			expiredCandidates = append(expiredCandidates, curr)
		}

		if entry.PredictNext() > now+int64(predictBias) {
			continue
		}

		hints := entry.GetHints()
		for _, h := range hints {
			if _, seen := visited[h]; !seen {
				visited[h] = struct{}{}
				if len(cluster) < clusterSize {
					cluster = append(cluster, h)
					queue = append(queue, h)
				}
			}
		}
	}

	visitedPool.Put(visited)

	return cluster, expiredCandidates
}

func (c *PPPCleaner) deleteBatch(shard ShardInterface, keys []string) {
	if len(keys) == 0 {
		return
	}

	shard.Lock()
	defer shard.Unlock()

	freed := int64(0)
	for _, key := range keys {
		size, ok := shard.DeleteItem(key)
		if ok {
			freed += int64(size)
		}
	}

	if freed > 0 {
		shard.AddBytes(-freed)
		c.store.AddTotalBytes(-freed)
		if mc := c.store.GetGlobalMemCtrl(); mc != nil {
			mc.Release(freed)
		}
	}
}

func (c *PPPCleaner) recycleBuffers(cluster []string, visited map[string]struct{}) {
	if cluster != nil {
		clusterPool.Put(cluster)
	}
	if visited != nil {
		visitedPool.Put(visited)
	}
}

func (c *PPPCleaner) avgClusterPredict(cluster []string, items map[string]interface{}) int64 {
	n := len(cluster)
	if n == 0 {
		return 0
	}
	preds := make([]float32, 0, n)
	for _, k := range cluster {
		if elem, ok := items[k]; ok {
			if e := extractEntry(elem); e != nil {
				preds = append(preds, float32(e.PredictNext()))
			} else {
				preds = append(preds, float32(1<<31-1))
			}
		} else {
			preds = append(preds, float32(1<<31-1))
		}
	}
	workers := runtime.NumCPU()
	total := parallelSumFloat32(preds, workers)
	avg := int64(total / float64(len(preds)))
	return avg
}
