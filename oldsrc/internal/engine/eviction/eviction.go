package eviction

import (
	"container/list"
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/engine/core/common"
	"github.com/AutoCookies/pomai-cache/shared/al/pomegranate"
	"github.com/AutoCookies/pomai-cache/shared/al/ppe"
	"github.com/AutoCookies/pomai-cache/shared/al/pqse"
)

type StoreInterface interface {
	GetShard(key string) ShardInterface
	GetShardByIndex(idx int) ShardInterface
	GetShardCount() int
	GetCapacityBytes() int64
	GetTotalBytes() int64
	AddTotalBytes(delta int64)
	GetTenantID() string
	GetFreqEstimator() FreqEstimator
	GetGlobalMemCtrl() MemoryController
	AddEviction()
	SetGlobalEfSearch(ef int)
	GetHits() uint64
	GetMisses() uint64
	GetAvgLatency() float64
	FreeEntry(entry *common.Entry)
}

type ShardInterface interface {
	Lock()
	Unlock()
	RLock()
	RUnlock()
	GetItems() map[string]interface{}
	GetLRUBack() interface{}
	// [UPDATE] Trả về *common.Entry thay vì size
	DeleteItem(key string) (*common.Entry, bool)
	GetBytes() int64
}

type FreqEstimator interface {
	Estimate(key string) uint32
	Increment(key string)
}

type MemoryController interface {
	Release(bytes int64)
}

var (
	evictionLogEnabled atomic.Bool
	lastLogTime        int64
)

func SetEvictionLogging(enabled bool) {
	evictionLogEnabled.Store(enabled)
}

func logEviction(format string, args ...interface{}) {
	if !evictionLogEnabled.Load() {
		return
	}
	now := time.Now().UnixNano()
	last := atomic.LoadInt64(&lastLogTime)
	if now-last < int64(time.Second) {
		return
	}
	if atomic.CompareAndSwapInt64(&lastLogTime, last, now) {
		log.Printf("[PPE-3.0] "+format, args...)
	}
}

type Manager struct {
	store      StoreInterface
	metrics    *EvictionMetrics
	predictors sync.Map
	membrane   *pomegranate.Membrane
	pie        *PIEBandit
	stopCh     chan struct{}
	lastHits   uint64
	lastMisses uint64
}

func NewManager(store StoreInterface) *Manager {
	m := &Manager{
		store:    store,
		metrics:  &EvictionMetrics{},
		membrane: pomegranate.NewMembrane(),
		pie:      NewPIEBandit(),
		stopCh:   make(chan struct{}),
	}
	go m.runPIELoop()
	return m
}

func (m *Manager) runPIELoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.optimizePIE()
		}
	}
}

func (m *Manager) optimizePIE() {
	currentHits := m.store.GetHits()
	currentMisses := m.store.GetMisses()

	deltaHits := float64(currentHits - m.lastHits)
	deltaMisses := float64(currentMisses - m.lastMisses)
	totalOps := deltaHits + deltaMisses

	m.lastHits = currentHits
	m.lastMisses = currentMisses

	if totalOps == 0 {
		return
	}

	hitRate := deltaHits / totalOps
	latency := m.store.GetAvgLatency()

	latencyPenalty := latency / 10.0
	reward := hitRate - latencyPenalty

	currentArm := m.pie.currentArm
	m.pie.UpdateReward(currentArm, reward)

	_, config := m.pie.SelectArm()
	m.store.SetGlobalEfSearch(config.EfSearch)
}

func (m *Manager) Stop() {
	close(m.stopCh)
}

func (m *Manager) RecordAccess(key string, now int64) {
	if key == "" {
		return
	}

	m.membrane.Pulse(key)

	if v, ok := m.predictors.Load(key); ok {
		v.(*ppe.Predictor).Update(now)
		return
	}

	p := ppe.NewPredictor()
	p.Update(now)
	m.predictors.Store(key, p)
}

func (m *Manager) BatchEvict(targetBytes int64) int64 {
	return m.runPPE3(targetBytes)
}

func (m *Manager) EmergencyEvict(targetBytes int64) error {
	if freed := m.runPPE3(targetBytes); freed >= targetBytes {
		return nil
	}
	return m.runPQSE(targetBytes)
}

func (m *Manager) EvictIfNeeded(threshold float64) error {
	capacityBytes := m.store.GetCapacityBytes()
	if capacityBytes <= 0 {
		return nil
	}
	currentBytes := m.store.GetTotalBytes()
	usageRatio := float64(currentBytes) / float64(capacityBytes)
	if usageRatio < threshold {
		return nil
	}
	if usageRatio >= 1.0 {
		return m.EmergencyEvict(currentBytes - capacityBytes)
	}
	return nil
}

func (m *Manager) runPPE3(targetBytes int64) int64 {
	start := time.Now()
	freed := int64(0)
	evicted := 0

	shardCount := m.store.GetShardCount()
	candidates := make([]candidate, 0, 128)
	now := start.UnixNano()

	maxShardsToScan := 8
	startShard := int(now % int64(shardCount))

	config := m.pie.GetCurrentConfig()
	sampleSize := config.EvictSample

	for i := 0; i < maxShardsToScan; i++ {
		idx := (startShard + i) % shardCount
		s := m.store.GetShardByIndex(idx)
		if s == nil {
			continue
		}

		s.RLock()
		elem := s.GetLRUBack()
		n := 0
		for elem != nil && n < sampleSize {
			ent := extractEntry(elem)
			if ent.Key != "" {
				score := m.calcPomegranateScore(ent.Key, now)
				candidates = append(candidates, candidate{
					key:      ent.Key,
					size:     ent.Size,
					score:    score,
					shardIdx: idx,
				})
			}
			elem = getPrevElement(elem)
			n++
		}
		s.RUnlock()
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	for _, c := range candidates {
		if freed >= targetBytes {
			break
		}

		s := m.store.GetShardByIndex(c.shardIdx)
		s.Lock()
		// [UPDATE] Nhận *common.Entry từ DeleteItem
		if entry, ok := s.DeleteItem(c.key); ok {
			size := entry.Size()
			freed += int64(size)
			evicted++
			m.cleanup(c.key, size, entry)
		}
		s.Unlock()
	}

	m.metrics.TotalEvictions.Add(uint64(evicted))
	m.metrics.BytesFreed.Add(freed)
	m.metrics.LastEvictionDuration.Store(time.Since(start).Milliseconds())
	return freed
}

func (m *Manager) calcPomegranateScore(key string, now int64) float64 {
	nextDelta := int64(time.Minute)
	if v, ok := m.predictors.Load(key); ok {
		nextDelta = v.(*ppe.Predictor).PredictNext()
	}

	clusterStr := m.membrane.GetClusterStrength(key)
	return float64(nextDelta) / clusterStr
}

func (m *Manager) runPQSE(targetBytes int64) error {
	freed := int64(0)
	shardCount := m.store.GetShardCount()

	for i := 0; i < shardCount && freed < targetBytes; i++ {
		s := m.store.GetShardByIndex(i)
		s.RLock()
		keys := s.GetItems()
		victims := make([]string, 0, 16)

		for k := range keys {
			freq := uint32(0)
			if fe := m.store.GetFreqEstimator(); fe != nil {
				freq = fe.Estimate(k)
			}
			if pqse.ShouldEvict(k, freq, 10000) {
				victims = append(victims, k)
				if len(victims) >= 16 {
					break
				}
			}
		}
		s.RUnlock()

		if len(victims) > 0 {
			s.Lock()
			for _, k := range victims {
				// [UPDATE] Nhận Entry
				if entry, ok := s.DeleteItem(k); ok {
					size := entry.Size()
					freed += int64(size)
					m.cleanup(k, size, entry)
					if freed >= targetBytes {
						break
					}
				}
			}
			s.Unlock()
		}
	}

	if freed < targetBytes {
		return ErrInsufficientStorage
	}
	return nil
}

// [UPDATE] Thêm tham số entry để free
func (m *Manager) cleanup(key string, size int, entry *common.Entry) {
	m.store.AddTotalBytes(-int64(size))
	m.store.AddEviction()
	if mc := m.store.GetGlobalMemCtrl(); mc != nil {
		mc.Release(int64(size))
	}

	// [NEW] Gọi Store để giải phóng bộ nhớ Arena
	if entry != nil {
		m.store.FreeEntry(entry)
	}

	m.predictors.Delete(key)
	m.membrane.Prune(key)
}

func (m *Manager) EvictKey(key string) int {
	s := m.store.GetShard(key)
	if s == nil {
		return 0
	}
	s.Lock()
	defer s.Unlock()
	// [UPDATE]
	entry, ok := s.DeleteItem(key)
	if ok {
		size := entry.Size()
		m.cleanup(key, size, entry)
		return size
	}
	return 0
}

func (m *Manager) GetMetrics() *EvictionMetrics {
	return m.metrics
}

type candidate struct {
	key      string
	size     int
	score    float64
	shardIdx int
}

type Entry struct {
	Key      string
	Size     int
	ExpireAt int64
}

func extractEntry(e interface{}) Entry {
	if e == nil {
		return Entry{}
	}
	if le, ok := e.(*list.Element); ok {
		if v, ok := le.Value.(interface {
			Key() string
			Size() int
			ExpireAt() int64
		}); ok {
			return Entry{
				Key:      v.Key(),
				Size:     v.Size(),
				ExpireAt: v.ExpireAt(),
			}
		}
	}
	return Entry{}
}

func getPrevElement(e interface{}) interface{} {
	if le, ok := e.(*list.Element); ok {
		return le.Prev()
	}
	return nil
}
