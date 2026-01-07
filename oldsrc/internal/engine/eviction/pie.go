package eviction

import (
	"math"
	"math/rand"
	"sync"
)

type ArmConfig struct {
	EfSearch    int
	EvictSample int
}

type PIEBandit struct {
	mu         sync.RWMutex
	arms       []ArmConfig
	counts     []int
	values     []float64
	total      int
	currentArm int
}

func NewPIEBandit() *PIEBandit {
	arms := []ArmConfig{
		{EfSearch: 40, EvictSample: 4},
		{EfSearch: 64, EvictSample: 8},
		{EfSearch: 100, EvictSample: 16},
		{EfSearch: 200, EvictSample: 24},
		{EfSearch: 400, EvictSample: 32},
	}

	return &PIEBandit{
		arms:       arms,
		counts:     make([]int, len(arms)),
		values:     make([]float64, len(arms)),
		currentArm: 2,
	}
}

func (b *PIEBandit) SelectArm() (int, ArmConfig) {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := b.total
	if n == 0 {
		idx := rand.Intn(len(b.arms))
		b.currentArm = idx
		return idx, b.arms[idx]
	}

	maxUCB := -math.MaxFloat64
	bestArm := 0

	for i := 0; i < len(b.arms); i++ {
		if b.counts[i] == 0 {
			b.currentArm = i
			return i, b.arms[i]
		}

		bonus := math.Sqrt(2 * math.Log(float64(n)) / float64(b.counts[i]))
		ucb := b.values[i] + bonus

		if ucb > maxUCB {
			maxUCB = ucb
			bestArm = i
		}
	}

	b.currentArm = bestArm
	return bestArm, b.arms[bestArm]
}

func (b *PIEBandit) UpdateReward(arm int, reward float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.counts[arm]++
	b.total++

	n := float64(b.counts[arm])
	oldVal := b.values[arm]
	b.values[arm] = oldVal + (reward-oldVal)/n
}

func (b *PIEBandit) GetCurrentConfig() ArmConfig {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.arms[b.currentArm]
}
