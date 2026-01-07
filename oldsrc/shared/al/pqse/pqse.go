package pqse

import (
	"sync/atomic"
	"time"
)

var rngSeed uint32

func init() {
	seed := uint32(time.Now().UnixNano() & 0xffffffff)
	if seed == 0 {
		seed = 1
	}
	atomic.StoreUint32(&rngSeed, seed)
}

func fastrand() uint32 {
	for {
		old := atomic.LoadUint32(&rngSeed)
		x := old
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		if atomic.CompareAndSwapUint32(&rngSeed, old, x) {
			return x
		}
	}
}

func Hash(key string) uint64 {
	var h uint64 = 0x9E3779B9
	for i := 0; i < len(key); i++ {
		h ^= uint64(key[i])
		h *= 0x5bd1e99558997a29
		h ^= h >> 47
	}
	return h
}

func ShouldEvict(key string, freq uint32, maxFreq float64) bool {
	h := Hash(key)
	probUint := h % 1000

	if maxFreq <= 0 {
		maxFreq = 1
	}

	freqFactor := 1.0 - (float64(freq) / maxFreq)
	if freqFactor < 0 {
		freqFactor = 0
	}
	threshold := float64(probUint) / 1000.0 * freqFactor

	if threshold < 0.05 {
		threshold = 0.05
	}
	randVal := float64(fastrand()) / 4294967295.0

	return randVal < threshold
}
