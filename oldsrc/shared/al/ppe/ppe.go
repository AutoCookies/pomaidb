package ppe

import (
	"math"
	"sync/atomic"
	"time"
)

type Predictor struct {
	emaBits    uint64
	lastAccess int64
}

func NewPredictor() *Predictor {
	return &Predictor{
		emaBits: math.Float64bits(float64(time.Second.Nanoseconds())),
	}
}

func (p *Predictor) Update(accessTime int64) {
	last := atomic.SwapInt64(&p.lastAccess, accessTime)

	if last == 0 {
		return
	}

	interval := float64(accessTime - last)

	for {
		oldBits := atomic.LoadUint64(&p.emaBits)
		oldEma := math.Float64frombits(oldBits)

		newEma := 0.2*interval + 0.8*oldEma
		newBits := math.Float64bits(newEma)

		if atomic.CompareAndSwapUint64(&p.emaBits, oldBits, newBits) {
			break
		}
	}
}

func (p *Predictor) PredictNext() int64 {
	last := atomic.LoadInt64(&p.lastAccess)
	emaBits := atomic.LoadUint64(&p.emaBits)
	ema := math.Float64frombits(emaBits)

	if last == 0 {
		return time.Now().UnixNano() + int64(ema)
	}
	return last + int64(ema)
}
