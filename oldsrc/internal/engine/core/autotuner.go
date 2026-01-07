// File: internal/engine/core/autotuner.go
package core

import (
	"runtime"
	"sync/atomic"
	"time"
)

type AutoTuner struct {
	store          *Store
	samples        int
	totalLatency   atomic.Int64
	requestCount   atomic.Uint64
	lastAdjust     time.Time
	currentProfile string
}

func NewAutoTuner(store *Store) *AutoTuner {
	return &AutoTuner{
		store:          store,
		lastAdjust:     time.Now(),
		currentProfile: "balanced",
	}
}

func (at *AutoTuner) Start() {
	go at.monitorLoop()
}

func (at *AutoTuner) monitorLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		at.analyze()
	}
}

func (at *AutoTuner) analyze() {
	requests := at.requestCount.Swap(0)
	if requests == 0 {
		return
	}

	qps := float64(requests) / 5.0

	var targetProfile string

	if qps > 100000 {
		targetProfile = "ultra"
	} else if qps > 50000 {
		targetProfile = "high"
	} else if qps > 10000 {
		targetProfile = "balanced"
	} else {
		targetProfile = "lowlatency"
	}

	if targetProfile != at.currentProfile {
		at.applyProfile(targetProfile)
	}
}

func (at *AutoTuner) applyProfile(profile string) {
	at.currentProfile = profile

	switch profile {
	case "ultra":
		runtime.GOMAXPROCS(runtime.NumCPU())
	case "high":
		runtime.GOMAXPROCS(runtime.NumCPU())
	case "balanced":
		runtime.GOMAXPROCS(runtime.NumCPU())
	case "lowlatency":
		if runtime.NumCPU() > 2 {
			runtime.GOMAXPROCS(runtime.NumCPU() - 1)
		}
	}
}

func (at *AutoTuner) RecordRequest() {
	at.requestCount.Add(1)
}
