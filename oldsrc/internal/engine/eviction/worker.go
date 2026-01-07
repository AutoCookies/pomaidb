// File: internal/engine/eviction/worker.go
package eviction

import (
	"context"
	"sync/atomic"
	"time"
)

// Worker handles background eviction
type Worker struct {
	manager           *Manager
	trigger           chan struct{}
	inFlight          *atomic.Bool
	lastEvictionTime  *atomic.Int64
	metrics           *EvictionMetrics
	evictionThreshold float64
	evictionTarget    float64
	ctx               context.Context
	cancel            context.CancelFunc
}

// NewWorker creates eviction worker
func NewWorker(
	manager *Manager,
	metrics *EvictionMetrics,
	threshold, target float64,
) *Worker {
	ctx, cancel := context.WithCancel(context.Background())

	return &Worker{
		manager:           manager,
		trigger:           make(chan struct{}, 1),
		inFlight:          &atomic.Bool{},
		lastEvictionTime:  &atomic.Int64{},
		metrics:           metrics,
		evictionThreshold: threshold,
		evictionTarget:    target,
		ctx:               ctx,
		cancel:            cancel,
	}
}

// Start starts the worker goroutine
func (w *Worker) Start() {
	go w.run()
}

// Stop stops the worker
func (w *Worker) Stop() {
	w.cancel()
}

// TriggerAsync triggers async eviction
func (w *Worker) TriggerAsync() {
	if !w.inFlight.CompareAndSwap(false, true) {
		return // Already in flight
	}

	select {
	case w.trigger <- struct{}{}:
		w.metrics.AsyncEvictions.Add(1)
	default:
		w.inFlight.Store(false)
	}
}

// run is the main worker loop
func (w *Worker) run() {
	tenantID := w.manager.store.GetTenantID()

	logEviction("Worker started for tenant=%s (threshold:  %.0f%%, target: %.0f%%)",
		tenantID, w.evictionThreshold*100, w.evictionTarget*100)

	for {
		select {
		case <-w.ctx.Done():
			logEviction("Worker stopped for tenant=%s", tenantID)
			return

		case <-w.trigger:
			w.performEviction()
		}
	}
}

// performEviction performs single eviction cycle
func (w *Worker) performEviction() {
	start := time.Now()

	capacityBytes := w.manager.store.GetCapacityBytes()
	currentBytes := w.manager.store.GetTotalBytes()
	targetBytes := int64(float64(capacityBytes) * w.evictionTarget)
	toFree := currentBytes - targetBytes

	if toFree <= 0 {
		w.inFlight.Store(false)
		return
	}

	// Perform eviction
	freed := w.manager.BatchEvict(toFree)
	duration := time.Since(start)

	// Update state
	w.inFlight.Store(false)
	w.lastEvictionTime.Store(time.Now().UnixNano())
	w.metrics.LastEvictionDuration.Store(duration.Milliseconds())

	if freed > 0 {
		w.metrics.BytesFreed.Add(freed)
		w.metrics.TotalEvictions.Add(1)

		// Update average eviction time
		totalEvictions := w.metrics.TotalEvictions.Load()
		if totalEvictions > 0 {
			avgTime := (w.metrics.AvgEvictionTimeMs.Load()*int64(totalEvictions-1) +
				duration.Milliseconds()) / int64(totalEvictions)
			w.metrics.AvgEvictionTimeMs.Store(avgTime)
		}

		// Log
		currentUsage := float64(w.manager.store.GetTotalBytes()) /
			float64(capacityBytes) * 100

		logEviction("tenant=%s freed=%s target=%s duration=%v usage=%.1f%%",
			w.manager.store.GetTenantID(),
			FormatBytes(freed),
			FormatBytes(toFree),
			duration,
			currentUsage)
	}
}

// IsInFlight returns whether eviction is in progress
func (w *Worker) IsInFlight() bool {
	return w.inFlight.Load()
}

// GetLastEvictionTime returns last eviction timestamp
func (w *Worker) GetLastEvictionTime() int64 {
	return w.lastEvictionTime.Load()
}
