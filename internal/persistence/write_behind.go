// File: internal/persistence/write_behind.go
package persistence

import (
	"context"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

type WriteOp struct {
	Key   string
	Value []byte
	TTL   time.Duration
}

type WriteBehindBuffer struct {
	mu            sync.RWMutex
	buffer        []WriteOp
	maxSize       int
	flushInterval time.Duration
	persister     Persister

	// Stats
	totalWrites   uint64
	totalFlushes  uint64
	failedFlushes uint64
	lastFlushTime int64
}

func NewWriteBehindBuffer(maxSize int, flushInterval time.Duration, p Persister) *WriteBehindBuffer {
	if maxSize <= 0 {
		maxSize = 1000
	}
	if flushInterval <= 0 {
		flushInterval = 5 * time.Second
	}

	return &WriteBehindBuffer{
		buffer:        make([]WriteOp, 0, maxSize),
		maxSize:       maxSize,
		flushInterval: flushInterval,
		persister:     p,
	}
}

// Add adds a write operation to the buffer
func (w *WriteBehindBuffer) Add(key string, value []byte, ttl time.Duration) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.buffer = append(w.buffer, WriteOp{Key: key, Value: value, TTL: ttl})
	atomic.AddUint64(&w.totalWrites, 1)

	if len(w.buffer) >= w.maxSize {
		// Async flush when buffer full
		go w.flush()
	}
}

// Start starts the periodic flush goroutine
func (w *WriteBehindBuffer) Start(ctx context.Context) {
	ticker := time.NewTicker(w.flushInterval)
	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				log.Println("[WRITE-BEHIND] Stopping, performing final flush...")
				w.flush()
				return
			case <-ticker.C:
				w.flush()
			}
		}
	}()

	log.Printf("[WRITE-BEHIND] Started:  maxSize=%d, flushInterval=%v", w.maxSize, w.flushInterval)
}

// flush writes buffered operations to disk
func (w *WriteBehindBuffer) flush() {
	w.mu.Lock()
	if len(w.buffer) == 0 {
		w.mu.Unlock()
		return
	}

	// Swap buffer
	ops := make([]WriteOp, len(w.buffer))
	copy(ops, w.buffer)
	w.buffer = w.buffer[:0]
	w.mu.Unlock()

	// Persist batch
	start := time.Now()
	if err := w.persister.PersistBatch(ops); err != nil {
		log.Printf("[WRITE-BEHIND] ERROR:  Flush failed: %v", err)
		atomic.AddUint64(&w.failedFlushes, 1)

		// Re-add failed operations back to buffer (optional)
		w.mu.Lock()
		w.buffer = append(w.buffer, ops...)
		w.mu.Unlock()
		return
	}

	atomic.AddUint64(&w.totalFlushes, 1)
	atomic.StoreInt64(&w.lastFlushTime, time.Now().Unix())

	duration := time.Since(start)
	log.Printf("[WRITE-BEHIND] Flushed %d operations in %v", len(ops), duration)
}

// FlushNow forces an immediate flush
func (w *WriteBehindBuffer) FlushNow() error {
	w.flush()
	return nil
}

// Stats returns write-behind statistics
type WriteBehindStats struct {
	TotalWrites   uint64
	TotalFlushes  uint64
	FailedFlushes uint64
	BufferSize    int
	LastFlushTime time.Time
}

func (w *WriteBehindBuffer) Stats() WriteBehindStats {
	w.mu.RLock()
	bufferSize := len(w.buffer)
	w.mu.RUnlock()

	lastFlush := atomic.LoadInt64(&w.lastFlushTime)

	return WriteBehindStats{
		TotalWrites:   atomic.LoadUint64(&w.totalWrites),
		TotalFlushes:  atomic.LoadUint64(&w.totalFlushes),
		FailedFlushes: atomic.LoadUint64(&w.failedFlushes),
		BufferSize:    bufferSize,
		LastFlushTime: time.Unix(lastFlush, 0),
	}
}

// Close flushes pending writes and closes the persister
func (w *WriteBehindBuffer) Close() error {
	w.flush()
	if w.persister != nil {
		return w.persister.Close()
	}
	return nil
}
