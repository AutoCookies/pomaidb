// File: internal/adapter/persistence/file/persister.go
package file

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/core/ports"
)

type FilePersister struct {
	dataDir string
	mu      sync.Mutex

	// Periodic snapshot support
	snapshotPath     string
	snapshotInterval time.Duration
	snapshotCtx      context.Context
	snapshotCancel   context.CancelFunc
}

func NewFilePersister(dataDir string) (*FilePersister, error) {
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create data dir: %w", err)
	}

	snapshotPath := filepath.Join(dataDir, "snapshot.gob")

	return &FilePersister{
		dataDir:      dataDir,
		snapshotPath: snapshotPath,
	}, nil
}

// Persist writes a single key-value pair to a file
func (f *FilePersister) Persist(key string, value []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	path := filepath.Join(f.dataDir, sanitizeFilename(key)+".cache")
	return os.WriteFile(path, value, 0o644)
}

// PersistBatch writes multiple key-value pairs
func (f *FilePersister) PersistBatch(ops []ports.WriteOp) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	for _, op := range ops {
		path := filepath.Join(f.dataDir, sanitizeFilename(op.Key)+".cache")
		if err := os.WriteFile(path, op.Value, 0o644); err != nil {
			return fmt.Errorf("failed to persist key %s: %w", op.Key, err)
		}
	}

	return nil
}

// Load reads a value by key from file
func (f *FilePersister) Load(key string) ([]byte, error) {
	path := filepath.Join(f.dataDir, sanitizeFilename(key)+".cache")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // Key not found
		}
		return nil, err
	}
	return data, nil
}

// [SỬA] Thay *engine.Store bằng ports.Serializable
func (f *FilePersister) Snapshot(target ports.Serializable) error {
	return f.snapshotToTempAndRename(target)
}

// [SỬA] Thay *engine.Store bằng ports.Serializable
func (f *FilePersister) Restore(target ports.Serializable) error {
	file, err := os.Open(f.snapshotPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Println("[PERSISTENCE] No snapshot found, starting fresh")
			return nil // No snapshot to restore
		}
		return fmt.Errorf("failed to open snapshot: %w", err)
	}
	defer file.Close()

	if err := target.RestoreFrom(file); err != nil {
		return fmt.Errorf("failed to restore from snapshot: %w", err)
	}

	log.Printf("[PERSISTENCE] Restored from snapshot: %s", f.snapshotPath)
	return nil
}

// [SỬA] Thay *engine.Store bằng ports.Serializable
func (f *FilePersister) StartPeriodicSnapshot(target ports.Serializable, interval time.Duration) {
	if interval <= 0 {
		log.Println("[SNAPSHOT] Periodic snapshot disabled (interval <= 0)")
		return
	}

	f.snapshotInterval = interval
	f.snapshotCtx, f.snapshotCancel = context.WithCancel(context.Background())

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		log.Printf("[SNAPSHOT] Periodic snapshot started: interval=%v, path=%s", interval, f.snapshotPath)

		for {
			select {
			case <-f.snapshotCtx.Done():
				// Final snapshot on shutdown
				log.Println("[SNAPSHOT] Stopping, creating final snapshot...")
				if err := f.snapshotToTempAndRename(target); err != nil {
					log.Printf("[SNAPSHOT] ERROR: Final snapshot failed: %v", err)
				} else {
					log.Println("[SNAPSHOT] Final snapshot completed")
				}
				return

			case <-ticker.C:
				if err := f.snapshotToTempAndRename(target); err != nil {
					log.Printf("[SNAPSHOT] ERROR: %v", err)
				} else {
					// Lưu ý: Không thể gọi s.Stats() ở đây nữa vì Serializable không có method Stats()
					// Nếu muốn log stats, bạn cần mở rộng interface Serializable hoặc chấp nhận log đơn giản hơn.
					log.Printf("[SNAPSHOT] Snapshot created successfully at %s", time.Now().Format(time.RFC3339))
				}
			}
		}
	}()
}

// StopPeriodicSnapshot stops the periodic snapshot goroutine
func (f *FilePersister) StopPeriodicSnapshot() {
	if f.snapshotCancel != nil {
		f.snapshotCancel()
	}
}

func (f *FilePersister) SnapshotNow(target ports.Serializable) error {
	return f.snapshotToTempAndRename(target)
}

func (f *FilePersister) snapshotToTempAndRename(target ports.Serializable) error {
	dir := filepath.Dir(f.snapshotPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create snapshot dir: %w", err)
	}

	tmpPath := fmt.Sprintf("%s.tmp.%d", f.snapshotPath, time.Now().UnixNano())

	file, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("failed to create temp snapshot: %w", err)
	}

	// Write snapshot using interface method
	if err := target.SnapshotTo(file); err != nil {
		file.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed to write snapshot: %w", err)
	}

	// Ensure fsync
	if err := file.Sync(); err != nil {
		file.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("failed to sync snapshot: %w", err)
	}

	if err := file.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to close snapshot: %w", err)
	}

	// Atomic rename
	if err := os.Rename(tmpPath, f.snapshotPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename snapshot: %w", err)
	}

	return nil
}

func (f *FilePersister) Close() error {
	f.StopPeriodicSnapshot()
	return nil
}

func sanitizeFilename(key string) string {
	result := make([]byte, len(key))
	for i, c := range []byte(key) {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' {
			result[i] = c
		} else {
			result[i] = '_'
		}
	}
	return string(result)
}
