// File: internal/persistence/wal_persister.go
package persistence

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/AutoCookies/pomai-cache/internal/cache"
)

// WALPersister uses Write-Ahead Log for durability
type WALPersister struct {
	walPath string
	file    *os.File
	encoder *gob.Encoder
	mu      sync.Mutex
}

type walEntry struct {
	Key   string
	Value []byte
	TTL   int64 // nanoseconds
}

func NewWALPersister(walPath string) (*WALPersister, error) {
	if err := os.MkdirAll(filepath.Dir(walPath), 0o755); err != nil {
		return nil, err
	}

	file, err := os.OpenFile(walPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}

	return &WALPersister{
		walPath: walPath,
		file:    file,
		encoder: gob.NewEncoder(file),
	}, nil
}

func (w *WALPersister) Persist(key string, value []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	entry := walEntry{
		Key:   key,
		Value: value,
		TTL:   0,
	}

	if err := w.encoder.Encode(&entry); err != nil {
		return err
	}

	return w.file.Sync()
}

func (w *WALPersister) PersistBatch(ops []WriteOp) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for _, op := range ops {
		entry := walEntry{
			Key:   op.Key,
			Value: op.Value,
			TTL:   int64(op.TTL),
		}

		if err := w.encoder.Encode(&entry); err != nil {
			return fmt.Errorf("failed to encode entry: %w", err)
		}
	}

	return w.file.Sync()
}

func (w *WALPersister) Load(key string) ([]byte, error) {
	// WAL doesn't support individual key lookup
	return nil, fmt.Errorf("WAL persister does not support Load")
}

func (w *WALPersister) Snapshot(s *cache.Store) error {
	snapshotPath := w.walPath + ".snapshot"
	tmpPath := snapshotPath + ". tmp"

	file, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	if err := s.SnapshotTo(file); err != nil {
		file.Close()
		os.Remove(tmpPath)
		return err
	}

	if err := file.Sync(); err != nil {
		file.Close()
		os.Remove(tmpPath)
		return err
	}

	file.Close()

	// Atomic rename
	if err := os.Rename(tmpPath, snapshotPath); err != nil {
		return err
	}

	// Truncate WAL after successful snapshot
	w.mu.Lock()
	defer w.mu.Unlock()

	w.file.Close()

	if err := os.Truncate(w.walPath, 0); err != nil {
		return err
	}

	file, err = os.OpenFile(w.walPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}

	w.file = file
	w.encoder = gob.NewEncoder(file)

	return nil
}

func (w *WALPersister) Restore(s *cache.Store) error {
	// Restore from snapshot first
	snapshotPath := w.walPath + ".snapshot"
	if _, err := os.Stat(snapshotPath); err == nil {
		file, err := os.Open(snapshotPath)
		if err != nil {
			return err
		}
		defer file.Close()

		if err := s.RestoreFrom(file); err != nil {
			return fmt.Errorf("failed to restore from snapshot: %w", err)
		}
	}

	// Replay WAL
	file, err := os.Open(w.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(bufio.NewReader(file))
	count := 0

	for {
		var entry walEntry
		if err := decoder.Decode(&entry); err != nil {
			break
		}

		// Replay entry into store
		s.Put(entry.Key, entry.Value, 0) // TTL already expired, use 0
		count++
	}

	if count > 0 {
		fmt.Printf("[WAL] Replayed %d entries from WAL\n", count)
	}

	return nil
}

func (w *WALPersister) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.file != nil {
		if err := w.file.Sync(); err != nil {
			return err
		}
		return w.file.Close()
	}

	return nil
}
