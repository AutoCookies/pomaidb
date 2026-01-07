package wal

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/AutoCookies/pomai-cache/internal/core/ports"
	"github.com/AutoCookies/pomai-cache/internal/ppcrc"
)

type WalTarget interface {
	ports.Serializable
	Put(key string, value []byte, ttl time.Duration) error
}

// WALPersister uses Write-Ahead Log for durability
type WALPersister struct {
	walPath string
	file    *os.File
	mu      sync.Mutex
}

// Struct nội bộ để lưu vào file
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
	}, nil
}

func (w *WALPersister) Persist(key string, value []byte) error {
	// Wrap single persist into batch of 1 for consistent format
	op := walEntry{
		Key:   key,
		Value: value,
		TTL:   0,
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	// Encode to buffer first so we can compute CRC over encoded bytes
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(&op); err != nil {
		return fmt.Errorf("gob encode failed: %w", err)
	}
	data := buf.Bytes()
	crc := ppcrc.CRC64(0, data)

	// Write: [uint32 len][data][uint64 crc]
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(data)))

	if _, err := w.file.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := w.file.Write(data); err != nil {
		return err
	}
	var crcBuf [8]byte
	binary.BigEndian.PutUint64(crcBuf[:], crc)
	if _, err := w.file.Write(crcBuf[:]); err != nil {
		return err
	}

	if err := w.file.Sync(); err != nil {
		return err
	}
	return nil
}

func (w *WALPersister) PersistBatch(ops []ports.WriteOp) error {
	if len(ops) == 0 {
		return nil
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	// We'll write entries one by one but keep everything in the file in the length+data+crc format.
	for _, opIn := range ops {
		entry := walEntry{
			Key:   opIn.Key,
			Value: opIn.Value,
			TTL:   int64(opIn.TTL),
		}

		var buf bytes.Buffer
		enc := gob.NewEncoder(&buf)
		if err := enc.Encode(&entry); err != nil {
			return fmt.Errorf("gob encode failed: %w", err)
		}
		data := buf.Bytes()
		crc := ppcrc.CRC64(0, data)

		// length
		var hdr [4]byte
		binary.BigEndian.PutUint32(hdr[:], uint32(len(data)))
		if _, err := w.file.Write(hdr[:]); err != nil {
			return err
		}
		// data
		if _, err := w.file.Write(data); err != nil {
			return err
		}
		// crc
		var crcBuf [8]byte
		binary.BigEndian.PutUint64(crcBuf[:], crc)
		if _, err := w.file.Write(crcBuf[:]); err != nil {
			return err
		}
	}

	// Ensure durable
	if err := w.file.Sync(); err != nil {
		return err
	}
	return nil
}

func (w *WALPersister) Load(key string) ([]byte, error) {
	return nil, fmt.Errorf("WAL persister does not support Load")
}

// Snapshot saves a snapshot of the target to a snapshot file and truncates WAL.
// Snapshot will create <walPath>.snapshot (atomic rename) and on success it truncates the WAL.
// The snapshot file itself keeps the format produced by target.SnapshotTo (no extra trailer here).
func (w *WALPersister) Snapshot(target WalTarget) error {
	snapshotPath := w.walPath + ".snapshot"
	tmpPath := snapshotPath + ".tmp"

	file, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	// Gọi qua interface to produce snapshot content (target is responsible for format)
	if err := target.SnapshotTo(file); err != nil {
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

	if w.file != nil {
		w.file.Close()
	}

	if err := os.Truncate(w.walPath, 0); err != nil {
		return err
	}

	fileHandle, err := os.OpenFile(w.walPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	w.file = fileHandle

	return nil
}

// RestoreFrom restores state from snapshot + WAL.
// It will first try to restore snapshot (if present) using target.RestoreFrom(snapshotFile).
// Then it will replay WAL entries; each WAL entry is validated using CRC64. Corrupted entries stop replay.
func (w *WALPersister) RestoreFrom(target WalTarget) error {
	// 1. Restore from snapshot first
	snapshotPath := w.walPath + ".snapshot"
	if _, err := os.Stat(snapshotPath); err == nil {
		file, err := os.Open(snapshotPath)
		if err != nil {
			return err
		}
		if err := target.RestoreFrom(file); err != nil {
			file.Close()
			return fmt.Errorf("failed to restore from snapshot: %w", err)
		}
		file.Close()
	}

	// 2. Replay WAL
	file, err := os.Open(w.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	count := 0

	for {
		// Read length (4 bytes)
		var lenBuf [4]byte
		if _, err := io.ReadFull(reader, lenBuf[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				// End of WAL
				break
			}
			return fmt.Errorf("failed to read entry length: %w", err)
		}
		entryLen := binary.BigEndian.Uint32(lenBuf[:])
		if entryLen == 0 {
			// invalid entry
			return fmt.Errorf("invalid entry length 0 at pos %d", count)
		}

		// Read encoded data
		data := make([]byte, entryLen)
		if _, err := io.ReadFull(reader, data); err != nil {
			return fmt.Errorf("failed to read entry data: %w", err)
		}

		// Read CRC
		var crcBuf [8]byte
		if _, err := io.ReadFull(reader, crcBuf[:]); err != nil {
			return fmt.Errorf("failed to read entry crc: %w", err)
		}
		expected := binary.BigEndian.Uint64(crcBuf[:])

		// Verify CRC
		got := ppcrc.CRC64(0, data)
		if got != expected {
			return fmt.Errorf("wal crc mismatch at entry %d: expected %016x got %016x", count, expected, got)
		}

		// Decode walEntry from data
		var entry walEntry
		if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&entry); err != nil {
			return fmt.Errorf("failed to decode wal entry: %w", err)
		}

		// Replay using interface method Put
		if err := target.Put(entry.Key, entry.Value, 0); err != nil {
			fmt.Printf("[WAL] replay put failed for key=%s: %v\n", entry.Key, err)
		}
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
