package data_types

import (
	"encoding/binary"
	"sync"

	"github.com/golang/snappy"

	"github.com/AutoCookies/pomai-cache/internal/ppcrc"
)

// Config parameters
const (
	ChunkShards    = 256 // Must be power of 2
	ChunkShardMask = ChunkShards - 1
	FixedChunkSize = 4096 // 4KB chunks
	MinObjectSize  = 1024 // Objects smaller than this won't be chunked
)

// ChunkID is now just a string type (holding raw hash bytes),
// but we define it for clarity.
type ChunkID string

type chunkEntry struct {
	data     []byte // Compressed data
	refCount int32  // Standard int32 is fine inside lock, no need for atomic here
	size     uint32 // Original uncompressed size
}

type chunkShard struct {
	mu    sync.RWMutex
	items map[string]*chunkEntry
}

type ChunkStore struct {
	shards []*chunkShard
}

func NewChunkStore() *ChunkStore {
	cs := &ChunkStore{
		shards: make([]*chunkShard, ChunkShards),
	}
	for i := 0; i < ChunkShards; i++ {
		cs.shards[i] = &chunkShard{
			items: make(map[string]*chunkEntry),
		}
	}
	return cs
}

// getShard selects the shard index based on the crc64 of the key
func (cs *ChunkStore) getShard(keyHash uint64) *chunkShard {
	return cs.shards[keyHash&ChunkShardMask]
}

// computeChunkKey returns a 64-bit hash and the raw 8-byte key string.
// It now uses the pure-Go CRC64 implementation (ppcrc) for hashing.
func computeChunkKey(data []byte) (uint64, string) {
	h := ppcrc.Sum(data)
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], h)
	return h, string(b[:])
}

// PutBatch processes multiple chunks at once to minimize locking overhead.
// It returns the list of ChunkIDs corresponding to the input data chunks.
func (cs *ChunkStore) PutBatch(chunks [][]byte) ([]ChunkID, error) {
	resultIDs := make([]ChunkID, len(chunks))

	// Pre-calculate hashes to group by shard (Optimization)
	type chunkOp struct {
		index int
		data  []byte
		hash  uint64
		key   string
	}

	// We group operations by shard to lock each shard only once
	opsByShard := make([][]chunkOp, ChunkShards)

	for i, data := range chunks {
		h, key := computeChunkKey(data)
		shardIdx := h & ChunkShardMask

		opsByShard[shardIdx] = append(opsByShard[shardIdx], chunkOp{
			index: i,
			data:  data,
			hash:  h,
			key:   key,
		})
		resultIDs[i] = ChunkID(key)
	}

	// Execute per shard
	var wg sync.WaitGroup

	for i := 0; i < ChunkShards; i++ {
		if len(opsByShard[i]) == 0 {
			continue
		}

		wg.Add(1)
		go func(shardIdx int) {
			defer wg.Done()
			shard := cs.shards[shardIdx]
			ops := opsByShard[shardIdx]

			shard.mu.Lock()
			defer shard.mu.Unlock()

			for _, op := range ops {
				entry, exists := shard.items[op.key]
				if exists {
					entry.refCount++
				} else {
					// Compress ONLY if it's a new chunk
					compressed := snappy.Encode(nil, op.data)
					shard.items[op.key] = &chunkEntry{
						data:     compressed,
						refCount: 1,
						size:     uint32(len(op.data)),
					}
				}
			}
		}(i)
	}

	wg.Wait()
	return resultIDs, nil
}

// GetChunk retrieves and decompresses a chunk
func (cs *ChunkStore) GetChunk(id ChunkID) ([]byte, bool) {
	// Reconstruct hash from the string bytes (which are raw bytes of uint64)
	if len(id) != 8 {
		return nil, false
	}
	h := binary.LittleEndian.Uint64([]byte(id))

	shard := cs.getShard(h)
	shard.mu.RLock()
	entry, ok := shard.items[string(id)]
	shard.mu.RUnlock()

	if !ok {
		return nil, false
	}

	// Decompress
	decompressed, err := snappy.Decode(nil, entry.data)
	if err != nil {
		return nil, false // Should treat corruption as miss
	}

	return decompressed, true
}

// DecRefBatch decreases reference counts for a list of chunk IDs.
// If refCount reaches 0, the chunk is deleted.
func (cs *ChunkStore) DecRefBatch(ids []ChunkID) {
	// Group by shard
	idsByShard := make([][]string, ChunkShards)
	for _, id := range ids {
		if len(id) != 8 {
			continue
		}
		h := binary.LittleEndian.Uint64([]byte(id))
		idx := h & ChunkShardMask
		idsByShard[idx] = append(idsByShard[idx], string(id))
	}

	for i := 0; i < ChunkShards; i++ {
		if len(idsByShard[i]) == 0 {
			continue
		}

		// Run async or sync? Async is better for delete latency but harder to track memory immediately.
		// Let's do sync for safety now.
		shard := cs.shards[i]
		keys := idsByShard[i]

		shard.mu.Lock()
		for _, key := range keys {
			if entry, ok := shard.items[key]; ok {
				entry.refCount--
				if entry.refCount <= 0 {
					delete(shard.items, key)
				}
			}
		}
		shard.mu.Unlock()
	}
}

// Stats returns approximate stats
func (cs *ChunkStore) Stats() (uniqueChunks int, dedupBytes int64) {
	for _, s := range cs.shards {
		s.mu.RLock()
		uniqueChunks += len(s.items)
		for _, e := range s.items {
			dedupBytes += int64(len(e.data))
		}
		s.mu.RUnlock()
	}
	return
}
