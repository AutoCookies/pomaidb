package storage

import (
	"bytes"
	"compress/flate"
	"hash/fnv"
	"io"
	"math"
	"sync"
	"sync/atomic"

	"github.com/golang/snappy"
)

const (
	CompNone   = 0
	CompSnappy = 1
	CompFlate  = 2
)

type PECCompressor struct {
	hist [256]int
}

func NewPEC() *PECCompressor {
	return &PECCompressor{}
}

func (c *PECCompressor) Analyze(data []byte) int {
	if len(data) < 64 {
		return CompNone
	}

	for i := range c.hist {
		c.hist[i] = 0
	}

	step := 1
	if len(data) > 10240 {
		step = 10
	}
	for i := 0; i < len(data); i += step {
		c.hist[data[i]]++
	}

	total := len(data) / step
	entropy := 0.0
	for _, count := range c.hist {
		if count > 0 {
			p := float64(count) / float64(total)
			entropy -= p * math.Log2(p)
		}
	}

	if entropy > 7.5 {
		return CompNone
	} else if entropy > 4.5 {
		return CompSnappy
	}
	return CompFlate
}

func (c *PECCompressor) Compress(data []byte, algo int) []byte {
	switch algo {
	case CompSnappy:
		return snappy.Encode(nil, data)
	case CompFlate:
		var buf bytes.Buffer
		fw, _ := flate.NewWriter(&buf, flate.BestCompression)
		fw.Write(data)
		fw.Close()
		return buf.Bytes()
	default:
		return data
	}
}

func (c *PECCompressor) Decompress(data []byte, algo int) ([]byte, error) {
	switch algo {
	case CompSnappy:
		return snappy.Decode(nil, data)
	case CompFlate:
		fr := flate.NewReader(bytes.NewReader(data))
		defer fr.Close()
		return io.ReadAll(fr)
	default:
		return data, nil
	}
}

type GranuleID VirtualPtr

type GranuleMeta struct {
	Ptr  VirtualPtr
	Refs int32
}

type pgusShard struct {
	items map[uint64]*GranuleMeta
	mu    sync.RWMutex
}

type PGUS struct {
	shards [64]*pgusShard
	vStore *VirtualStore
	pec    *PECCompressor
}

func NewPGUS(virtualPath string) *PGUS {
	vStore, err := NewVirtualStore(virtualPath, 1<<30)
	if err != nil {
		panic(err)
	}

	p := &PGUS{
		vStore: vStore,
		pec:    NewPEC(),
	}
	for i := 0; i < 64; i++ {
		p.shards[i] = &pgusShard{
			items: make(map[uint64]*GranuleMeta),
		}
	}
	return p
}

func (p *PGUS) Write(value []byte) []GranuleID {
	if len(value) == 0 {
		return nil
	}

	chunkSize := 4096
	numChunks := (len(value) + chunkSize - 1) / chunkSize
	ids := make([]GranuleID, 0, numChunks)

	for i := 0; i < len(value); i += chunkSize {
		end := i + chunkSize
		if end > len(value) {
			end = len(value)
		}

		chunkData := value[i:end]

		algo := p.pec.Analyze(chunkData)
		compressed := p.pec.Compress(chunkData, algo)

		storageData := make([]byte, 1+len(compressed))
		storageData[0] = byte(algo)
		copy(storageData[1:], compressed)

		h := fnv.New64a()
		h.Write(storageData)
		keyHash := h.Sum64()

		shard := p.shards[keyHash%64]
		shard.mu.Lock()

		if meta, exists := shard.items[keyHash]; exists {
			atomic.AddInt32(&meta.Refs, 1)
			ids = append(ids, GranuleID(meta.Ptr))
		} else {
			ptr, err := p.vStore.Alloc(storageData, true)
			if err == nil {
				shard.items[keyHash] = &GranuleMeta{Ptr: ptr, Refs: 1}
				ids = append(ids, GranuleID(ptr))
			}
		}
		shard.mu.Unlock()
	}
	return ids
}

func (p *PGUS) Read(ids []GranuleID) []byte {
	var out []byte
	for _, id := range ids {
		data, ok := p.vStore.Get(VirtualPtr(id))
		if ok && len(data) > 0 {
			algo := int(data[0])
			payload := data[1:]
			decoded, err := p.pec.Decompress(payload, algo)
			if err == nil {
				out = append(out, decoded...)
			}
		}
	}
	return out
}

func (p *PGUS) Dereference(ids []GranuleID) {
	for _, id := range ids {
		_ = id
	}
}

func (p *PGUS) EternalOptimize() {
	for i := range 64 {
		shard := p.shards[i]
		shard.mu.Lock()
	}
}
