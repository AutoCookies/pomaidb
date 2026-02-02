# PomaiDB Production Transformation - Implementation Plan

## Overview

Transform PomaiDB into production-grade embedded vector database combining:
- **SQLite-grade**: durability, simplicity, embeddability
- **Dragonfly-grade**: throughput, tail-latency, multi-core scaling

---

## Phase 1: Correctness Foundation (SQLite-grade)

### Goal
Establish rock-solid durability and crash safety.

### Deliverables

#### 1.1 Error Handling Audit
**Files to modify:**
- `include/pomai/status.h` - Add missing error codes
- `src/api/db.cc` - Standardize error paths
- `src/core/engine/engine.cc` - Propagate errors correctly
- `src/storage/wal/wal.cc` - Robust error handling

**Changes:**
- Add `ErrorCode::kIOError`, `kCorruption`, `kInvalidArgument`
- Unify error messages (format: "Operation failed: {detail}")
- Audit all `Status::Ok()` checks in hot paths

**Tests:**
- `tests/unit/status_test.cc` - Error code coverage

---

#### 1.2 WriteBatch API
**Files to create:**
- `include/pomai/write_batch.h` - Public API
- `src/api/write_batch.cc` - WriteBatch implementation
- `src/core/shard/commands.h` - `WriteBatchCmd`
- `tests/unit/write_batch_test.cc`
- `tests/integ/db_write_batch_test.cc`

**API:**
```cpp
class WriteBatch {
public:
    WriteBatch();
    void Put(VectorId id, span<const float> vec);
    void Delete(VectorId id);
    void Clear();
    size_t Count() const;
private:
    struct Op { enum Type { kPut, kDelete }; /* ... */ };
    vector<Op> ops_;
};

class DB {
    virtual Status Write(const WriteBatch& batch) = 0;
    virtual Status Write(string_view membrane, const WriteBatch& batch) = 0;
};
```

**Implementation:**
- Group ops by shard
- Enqueue `WriteBatchCmd` per shard
- Single WAL batch write per shard (one fsync)
- Atomic apply to MemTable + IVF

**Tests:**
- Unit: batch encode/decode, empty batch
- Integration: atomic apply, crash mid-batch

---

#### 1.3 Snapshot/MVCC-lite
**Files to modify:**
- `include/pomai/snapshot.h` - New public API
- `src/core/shard/shard_state.h` - Immutable state struct
- `src/core/shard/runtime.h` - Add sequence tracking
- `src/storage/wal/wal.h` - Add `seq` to WAL records
- `src/table/memtable.h` - Add version support
- `tests/integ/db_snapshot_test.cc`

**Design:**
```cpp
struct ShardState {
    uint64_t seq;
    shared_ptr<const MemTable> mem;
    shared_ptr<const IvfCoarse> ivf;
};

class Snapshot {
    uint64_t seq() const;
private:
    map<uint32_t, ShardState> shard_states_;
};

class DB {
    virtual unique_ptr<Snapshot> GetSnapshot() = 0;
    virtual Status Search(const Snapshot* snap, ...) = 0;
};
```

**Implementation:**
- Add `seq` to `PutCmd`, `DelCmd`, `WriteBatchCmd`
- WAL records include `seq`
- MemTable entries: `(VectorId, seq) → float*`
- `GetSnapshot()` atomically captures current `ShardState` refs
- Search with snapshot filters by `entry.seq <= snap.seq`

**Tests:**
- Snapshot consistency: write, snapshot, write, read from snapshot
- Concurrent reads from different snapshots

---

#### 1.4 Checkpoint v1
**Files to create:**
- `include/pomai/checkpoint.h` - Public API
- `src/storage/checkpoint/checkpoint.h` - Checkpoint writer
- `src/storage/checkpoint/checkpoint.cc`
- `src/storage/segment/segment_writer.h` - SSTable-lite writer
- `src/storage/segment/segment_reader.h` - SSTable-lite reader
- `tests/unit/checkpoint_test.cc`
- `tests/integ/db_checkpoint_test.cc`

**API:**
```cpp
class DB {
    virtual Status Checkpoint() = 0;
};
```

**Segment file format v1:**
```
Header:
    magic: "POMAI.SEG" (8 bytes)
    version: uint32 = 1
    shard_id: uint32
    min_seq: uint64
    max_seq: uint64
    num_vectors: uint64
    dim: uint32
    crc32: uint32 (header checksum)

Data blocks:
    Block: [VectorId (8) | seq (8) | floats (dim*4)]
    Repeated num_vectors times

Footer:
    Index: [VectorId → offset] (sorted by VectorId)
    Footer magic: "POMAISEG" (8 bytes)
    Footer offset: uint64
    Footer CRC: uint32
```

**Implementation:**
- `Checkpoint()`:
  1. For each shard: enqueue `CheckpointCmd`
  2. Shard thread:
     - Atomically swap `MemTable` → immutable
     - Write segment file: `segment_{shard}_{seq}.sst`
     - Rotate WAL: truncate old, start new
     - Update manifest with checkpoint metadata
     - Switch to new MemTable
  3. Atomic manifest update
- Startup: load segment + replay WAL from `checkpoint_seq`

**Tests:**
- Unit: segment write/read roundtrip
- Integration: checkpoint, close, reopen (fast startup)
- Crash: kill during checkpoint → recovers

---

#### 1.5 Manifest v3 with Checkpoint Metadata
**Files to modify:**
- `src/storage/manifest/manifest.h` - Extend format
- `src/storage/manifest/manifest.cc` - Parse v3

**Format v3:**
```
pomai.manifest.v3
membrane {name} shards {N} dim {D} checkpoint_seq {seq}
membrane {name} shards {N} dim {D} checkpoint_seq {seq}
```

Per-membrane manifest:
```
pomai.membrane.v2
name {name}
shards {N}
dim {D}
checkpoint_seq {seq}
wal_gen {gen}
segment segment_{shard}_{seq}.sst {min_seq} {max_seq} {num_vectors}
segment segment_{shard}_{seq}.sst {min_seq} {max_seq} {num_vectors}
```

**Tests:**
- Manifest upgrade v2 → v3
- Corruption detection (checksum mismatch)

---

#### 1.6 Crash Safety Hardening
**Files to modify:**
- `src/util/posix_file.h` - Add `fsync(fd)` + `fsync(dir_fd)`
- `src/storage/manifest/manifest.cc` - Fsync parent directory after rename
- `tests/crash/checkpoint_crash_test.cc`

**Changes:**
- After `rename(MANIFEST.tmp, MANIFEST)`: fsync parent directory
- After segment file close: fsync file + parent directory
- Add `DBOptions::paranoid_checks` flag

**Tests:**
- Crash during manifest update
- Crash during segment write
- Verify recovery always reaches last committed checkpoint

---

### Exit Criteria (Phase 1)

✅ All existing tests pass (unit, integ, TSAN, crash)
✅ New tests for WriteBatch, Snapshot, Checkpoint pass
✅ Reopen after crash recovers to last checkpoint (fast startup)
✅ WAL bounded by checkpoint rotation
✅ No data races under TSAN

---

## Phase 2: Search Routing Baseline (Fast without compression)

### Goal
Implement IVF coarse routing to avoid full shard fanout.

### Deliverables

#### 2.1 Centroid Training
**Files to create:**
- `src/core/index/ivf_trainer.h` - K-means trainer
- `src/core/index/ivf_trainer.cc`
- `tests/unit/ivf_trainer_test.cc`

**API:**
```cpp
class IvfTrainer {
public:
    Status Train(span<const float*> samples, uint32_t dim,
                 uint32_t nlist, vector<float>* centroids_out);
private:
    // Lloyd's algorithm: k-means++ init + EM iterations
};
```

**Implementation:**
- K-means++ initialization (D^2 sampling)
- EM iterations (max 100, early stop on convergence)
- Single-threaded, called at membrane creation

**Tests:**
- Convergence on synthetic Gaussian clusters
- Deterministic with fixed seed

---

#### 2.2 Centroid Persistence
**Files to modify:**
- `src/storage/manifest/manifest.h` - Store centroid file reference
- `src/core/index/ivf_coarse.h` - Add `Load(centroids_file)`
- `src/core/engine/engine.cc` - Load centroids at startup

**Format:**
```
centroids_{membrane}.bin:
    magic: "POMAIIVF" (8 bytes)
    version: uint32 = 1
    nlist: uint32
    dim: uint32
    centroids: float[nlist * dim]
    crc32: uint32
```

**Changes:**
- On membrane creation: train centroids, save to file
- On membrane open: load centroids, pass to IvfCoarse
- Fallback: if file missing, use online learning

**Tests:**
- Save/load roundtrip
- Corruption detection

---

#### 2.3 Candidate Selection with Cap
**Files to modify:**
- `src/core/index/ivf_coarse.h` - Add `max_candidates` parameter
- `src/core/shard/runtime.cc` - Enforce candidate cap

**Changes:**
```cpp
struct IvfSearchParams {
    uint32_t nprobe = 10;
    uint32_t max_candidates = 10000;  // Cap for tail latency
};
```

**Implementation:**
- Select top `nprobe` centroids
- For each centroid: gather posting list IDs
- If total > `max_candidates`: sample uniformly or take first N
- Rerank only capped candidates

**Tests:**
- Verify cap is enforced
- Latency stability under varying data distribution

---

#### 2.4 SIMD Distance Computation
**Files to create:**
- `src/core/distance/dot_product.h` - SIMD dot product
- `src/core/distance/dot_product.cc`
- `tests/unit/dot_product_test.cc`

**Implementation:**
- AVX2 path (8 floats per instruction)
- SSE path (4 floats per instruction)
- Scalar fallback
- Runtime dispatch (CPUID check)

**Tests:**
- Correctness vs scalar reference
- Alignment handling

---

#### 2.5 Deterministic Tie-Breaking
**Files to modify:**
- `src/core/engine/engine.cc` - Stable sort in `MergeTopK`

**Changes:**
```cpp
struct SearchHit {
    float score;
    VectorId id;

    bool operator<(const SearchHit& other) const {
        if (score != other.score) return score > other.score;  // Higher score first
        return id < other.id;  // Stable tie-break by ID
    }
};
```

**Tests:**
- Verify same query returns same results (order-independent of shard interleaving)

---

#### 2.6 Search Routing Metrics
**Files to create:**
- `src/util/metrics.h` - Simple counter/histogram
- `src/core/shard/runtime.cc` - Track candidates_selected, rerank_time

**Metrics:**
- `search_candidates_selected` (histogram)
- `search_rerank_time_us` (histogram)
- `search_ivf_hit_rate` (ratio: IVF vs brute force)

**Tests:**
- Verify metrics increment correctly

---

### Exit Criteria (Phase 2)

✅ Search no longer fans out to all shards by default (routing via IVF)
✅ Centroid training deterministic + tested
✅ Candidate cap enforced (tail latency bounded)
✅ SIMD distance computation tested
✅ Results deterministic (stable tie-breaking)
✅ Correctness verified vs brute force on small dataset

---

## Phase 3: Dragonfly Performance Discipline

### Goal
Achieve multi-core scaling, low tail latency, zero-copy hot paths.

### Deliverables

#### 3.1 Zero-Copy Ingest
**Files to modify:**
- `src/table/memtable.h` - Use `shared_ptr<const vector<float>>`
- `src/core/shard/commands.h` - Pass vectors by move
- `src/api/db.cc` - Move semantics in Put

**Changes:**
- API accepts `span<const float>` (zero-copy view)
- Internal copy into arena (single allocation per vector)
- WAL encode uses arena pointer (no intermediate copy)

**Tests:**
- Verify single copy (user buffer → arena)
- TSAN verify no data races

---

#### 3.2 WAL Batching & Async I/O
**Files to modify:**
- `src/storage/wal/wal.h` - Add `AppendBatch(ops)`
- `src/core/shard/runtime.cc` - Batch flushes

**Implementation:**
- Accumulate WAL records in buffer
- Single `pwrite()` call for batch (vectorized I/O)
- Single `fsync()` per batch (not per record)

**Tests:**
- Verify crash recovery with batched WAL
- Throughput improvement on bulk ingest

---

#### 3.3 Thread Pool for Search
**Files to create:**
- `src/util/thread_pool.h` - Fixed-size thread pool
- `src/util/thread_pool.cc`
- `src/core/engine/engine.cc` - Use pool instead of spawning

**Implementation:**
- Pool size = `std::thread::hardware_concurrency()`
- Work-stealing queue (optional: simple task queue)
- Reuse threads across searches

**Tests:**
- Verify pool reuse (no thread spawn per search)
- TSAN verify no data races

---

#### 3.4 Thread Pinning (Optional)
**Files to modify:**
- `include/pomai/options.h` - Add `DBOptions::pin_threads`
- `src/core/shard/shard.cc` - CPU affinity via `pthread_setaffinity_np`

**Implementation:**
- If `pin_threads == true`: pin shard thread to core `shard_id % num_cores`
- Cross-platform: Linux/Windows API

**Tests:**
- Integration test verifies affinity (read `/proc/self/status`)

---

#### 3.5 Backpressure Policy
**Files to modify:**
- `src/core/shard/mailbox.h` - Add `TryPush(timeout)`
- `include/pomai/options.h` - Add `DBOptions::mailbox_capacity`
- `src/api/db.cc` - Return `Status::kBusy` on timeout

**API:**
```cpp
struct DBOptions {
    uint32_t mailbox_capacity = 4096;
    uint32_t enqueue_timeout_ms = 5000;  // 0 = blocking
};
```

**Implementation:**
- `Mailbox::TryPush(cmd, timeout)`: block up to timeout, return false if full
- User-facing API returns `Status::Busy("Queue full")`

**Tests:**
- Fill queue, verify timeout triggers
- Verify no deadlock

---

#### 3.6 Metrics Framework
**Files to create:**
- `include/pomai/metrics.h` - Public metrics API
- `src/util/metrics.cc` - Counter, Histogram, Registry

**Metrics:**
- `pomai_ops_total{op, membrane, status}` (counter)
- `pomai_latency_seconds{op, quantile}` (histogram)
- `pomai_queue_depth{shard}` (gauge)
- `pomai_wal_bytes_written` (counter)
- `pomai_search_candidates{membrane}` (histogram)

**API:**
```cpp
class DB {
    virtual void GetMetrics(vector<Metric>* out) const = 0;
};
```

**Tests:**
- Verify metrics increment
- Histogram quantile accuracy

---

#### 3.7 Pre-allocated Result Buffers
**Files to modify:**
- `src/core/engine/engine.cc` - Reuse `SearchResult` buffers

**Implementation:**
- Thread-local `SearchResult` buffers
- Reset and reuse across searches (avoid malloc per search)

**Tests:**
- Verify no allocation in hot path (benchmark)

---

### Exit Criteria (Phase 3)

✅ Throughput scales roughly linear with core count (benchmark)
✅ Tail latency (p99) stable under load (benchmark)
✅ Zero-copy ingest verified (single copy: user → arena)
✅ Thread pool eliminates per-search thread spawn
✅ Backpressure prevents queue overflow
✅ Metrics framework in place

---

## Phase 4: Compression + Two-Stage Search (v2)

### Goal
Reduce memory footprint, improve search throughput with quantization.

### Deliverables

#### 4.1 Scalar Quantization (SQ8)
**Files to create:**
- `src/core/quantization/sq8.h` - SQ8 encoder/decoder
- `src/core/quantization/sq8.cc`
- `tests/unit/sq8_test.cc`

**Format:**
```cpp
struct SQ8Code {
    uint8_t codes[dim];  // Quantized to [0, 255]
    float min, max;      // Codebook: min/max per vector
};
```

**Implementation:**
- Per-vector min/max
- Linear quantization: `code[i] = (vec[i] - min) / (max - min) * 255`
- Approximate dot product on codes (faster, less accurate)

**Tests:**
- Roundtrip error < 1%
- Approximate distance vs exact (correlation > 0.95)

---

#### 4.2 Product Quantization (PQ) (Optional)
**Files to create:**
- `src/core/quantization/pq.h` - PQ encoder/decoder
- `src/core/quantization/pq.cc`

**Implementation:**
- Split vector into M subvectors
- K-means per subvector → codebooks
- Store codes (M bytes per vector)
- Asymmetric distance: precompute query-to-codebook distances

**Tests:**
- Recall@10 > 90% on synthetic data

---

#### 4.3 Two-Stage Search Pipeline
**Files to modify:**
- `src/core/shard/runtime.cc` - Add quantized search path
- `src/storage/segment/segment_writer.h` - Store quantized codes

**Pipeline:**
1. **Stage 1 (Fast prefilter):**
   - IVF coarse routing (nprobe centroids)
   - Load SQ8 codes for candidates
   - Approximate distance on codes
   - Select top-K' candidates (K' = 10 * K)

2. **Stage 2 (Exact rerank):**
   - Load float vectors for top-K' candidates
   - Exact dot product
   - Return top-K results

**Tests:**
- Verify recall vs brute force
- Latency improvement benchmark

---

#### 4.4 Segment File v2 with Quantized Codes
**Files to modify:**
- `src/storage/segment/segment_writer.h` - Add quantization metadata

**Format v2:**
```
Header:
    magic: "POMAI.SEG" (8 bytes)
    version: uint32 = 2
    quantization: uint8 (0=None, 1=SQ8, 2=PQ)
    ...

Data blocks:
    Block: [VectorId | seq | SQ8_code | floats]
```

**Tests:**
- Roundtrip with quantized codes
- Backward compatibility with v1

---

#### 4.5 Tuning Parameters
**Files to modify:**
- `include/pomai/options.h` - Add tuning knobs

**New options:**
```cpp
struct SearchOptions {
    uint32_t nprobe = 10;              // IVF: number of centroids to probe
    uint32_t max_candidates = 10000;   // Cap for tail latency
    uint32_t rerank_multiplier = 10;   // Stage 1: select topK * multiplier
    bool use_quantization = true;      // Enable SQ8/PQ
};
```

**Tests:**
- Verify tuning impact on recall/latency

---

#### 4.6 Recall/Precision Tests
**Files to create:**
- `tests/integ/search_recall_test.cc`

**Tests:**
- Generate synthetic dataset (Gaussian clusters)
- Compute ground truth (brute force)
- Measure recall@10, recall@100
- Verify recall > 90% with default params

---

### Exit Criteria (Phase 4)

✅ SQ8 quantization tested (error < 1%)
✅ Two-stage search tested (recall > 90%)
✅ Search speed improves 2-3x with quantization
✅ Segment file v2 with quantized codes
✅ Tuning parameters documented
✅ All tests pass (unit, integ, TSAN)

---

## Build & Test Commands

### Normal Build
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPOMAI_BUILD_TESTS=ON \
  -DPOMAI_BUILD_BENCH=ON

cmake --build build -j

ctest --test-dir build --output-on-failure
```

### TSAN Build
```bash
cmake -S . -B build-tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPOMAI_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"

cmake --build build-tsan -j

ctest --test-dir build-tsan --output-on-failure -L tsan
```

### Benchmarks
```bash
./build/bench/ingest_bench --num_vectors=1000000 --dim=768 --batch_size=1000
./build/bench/search_bench --num_queries=10000 --topk=10
```

---

## Tuning Knobs & Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shard_count` | 8 | Number of shards (should match core count) |
| `mailbox_capacity` | 4096 | Max pending commands per shard |
| `fsync_policy` | `kOnFlush` | `kNever` \| `kOnFlush` \| `kAlways` |
| `nprobe` | 10 | IVF: number of centroids to probe |
| `max_candidates` | 10000 | Cap candidates for tail latency |
| `rerank_multiplier` | 10 | Stage 1 selects topK × this |
| `checkpoint_interval` | 10000 | Auto-checkpoint every N ops |
| `pin_threads` | false | CPU affinity for shard threads |

---

## Future Roadmap

### Short-term (3-6 months)
- **HNSW graph index** (replace IVF for better recall/latency)
- **Tiered storage** (hot memtable, warm SSD, cold object storage)
- **Distributed mode** (Raft consensus, sharded cluster)
- **Filtering** (metadata predicates: `WHERE color='red'`)

### Medium-term (6-12 months)
- **GPU acceleration** (batch search offload)
- **Incremental index update** (avoid full rebuild on insert)
- **Multi-tenancy** (quotas, rate limits, isolation)
- **Backup/restore** (snapshot export/import)

### Long-term (12+ months)
- **Federated search** (multi-region, geo-sharding)
- **Semantic reranking** (hybrid sparse+dense)
- **Versioned vectors** (time-travel queries)
- **Streaming ingest** (Kafka/Kinesis integration)

---

## Notes

- **No external deps** except standard library (keep it embedded)
- **API stability**: public headers versioned (POMAI_VERSION_MAJOR)
- **Backward compat**: old segment files readable by new code
- **Test discipline**: every feature has unit + integration + TSAN tests
- **Incremental delivery**: each phase is shippable intermediate state
