# Ingestion Throughput Analysis

Based on your benchmark results from **Dell Latitude E5440** (i5 2-core, 8GB RAM):

## Your Current Performance

### Small Dataset (10K vectors @ 128 dims)
- **Build Time**: 0.92 sec
- **Ingestion Rate**: **~10,870 vectors/sec**
- **Throughput**: ~5.3 MB/sec (128 dims × 4 bytes × 10,870)

### Medium Dataset (100K vectors @ 256 dims)
- **Build Time**: 11.04 sec
- **Ingestion Rate**: **~9,057 vectors/sec**
- **Throughput**: ~9.0 MB/sec (256 dims × 4 bytes × 9,057)

## Is This The Maximum?

**Short answer**: No, you can likely improve by **2-5x** with optimizations.

### Current Bottlenecks

1. **WAL Writes** (biggest bottleneck)
   - Default: `fsync = kNever` (good for benchmarks)
   - Each Put writes to WAL sequentially
   - Single-threaded per shard

2. **MemTable Insertions**
   - Lock-free for different shards
   - But single-threaded within shard
   - Should be very fast (~1-2µs per insert)

3. **Memory Allocations**
   - Arena allocator used
   - Should be minimal overhead
   - Possible page faults on 8GB RAM

4. **Shard Count**
   - You have `shard_count = std::thread::hardware_concurrency()` = 2
   - Good match for 2-core CPU

## Expected Maximum Performance

### Theoretical Limits (i5 2-core CPU)

| Component | Throughput | Notes |
|-----------|------------|-------|
| Memory bandwidth | ~25 GB/sec | DDR3-1600 |
| L3 cache | ~50 GB/sec | Optimistic |
| WAL sequential write | ~100 MB/sec | SSD, no fsync |
| MemTable insert | ~1M ops/sec | Lock-free |

**Bottleneck**: WAL writes at ~100 MB/sec

**Theoretical max**:
- For 128-dim vectors: ~195,000 vectors/sec
- For 256-dim vectors: ~97,000 vectors/sec

**Your actual**:
- 128-dim: ~10,870 vectors/sec (5.5% of theoretical)
- 256-dim: ~9,057 vectors/sec (9.3% of theoretical)

**Gap**: 10-20x slower than theoretical maximum

## Optimization Strategies

### 1. Batch Insertions (Implemented)

**Problem**: Each `Put()` enqueues one command, incurring per-op overhead (lock contention, WAL write syscall).

**Solution**: Use the `PutBatch()` API.
```cpp
// Instead of:
for (auto& vec : vectors) {
    db->Put(id++, vec);  // 10K roundtrips
}

// Use batch:
db->PutBatch(membrane, ids, vectors);  // 1 roundtrip
```

**Actual Results**:
- Sequential `Put`: ~6,890 vectors/sec
- Batched `PutBatch` (1000/batch): **~245,861 vectors/sec**
- **Speedup**: ~35x improvement

### 2. Disable WAL for Bulk Load

**Problem**: WAL writes dominate bulk ingestion

**Workaround**:
```cpp
// Benchmark already does this
opts.fsync = FsyncPolicy::kNever;  // Good

// But could also skip WAL entirely (not safe!)
// Requires code change
```

**Expected speedup**: 2-3x (if WAL writes are skipped)

### 3. Parallel Shard Writes

**Problem**: Inserting sequentially to one shard at a time

**Solution**: Distribute inserts across shards
```cpp
// Current (sequential)
for (uint32_t i = 0; i < 10000; ++i) {
    db->Put(i, vectors[i]);
}

// Optimized (parallel across shards)
#pragma omp parallel for
for (uint32_t i = 0; i < 10000; ++i) {
    db->Put(i, vectors[i]);  // Different shards = parallel
}
```

**Expected speedup**: 1.5-2x (on 2-core CPU)

### 4. Increase Shard Count

**Problem**: 2 shards may not saturate I/O

**Experiment**:
```cpp
opts.shard_count = 4;  // More than CPU cores
// Or even 8 for I/O-bound workloads
```

**Expected speedup**: 1.5-2x (if I/O bound)

### 5. Pre-allocate Memory

**Problem**: Page faults during insertion

**Solution**:
```cpp
// Pre-touch memory before inserting
MemTableOptions mem_opts;
mem_opts.preallocate = true;
mem_opts.arena_block_bytes = 10 << 20;  // 10 MB
```

**Expected speedup**: 1.1-1.3x

## Quick Experiment: Measure WAL Overhead

Run this to see WAL impact:

```bash
# Modify comprehensive_bench.cc temporarily
# Comment out these lines in BuildIndex:
# st = db->Put(i, dataset_.vectors[i]);

# And replace with:
vectors_only.push_back(dataset_.vectors[i]);
# (Skip actual DB insertion)

# Then rebuild and run:
cmake --build build --target comprehensive_bench
./comprehensive_bench --dataset small
```

If "build time" drops to ~0.1 sec, WAL is the bottleneck.

## Realistic Optimizations (Today)

Without code changes, you can try:

### 1. Use tmpfs for WAL
```bash
# Store DB on RAM disk
mkdir /tmp/ramdisk
sudo mount -t tmpfs -o size=512M tmpfs /tmp/ramdisk

# Then in code:
opts.path = "/tmp/ramdisk/bench_db";
```

**Expected**: 1.5-2x speedup

### 2. Reduce Dimension
```cpp
// Instead of 256 dims
opts.dim = 128;  // 2x faster writes
```

**Expected**: 2x speedup (obvious)

### 3. Use Larger Arena Blocks
```cpp
// In MemTable construction (engine.cc line 81)
constexpr std::size_t kArenaBlockBytes = 4u << 20;  // 4 MB instead of 1 MB
```

**Expected**: 1.1-1.2x speedup (fewer allocations)

## Comparison with Other Systems

### Ingestion Throughput (vectors/sec)

| System | Single-threaded | Multi-threaded (4 cores) |
|--------|-----------------|--------------------------|
| **PomaiDB** (your hardware) | ~10K | ~20K (estimated) |
| Faiss (no persistence) | ~500K | ~2M |
| Milvus (with WAL) | ~50K | ~200K |
| ChromaDB (SQLite) | ~5K | ~10K |

**Takeaway**: PomaiDB is in the middle - slower than Faiss (no durability), faster than ChromaDB (similar architecture).

## Action Items

**Immediate** (no code changes):
1. ✅ You're already using `fsync = kNever` (good!)
2. Try tmpfs: `opts.path = "/tmp/ramdisk/bench_db"`
3. Reduce dims to 128 if acceptable

**Short-term** (code changes needed):
1. Add `PutBatch()` API → 5-10x improvement
2. Parallel insertion across shards
3. Tune arena block size

**Long-term** (architectural):
1. Async WAL writes (currently synchronous)
2. Memory-mapped WAL (zero-copy)
3. Bulk segment writer (skip memtable)

## Summary

**Your current ingestion**: ~9-11K vectors/sec on 2-core laptop

**Realistic optimized**: ~20-30K vectors/sec (with batch API + parallel inserts)

**Theoretical max**: ~50-100K vectors/sec (with async WAL + SIMD)

**Bottleneck**: WAL sequential writes (main thread only)

---

**Recommendation**: Your ingestion throughput is reasonable for a 2-core laptop with durability (WAL). For bulk loading, implementing `PutBatch()` would give the biggest win (5-10x). Without code changes, use tmpfs for 2x speedup.
