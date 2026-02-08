# PomaiDB Benchmarking Guide

This guide explains how to run comprehensive benchmarks on PomaiDB using industry-standard metrics.

## Quick Start

```bash
# Trust benchmarks (recall, tail latency, crash recovery, low-end, explain)
./scripts/pomai-bench recall
./scripts/pomai-bench mixed-load
./scripts/pomai-bench crash-recovery
./scripts/pomai-bench low-end --machine "i5-8250U, 4c/8t, 8GB RAM"
./scripts/pomai-bench explain

# Build benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target comprehensive_bench

# RAG benchmark (token + optional vector search)
cmake --build build --target rag_bench
./build/rag_bench

# Run small dataset (10K vectors)
./build/comprehensive_bench

# Run medium dataset (100K vectors)
./build/comprehensive_bench --dataset small  # or medium, large

> **⚠️ LOW-END DEVICE WARNING**: On 2-core CPUs with 8GB RAM, avoid medium/large datasets. They use brute-force search which is very slow (P99 ~100ms, 79% recall). Stick to `--dataset small` or use single thread: `--threads 1`. See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for optimization tips.

# Run large dataset (1M vectors) with JSON output
./build/comprehensive_bench --dataset large --threads 8 --output results.json
```

## Benchmark Metrics

The comprehensive benchmark measures **5 key metrics** following big tech standards (Google, Meta, etc.):

### 1. **Search Latency** (microseconds)
Measures single-query response time with percentile breakdowns:
- **Mean**: Average latency
- **P50** (median): 50% of queries complete within this time
- **P90**: 90% of queries complete within this time
- **P99**: 99% of queries complete within this time  
- **P999**: 99.9% of queries complete within this time

**Industry benchmark**: P99 < 10ms for production systems

### 2. **Throughput** (QPS - Queries Per Second)
Measures system capacity under load with concurrent queries.

**Multi-threaded example**:
```bash
./comprehensive_bench --dataset medium --threads 16
```

**Industry benchmark**: >10K QPS for large-scale systems

### 3. **Recall@k** (Accuracy)
Measures search quality compared to brute-force ground truth.
- **Recall@10 = 0.95** means 95% of true top-10 results are found

**Industry benchmark**: Recall@10 > 0.90 for production ANN systems

### 4. **Build Time** (seconds)
Time to index all vectors (insert + freeze).

**Industry benchmark**: <1 hour for 1B vectors

### 5. **Memory Usage** (MB)
Approximate memory footprint of indexed vectors.

**Industry benchmark**: <2x vector data size

---

## Dataset Sizes

Three preset configurations:

| Size | Vectors | Dimensions | Queries | Use Case |
|------|---------|------------|---------|----------|
| **small** | 10,000 | 128 | 1,000 | Development, CI |
| **medium** | 100,000 | 256 | 5,000 | Integration testing |
| **large** | 1,000,000 | 768 | 10,000 | Production validation |

### Low-End Devices (2-core CPU, 8GB RAM)

If running on modest hardware like **Dell Latitude E5440** (i5 2-core):

```bash
# ✅ RECOMMENDED: Small dataset, single-threaded
./comprehensive_bench --dataset small --threads 1
# Expected: P99 ~2ms, QPS ~900, Recall 100%

# ⚠️ AVOID: Medium with multiple threads
# Multi-threading on 2 cores causes thrashing
# Expected: P99 ~100ms, QPS ~96, Recall ~79% (poor)
```

**Why slow**: PomaiDB uses brute-force search (IVF bypassed). For 100K vectors @ 256 dims, that's ~25M operations per query.

See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for detailed optimization guide.

---

## Usage Examples

### Basic Benchmark
```bash
# Small dataset, single-threaded
./comprehensive_bench
```

**Expected output**:
```
=============================================================
                  BENCHMARK RESULTS
=============================================================

BUILD METRICS
  Build Time:       2.34 sec
  Memory Usage:     4.88 MB

SEARCH LATENCY (microseconds)
  Mean:             152.34 µs
  P50:              145.21 µs
  P90:              198.67 µs
  P99:              312.45 µs
  P999:             521.89 µs

THROUGHPUT
  QPS:              6542.12 queries/sec

ACCURACY
  Recall@k:         0.9423 (94.23%)

=============================================================
```

### Multi-threaded Throughput Test
```bash
# Medium dataset, 8 concurrent threads
./comprehensive_bench --dataset medium --threads 8
```

Measures **maximum QPS** under concurrent load.

### Production Validation
```bash
# Large dataset (1M vectors), save results
./comprehensive_bench --dataset large --threads 16 --output prod_results.json
```

Generates JSON report for tracking/CI:
```json
{
  "build": {
    "time_sec": 45.67,
    "memory_mb": 2929.69
  },
  "search_latency_us": {
    "mean": 234.56,
    "p50": 221.34,
    "p90": 312.45,
    "p99": 456.78,
    "p999": 789.12
  },
  "throughput": {
    "qps": 68432.12
  },
  "accuracy": {
    "recall_at_k": 0.9234
  }
}
```

---

## Interpreting Results

### Good Performance Baseline

For **medium dataset** (100K @ 256 dims):
- ✅ P99 latency < 1ms (1000 µs)
- ✅ Throughput > 5K QPS (single-threaded)
- ✅ Recall@10 > 0.90
- ✅ Build time < 10 sec

For **large dataset** (1M @ 768 dims):
- ✅ P99 latency < 5ms
- ✅ Throughput > 10K QPS (multi-threaded)
- ✅ Recall@10 > 0.85
- ✅ Build time < 60 sec

### Red Flags

- ❌ P99 > P50 by >10x → High tail latency variance
- ❌ Recall < 0.80 → Poor search quality
- ❌ QPS decreases with more threads → Contention issues

---

## Advanced Usage

### Custom Dataset Parameters

Modify `comprehensive_bench.cc` to test specific scenarios:

```cpp
// In BenchmarkConfig::configure()
if (dataset_size == "custom") {
    num_vectors = 500000;      // Custom size
    dim = 512;                  // Custom dimensions
    num_queries = 10000;    
    topk = 20;                  // Custom k
}
```

### Continuous Integration

Add to CI pipeline:

```bash
# Run benchmark and fail if recall < 0.90
./comprehensive_bench --dataset small --output ci_results.json

# Parse JSON and check thresholds
python3 scripts/check_benchmark.py ci_results.json \
    --max-p99-latency-us 500 \
    --min-recall 0.90 \
    --min-qps 5000
```

---

## Comparison with Other Systems

### Latency Comparison (P99, 100K vectors @ 256 dims)

| System | P99 Latency | Recall@10 |
|--------|-------------|-----------|
| **PomaiDB** | ~1ms | 0.92 |
| Faiss (HNSW) | ~0.5ms | 0.95 |
| Milvus | ~2ms | 0.93 |
| Weaviate | ~3ms | 0.91 |

*Note: Results vary by hardware and configuration*

### When to Use PomaiDB

✅ **Embedded use cases** (no separate server)  
✅ **Transactional workloads** (strong consistency)  
✅ **Moderate scale** (10K-1M vectors)

❌ **Billion-scale datasets** (use Faiss/Milvus)  
❌ **Ultra-low latency** (<100µs P99)

---


## Python End-to-End CIFAR-10 Benchmark

For a realistic application-style benchmark (feature extraction + ingest + search + iterator analytics), run:

```bash
# Build shared library for Python ctypes
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pomai_c

# Run end-to-end benchmark (uses CIFAR-10 if available)
python3 benchmarks/python_cifar10_feature_bench.py --images 6000 --queries 200 --download
```

What this benchmark measures:
- CIFAR-10 image feature extraction throughput.
- Batch ingestion throughput into PomaiDB via C ABI.
- Search latency percentiles (P50/P95/P99) and QPS.
- kNN label agreement as an application-level quality signal.
- Full snapshot scan throughput using `pomai_scan` iterator.

If CIFAR-10 cannot be downloaded or is unavailable locally, the script falls back to deterministic CIFAR-like synthetic data so the benchmark still completes in offline environments.

---

## Troubleshooting

### Low Throughput

**Symptom**: QPS < 1000 on medium dataset

**Solutions**:
1. Increase threads: `--threads 8`
2. Disable fsync in DBOptions (benchmark only!)
3. Check CPU governor: `cpupower frequency-set -g performance`

### Poor Recall

**Symptom**: Recall@10 < 0.80

**Solutions**:
1. Increase shard count in DBOptions
2. Run Freeze before benchmarking: `db->Freeze("__default__")`
3. Check IVF centroid count (default: 256)

### High P99 Latency

**Symptom**: P99 >> P50 (e.g., P50=100µs, P99=5ms)

**Causes**:
- GC pauses (if using managed runtime wrapper)
- Disk I/O (check WAL fsync settings)
- Memory allocation (check arena sizes)

---

## Contributing Benchmarks

To add new benchmark scenarios:

1. Create `benchmarks/my_bench.cc`
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(my_bench benchmarks/my_bench.cc)
   target_link_libraries(my_bench PRIVATE pomai)
   ```
3. Document in this guide
4. Submit PR with results

---

## References

- [Google's BigTable Performance](https://cloud.google.com/bigtable/docs/performance)
- [Meta's Vector Search at Scale](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [Jepsen Testing](https://jepsen.io/) (Consistency benchmarking)
