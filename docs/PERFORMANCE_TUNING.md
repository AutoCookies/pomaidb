# Performance Tuning Guide

This guide helps optimize PomaiDB for different hardware configurations, especially low-end devices.

## Current Performance Limitations

> **⚠️ IMPORTANT**: PomaiDB currently uses **brute-force search** (IVF index is bypassed for correctness). This has serious performance implications:

| Dataset Size | Hardware Impact |
|--------------|-----------------|
| < 10K vectors | ✅ Works well on all hardware |
| 10K - 50K | ⚠️ Acceptable on modern CPUs |
| 50K - 100K | ❌ Slow on low-end devices |
| > 100K | ❌ Not recommended without IVF |

**Why it's slow**: Each query compares against ALL vectors in brute-force mode. For 100K vectors @ 256 dims, that's ~25 million float operations per query.

---

## Hardware Recommendations

### Minimum Specs (10K vectors)
- CPU: 2-core @ 2.5 GHz
- RAM: 4 GB
- Expected: P99 < 5ms, ~500 QPS

### Recommended Specs (100K vectors)
- CPU: 4-core @ 3.0 GHz
- RAM: 8 GB
- Expected: P99 < 50ms, ~100 QPS (brute-force)

### High Performance (1M vectors)
- CPU: 8+ cores @ 3.5 GHz
- RAM: 16+ GB
- Expected: P99 < 500ms, ~50 QPS (brute-force)

> **Note**: Once IVF is enabled, expect 10-100x speedup

---

## Benchmarking on Low-End Devices

### Use Appropriate Dataset Sizes

For a **Dell Latitude E5440** (i5 2-core, 8GB RAM):

```bash
# ✅ GOOD: Small dataset
./comprehensive_bench --dataset small --threads 1
# Expected: P99 ~2ms, QPS ~900, Recall 100%

# ⚠️ MARGINAL: Medium with single thread
./comprehensive_bench --dataset medium --threads 1
# Expected: P99 ~50ms, QPS ~50-100, Recall ~80%

# ❌ BAD: Medium with 4 threads on 2 cores
./comprehensive_bench --dataset medium --threads 4
# Causes context switching overhead, worse than single-threaded
```

**Rule of thumb**: On 2-core CPUs, use `--threads 2` max to avoid thrashing.

### Create Custom Benchmarks

Add to `comprehensive_bench.cc`:

```cpp
// In BenchmarkConfig::configure()
if (dataset_size == "tiny") {
    num_vectors = 5000;
    dim = 64;
    num_queries = 500;
}
```

Rebuild:
```bash
cmake --build build --target comprehensive_bench
./comprehensive_bench --dataset tiny  # Faster on low-end devices
```

---

## Optimization Checklist

### 1. Reduce Thread Count

**Problem**: More threads than CPU cores causes context switching.

**Fix**:
```bash
# Bad: 4 threads on 2-core CPU
./comprehensive_bench --threads 4  # Slow!

# Good: Match CPU cores
./comprehensive_bench --threads 2  # Better
```

### 2. Reduce Dataset Size

**Problem**: Large datasets cause memory pressure and cache misses.

**Fix** (in your application):
```cpp
pomai::DBOptions opts;
opts.dim = 128;  // Use 128 instead of 768
opts.shard_count = 2;  // Match CPU cores
```

### 3. Disable Logging (if added)

**Problem**: I/O overhead during benchmarking.

**Fix**: Ensure no debug logging in release builds.

### 4. Use Release Build

**Problem**: Debug builds are 5-10x slower.

**Fix**:
```bash
# Always use Release for benchmarking
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

### 5. CPU Governor

**Problem**: CPU frequency scaling throttles performance.

**Fix** (Linux):
```bash
# Check current
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance mode
sudo cpupower frequency-set -g performance

# Revert after benchmarking
sudo cpupower frequency-set -g powersave
```

### 6. Reduce Vector Dimensions

**Problem**: 768-dim vectors (e.g., BERT embeddings) are expensive.

**Fix**: Use PCA/dimensionality reduction:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=128)  # 768 → 128
reduced_vecs = pca.fit_transform(vectors)
```

---

## Expected Performance by Hardware

### Example: Dell Latitude E5440 (i5-4300U, 2-core, 8GB)

| Dataset | Threads | P99 | QPS | Recall | Rating |
|---------|---------|-----|-----|--------|--------|
| Small (10K @ 128) | 1 | 2ms | 900 | 100% | ✅ Excellent |
| Small (10K @ 128) | 2 | 3ms | 1400 | 95% | ✅ Good |
| Medium (100K @ 256) | 1 | 50ms | 80 | 80% | ⚠️ Marginal |
| Medium (100K @ 256) | 4 | 100ms | 96 | 79% | ❌ Poor (thrashing) |

**Recommendation**: Stick to small datasets (<10K) or wait for IVF optimization.

### Example: Modern Desktop (i7-12700K, 12-core, 32GB)

| Dataset | Threads | P99 | QPS | Recall | Rating |
|---------|---------|-----|-----|--------|--------|
| Small (10K @ 128) | 8 | 0.5ms | 15K | 100% | ✅ Excellent |
| Medium (100K @ 256) | 8 | 10ms | 800 | 95% | ✅ Very Good |
| Large (1M @ 768) | 16 | 200ms | 150 | 90% | ✅ Good |

---

## Troubleshooting Slow Performance

### Symptom: P999 >> P99 (e.g., P99=50ms, P999=2500ms)

**Cause**: Memory allocations, page faults, or GC (if using language bindings)

**Solutions**:
1. Reduce concurrent threads
2. Pre-allocate memory arenas
3. Check `dmesg` for OOM killer activity

### Symptom: QPS decreases with more threads

**Cause**: Lock contention or context switching

**Solutions**:
1. Use `--threads N` where N = CPU core count
2. Check thread count: `lscpu | grep "CPU(s)"`
3. Monitor with `htop` during benchmark

### Symptom: Low recall (<80%)

**Cause**: Incorrect centroid configuration or numerical instability

**Solutions**:
1. Run `db->Freeze()` before benchmarking
2. Check `ivf_->SetCentroidCount()` (default 256)
3. Verify vectors are normalized

### Symptom: Build time > 60 sec for 100K vectors

**Cause**: Slow disk I/O or WAL fsync

**Solutions**:
1. Benchmark only: `opts.fsync = FsyncPolicy::kNever`
2. Use SSD instead of HDD
3. Use tmpfs: `opts.path = "/tmp/pomai_bench"`

---

## Optimizations Roadmap

These will improve performance in future releases:

### Short-term (Planned)
- ✅ Snapshot-based reads (implemented)
- ⏳ **Enable IVF search** (10-100x speedup)
- ⏳ SIMD distance calculations
- ⏳ Batch search API

### Medium-term
- ⏳ Product quantization (reduce memory)
- ⏳ HNSW graph index (faster than IVF)
- ⏳ GPU acceleration

### Long-term
- ⏳ Distributed sharding
- ⏳ Approximate top-k early termination

---

## When NOT to Use PomaiDB

Based on current limitations, avoid PomaiDB if:

❌ **Billion-scale datasets** (use Faiss/Milvus)  
❌ **Ultra-low latency** (<100µs P99 required)  
❌ **Low-end embedded devices** (Raspberry Pi, mobile)  
❌ **Real-time updates + search** (100K+ vectors)

---

## Recommended Alternatives

For specific use cases:

| Use Case | Recommended |
|----------|-------------|
| Billion vectors, cloud | **Milvus**, **Pinecone** |
| Ultra-low latency | **Faiss** (HNSW) |
| Embedded, <10K vectors | **PomaiDB** ✅ |
| Python simplicity | **ChromaDB**, **LanceDB** |
| On-device mobile | **ONNX Runtime**, **TFLite** |

---

## Benchmark Interpretation

### Good Results ✅

```
SEARCH LATENCY:
  P50:  1ms
  P99:  5ms
  P999: 20ms  ← Within 10x of P50

THROUGHPUT:
  QPS: 500+ (single-threaded)

ACCURACY:
  Recall@10: >90%
```

### Bad Results ❌

```
SEARCH LATENCY:
  P50:  50ms
  P99:  100ms
  P999: 2500ms  ← 50x worse than P50 (indicates issues)

THROUGHPUT:
  QPS: <100 (on modern CPU)

ACCURACY:
  Recall@10: <80%
```

**If you see bad results**: Reduce dataset size or wait for IVF optimization.

---

## Contributing Performance Improvements

If you implement optimizations:

1. Run before/after benchmarks
2. Document bottleneck analysis
3. Submit PR with results comparison
4. Update this guide

Example:
```bash
# Before optimization
./comprehensive_bench --dataset medium > before.txt

# After optimization
./comprehensive_bench --dataset medium > after.txt

# Compare
diff before.txt after.txt
```

---

## FAQ

**Q: Why is PomaiDB slower than Faiss?**  
A: Currently using brute-force search. IVF is implemented but not enabled in read path. Once enabled, expect 10-100x speedup.

**Q: Is PomaiDB suitable for production?**  
A: Yes, for small-medium datasets (<50K vectors) on modern hardware. For larger datasets, wait for IVF optimization or use Faiss/Milvus.

**Q: Can I use PomaiDB on Raspberry Pi?**  
A: Not recommended for >1K vectors. Too slow without SIMD/IVF optimizations.

**Q: How to improve recall?**  
A: Run `db->Freeze()` before search, increase shard count, or wait for IVF tuning improvements.
