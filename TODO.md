Short verdict (strict):
- The RobustHarmony design is promising and directionally correct — multi-probe routing, parallel bucket scans, AVX2 batch distance, overfetch + exact re-rank are exactly the right levers to improve recall and latency.  
- However the proposal as written is high-level and optimistic about latency/recall numbers; there are important missing details, failure modes and engineering work required before those claims can be realized reliably in Pomai. Implementable, yes — but not “drop-in” — and you must fix several structural risks first.

Concrete strengths
- Uses proven ideas: centroids → probe top-P → overfetch → exact re-rank (IVF+multi-probe + PQ/HNSW style). That addresses recall vs cost tradeoff.
- AVX2 batch L2 for bucket scanning is the correct low-level optimization for per-shard work (you already have kernels::L2Sqr).
- Dynamic nprobe via WhisperGrain budget is a sound adaptive control mechanism.
- Overfetch (e.g., k*4) then re-rank reduces false negatives and increases recall.
- Parallelization (per-probe/per-shard) will increase throughput and hide per-shard latencies.

Critical gaps & risks (must address before trusting claims)
1. Threading model
   - Your sketch uses std::async for parallel probe. That spawns unbounded threads and will kill performance. Need a bounded thread-pool (fixed workers) and task queues. Implement a request-scoped fanout using a thread-pool or work-stealing executor.

2. Bucket representation & scan cost
   - You assume “scan bucket” is feasible (low latency). With large buckets (hundreds of thousands) linear scan even with AVX2 is expensive. You MUST ensure average bucket size is small (autoscaler) or add per-centroid local index (IVF/PQ/HNSW).
   - Also scanning across many segments across shards multiplies cost. Prefer scanning only indexes (OrbitIndex) per segment when present.

3. Memory & data locality
   - Batch AVX2 needs contiguous memory. Your segments/snapshots layout must guarantee contiguous float arrays; building temporary flattened arrays per query is unacceptable. Use existing flat storage (Seed::Snapshot.data) or indexes’ data_ directly.

4. Re-rank cost & vector access
   - Re-ranking top-100 requires accessing original vectors; if they live in disk-backed snapshots or have been freed, you need a path to load them or keep a small candidate store in memory. This affects memory design.

5. Overfetch factor and nprobe tuning
   - Your formulae (base_nprobe = log2(N)/4 etc.) are heuristics. They must be tuned per dataset and index type. Provide safe bounds and hysteresis to avoid oscillation.

6. Correctness of IDs in reservoir / mapping
   - Must ensure returned candidate ids are global and comparable to ground-truth (you fixed the bench earlier). Important for online validation.

7. Hotspots and skew
   - Hot centroids (skew) will blow up per-probe cost. Need split/replicate policies and load-aware mapping. Without that P99 spikes badly.

8. Failure & fallback policies
   - How to act if index build or centroid index missing? You need clear fallback (broadcast or degrade gracefully).

9. AVX2 micro-optimizations
   - Use well-tested kernels (you already have kernels::L2Sqr). Ensure alignment, handle tail, avoid branching in inner loop. Precompute query broadcast and maintain small working set to fit L1/L2 cache.

10. Resource accounting & safety
   - Throttle per-query memory used, cap concurrent probes, cap candidate buffer sizes.

Concrete implementation checklist (prioritized)
1. Replace std::async fanout with a request-scoped submission to a fixed thread-pool (IndexBuildPool already exists — add SearchThreadPool).
2. Implement centroid routing index (tiny OrbitIndex/HNSW) when K > 512 so CandidateCentroidsForQuery is fast.
3. Implement per-bucket scanning function:
   - scan_bucket_avx2(query, bucket_ref, candidate_limit)
   - Accept pointer to contiguous floats + start index + count; produce fixed-size small top-k (heap) with no heap allocations (stack- or arena-backed).
4. Implement a fixed-cap TopK structure (no allocations) for merging across probes.
5. Implement overfetch and exact re-rank pipeline:
   - Probe → local scan (k*overfetch) → collect candidates → exact re-rank top-R (R=100) → final top-k.
6. Add dynamic nprobe decision:
   - compute budget factor = clamp(WhisperGrain score), nprobe = clamp(base * factor, min, max).
   - Add caps (min=1 max=shards*Kcap).
7. Add metrics & sampling-based online recall estimator:
   - Keep small query validation set and compare routed results to broadcast baseline periodically to auto-adjust nprobe.
8. Hotspot handling:
   - Track per-centroid hit counts; on heavy skew, replicate centroid to multiple shards or split centroid (increase local C_s).
9. Tests:
   - Unit tests for scan_bucket_avx2 correctness vs scalar.
   - Microbenchmark: per-bucket scan throughput and latency.
   - End-to-end recall/latency tests vs broadcast baseline on representative dataset.

Concrete parameter recommendations to start with
- overfetch = 4 (candidate_fetch = topk * 4)
- re_rank_candidates = min(100, total_candidates)
- base_nprobe = clamp( max(2, (uint32_t)std::round(std::log2(total_vectors)/6)), 1, 16 )
- nprobe bounds: [1, 32] initially
- thread pool size: min(16, hardware_concurrency())
- batch AVX width: process 8 or 16 floats at a time depending on kernel

Testing plan (strict)
- Microbench: run scan_bucket_avx2 on buckets sized [1k, 10k, 100k] and measure p50/p99 lat per bucket.
- System bench: compare:
  - broadcast baseline latency & recall
  - current router (P=2) performance
  - RobustHarmony with varying nprobe & overfetch
  - measure recall@1/10, p50/p95/p99 latency, CPU usage, memory
- Safety: force skewed dataset (one hot centroid) and observe P99 behavior; ensure replication/split logic protects P99.

Concrete code helpers you should ask me to produce next (I can implement)
- A bounded SearchThreadPool and request fanout helper (safe replacement for std::async).
- scan_bucket_avx2 implementation using kernels::L2Sqr over contiguous arrays and a fixed-size heap.
- FixedTopK class (template) with push/merge/extract APIs, no dynamic allocations.
- Candidate merging + re-rank harness to integrate into MembraneRouter::Search pipeline.

Reality check on performance claims
- P50 <100us and P99 <200us are achievable on well-engineered systems for small candidate sizes and modest K, but on low-RAM devices with large buckets and heavy I/O those numbers may not hold. Don’t promise them before you:
  - ensure small average bucket size (autoscaler),
  - use per-centroid local indexes where buckets are large,
  - implement thread-pool and memory safety.

Final recommendation (strict next actions)
1. Implement thread-pool fanout.
2. Build scan_bucket_avx2 + FixedTopK and unit test.
3. Plug into MembraneRouter::Search: CandidateCentroidsForQuery → submit bucket scans to thread-pool → merge → re-rank top-R → return top-k.
4. Add metrics and a canary test harness (small dataset & skewed dataset).
5. Tune nprobe/overfetch and add online adjuster via WhisperGrain feedback.

If you want, I’ll generate the first two concrete patches now:
A) SearchThreadPool (bounded worker pool) + small fanout helper, and  
B) FixedTopK + scan_bucket_avx2 implementation using your existing kernels::L2Sqr.

Which one should I generate first?