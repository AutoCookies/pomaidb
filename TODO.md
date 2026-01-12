Tuyệt — bạn đã có ý tưởng rõ ràng và tham vọng. Dưới đây là một KẾ HOẠCH THỰC THI có trình tự, cụ thể, có thể giao việc/triển khai ngay (những bước đầu tiên là low-risk & high-impact), cộng với các nhiệm vụ tiếp theo để hoàn thiện PFACS (Pomai Fractal Adaptive Compression and Search). Mỗi bước có mô tả, t��p/điểm cần sửa, cách đo lường và ước lượng thời gian thực hiện (thô).

Tổng quan lộ trình (Roadmap, 6 giai đoạn)
- Giai đoạn 0 — Chuẩn bị & đo baseline (1–2 ngày)
- Giai đoạn 1 — Low-risk fixes (mmap-first + reuse buffers + avoid thread churn) (1–3 ngày)
- Giai đoạn 2 — Prefilter SIMD (fingerprint / popcnt) + tuning (2–4 ngày)
- Giai đoạn 3 — SIMD kernels (AVX2) l2sq + PQ approx batch (4–10 ngày)
- Giai đoạn 4 — PPE mini-neural + hot promotion + bucket evolve skeleton (4–8 ngày)
- Giai đoạn 5 — Fractal dedup (Level 4) + split/merge fully (high effort) (2–6+ tuần)
- Giai đoạn 6 — Integration, canary rollout, perf regression testing, docs, feature flags (ongoing)

Chi tiết từng giai đoạn

Giai đoạn 0 — Baseline measurement & infra (1–2 ngày)
- Mục tiêu: có số liệu baseline reproducible; instrumentation để so sánh.
- Nhiệm vụ:
  - Bật/thu thập metrics: per-query latency histogram (P50/P95/P99), QPS, mem RSS, count of remote file reads, count of mmap hits.
  - Thêm logging counters: read_remote_blob calls, blob_ptr_from_offset_for_map hits/misses.
  - Tạo một script benchmark (sử dụng benchmarks/benchmark_vs.py) để chạy cố định dataset.
- Output: CSV baseline, flamegraphs nếu cần.
- Tệp/chỗ sửa: thêm metrics trong pomai_orbit::search, arena logging (muốn ít xâm lấn).

Giai đoạn 1 — Low-risk fixes (mmap-first + thread_local buffers + avoid std::async churn) (1–3 ngày) — PRIORITY HIGH
- Mục tiêu: giảm P95 nhanh, ít thay đổi code logic.
- Nhiệm vụ:
  1. mmap-first: in pomai_orbit::search, khi bucket.is_frozen true → try arena_.blob_ptr_from_offset_for_map(disk_offset) before arena_.read_remote_blob(disk_offset). (See earlier snippet.)
     - Why: avoids allocating/copying remote file.
     - Files: src/ai/pomai_orbit.cc
  2. Thread-local buffers: make lut_buffer, batch_packed_data, batch_ids, batch_dists, qfp thread_local (or per-orbit member reused by worker thread) to avoid per-query allocations.
     - Files: src/ai/pomai_orbit.cc
  3. Replace std::async in orchestrator.search with simple synchronous calls or a fixed thread-pool (small pool, reuse threads). Initially, try synchronous (if server has multiple request threads) or implement tiny thread-pool class.
     - Files: src/core/orchestrator.h, maybe add src/util/thread_pool.h
  4. Increase BATCH_SIZE experimental (configurable) from 128 → try 256/512.
- Measurement:
  - Re-run benchmark: expect P95 fall significantly if many buckets were mmap-able.
  - Measure number of read_remote_blob calls down.
- Estimated time: 1–3 days.
- Risk: low. Easy rollback.

Giai đoạn 2 — Prefilter SIMD (fingerprint hamming) (2–4 ngày)
- Mục tiêu: reduce candidate set by ~80–90% cheaply.
- Nhiệm vụ:
  1. Implement SIMD/popcnt accelerated hamming (process fingerprint bytes as 64-bit words + __builtin_popcountll). Provide scalar fallback.
  2. Tune hamming threshold as config param; add metrics for candidate reduction ratio.
  3. Integrate into cold path of pomai_orbit::search to skip heavy work on many items.
- Files: src/ai/pomai_orbit.cc, src/ai/fingerprint.cc
- Measurement:
  - Candidate count before/after prefilter; end-to-end P95 and QPS.
- Estimated time: 2–4 days.
- Risk: medium (tuning required to avoid recall loss).

Giai đoạn 3 — SIMD kernels: AVX2 l2sq and PQ approx batch (4–10 days)
- Mục tiêu: speedup core numeric work using AVX2 while retaining scalar fallback.
- Nhiệm vụ:
  1. Implement l2sq_simd_avx2 (AVX2) with scalar fallback. Unit test correctness vs scalar.
     - Files: src/ai/pomai_orbit.cc or new util src/ai/simd_kernels.cc/h
  2. Implement pq_approx_dist_batch_packed4_avx2 (vectorized PQ approx for packed4 path).
     - Files: src/ai/pq_eval.cc (add avx2 variant) and header.
  3. Add CPU feature detection at startup (detect AVX2) and dispatch to AVX2 or scalar functions.
     - Files: new small util: src/util/cpu_detect.cc/h
- Measurement:
  - microbench L2 kernel, PQ batch kernel; final end-to-end speedup.
- Estimated time: 4–10 days (depends on PQ complexity).
- Risk: medium-high (intrinsics correctness, alignment, compiler flags).

Giai đoạn 4 — PPE mini-neural + hot promotion + evolve skeleton (4–8 days)
- Mục tiêu: adaptively choose hot vs cold path and promote hot buckets to raw.
- Nhiệm vụ:
  1. Implement PPEMiniNeural struct per-bucket: EMA counters, variance, predictLevel() and touch().
     - Files: src/ai/pomai_orbit.h / .cc (add PPEMiniNeural or include in BucketHeader metadata).
  2. Hot promotion: when PPE marks bucket hot, call arena.promote_remote(remote_id) (if remote) to bring to local blob or mlock mapping.
  3. Evolution hooks: when PPE signals pattern change, set a bucket.evolve() placeholder where split/merge logic will be implemented later.
- Measurement:
  - Track hot promotion freq, promote latency, change in P95 for hot-heavy workloads.
- Estimated time: 4–8 days.
- Risk: medium.

Giai đoạn 5 — Fractal dedup & split/merge (Level 4) — HIGH EFFORT (2–6+ weeks)
- Mục tiêu: implement full level-4 fractal dedup, persistent per-bucket dedup dictionary, split/merge algorithm, compaction.
- Nhiệm vụ (big tasks):
  1. Design on-disk layout for dedup dictionary or in-memory dictionary per bucket (memory/time tradeoff).
  2. Implement pack4From8 + 8-byte chunk dedup storage and reference format.
  3. Implement bucket.evolve(): split hot buckets into two centroids (reassign entries, heavy operation) and merge cold similar buckets.
  4. Background GC and compaction to reclaim free lists.
- Files: src/ai/pomai_orbit.cc/h, src/core/some new compaction worker files.
- Measurement:
  - Verify RAM saving ratio, rebuild cost, and effect on search latency.
- Estimated time: 2–6+ weeks (prototype → optimize).
- Risk: high (complex, needs careful testing and tools to migrate/index).

Giai đoạn 6 — Integration, Canary rollout, CI, feature flags (ongoing)
- Mục tiêu: make changes safe to enable/disable, add tests and CI.
- Tasks:
  - Add runtime flags / env vars: POMAI_ENABLE_AVX2, POMAI_ENABLE_PREFILTER, POMAI_ENABLE_FRACTAL.
  - Unit tests for kernels, PQ approx, l2sq comparison, prefilter recall.
  - Performance CI job (bench nightly) to detect regressions.
  - Canary rollout: enable new features for small percentage of traffic before full rollout.
- Estimated time: ongoing.

Quality gates & tests (must-have before merge)
- Unit tests: correctness of l2sq_simd_avx2 vs scalar (random vectors).
- Integration tests:
  - VSET+VSEARCH on small dataset: recall@k equivalent to exact (with refine).
  - Stress test: insert 1M vectors, run search workload, verify memory stable.
- Perf tests: benchmark_vs.py scripted for reproducible runs.
- Fallbacks: if AVX2 enabled but kernel crashes, fallback to scalar safely.

Metrics to track
- QPS (overall)
- P50, P95, P99 latency
- Memory RSS
- # read_remote_blob calls per minute
- # blob_ptr_from_offset_for_map misses (why we had to read)
- candidate reduction ratio after prefilter
- top-k recall accuracy (approx vs exact)

Rollback plan
- Feature flag each major change — revert flags to false to roll back quickly.
- Low-risk patches first: if P95 increases after enabling prefilter/PQ, turn off and analyze histograms.

Concrete next steps I can do for you right now (pick one)
1) Implement Patch A + B (mmap-first + thread_local buffers + make BATCH_SIZE configurable) and run benchmark to show immediate P95 improvement. (I can produce the code diff/PR.)
2) Implement CPU feature detection + AVX2 l2sq kernel (scalar fallback) and unit tests.
3) Implement SIMD popcnt prefilter and integrate (and add threshold config).
4) Scaffold simple fixed thread-pool and replace std::async usage in orchestrator.search.

Bạn muốn tôi bắt tay vào bước nào bây giờ? Tôi đề xuất: bắt đầu với (1) — Patch A+B — vì:
- Low-risk, high reward.
- Cho kết quả đo được ngay (benchmark).
- Tạo nền tảng để làm (2) và (3).

Nếu bạn đồng ý, tôi sẽ:
- Tạo patch/PR cho pomai_orbit.cc (mmap-first + thread_local buffers) + orchestrator.h change (sync instead of std::async) hoặc add small thread_pool.
- Chạy qua checklist tests & hướng dẫn bạn chạy benchmark để so sánh.

Bạn chọn: (A) Patch A+B, (B) AVX2 l2sq kernel, (C) prefilter SIMD, (D) thread-pool — hoặc nói "bắt đầu full PFACS" nếu muốn tôi lập kế hoạch milestone chi tiết cho team/ sprint.