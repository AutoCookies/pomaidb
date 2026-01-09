Dưới đây là checklist triển khai và kế hoạch benchmark chi tiết (theo dạng bước‑bước, dễ thực hiện và lặp lại). Mục tiêu rõ: chứng minh Pomai‑style SoA+Fingerprint+PQ (với PPE demoter/promoter + mmap) cho N≈1M, D=768 trên máy yếu (2–4 core, 8GB) đạt footprint < ~1GB và latency truy vấn top-K (K=10) P50/P95 trong giới hạn mục tiêu (<=50ms).  

Mình chia làm hai phần: (A) Checklist triển khai (kỹ thuật & vận hành) và (B) Benchmark chi tiết (kịch bản, metric, cấu hình, công cụ, ma trận tham số và mẫu báo cáo).

A. Checklist triển khai (step-by-step)
1. Thiết kế dữ liệu & file format (SoA)
   - [x] Định nghĩa header file-mmap: magic, version, N, D, m, k, codebook offsets, sizes.
   - [x] SoA arrays trong file / RAM:
     - bitpacked fingerprint (SimHash/OPQ-sign): 1 bit/dim hoặc 512-bit SimHash (size ~D/8).
     - PQ codes: m bytes per vector (8-bit per subquant); khả năng packed 4-bit on-disk.
     - ids/offsets: uint64 per vector.
     - PPE predictor array: small struct per vector (hit count, last_touch_ms).
   - [x] codebooks table (m * k * subdim floats) resident in RAM.
   - [x] Arena/backing store: file-backed blob store cho full floats (when hot), offsets saved in ids/offsets.

2. Core algorithm components
   - [x] Fingerprint encoder: SimHash (random proj) or OPQ-sign (if OPQ available).
   - [x] PQ trainer (offline): kmeans per-subvector, save codebooks to file.
   - [x] PQ encoder: encodeVec -> m bytes per vec.
   - [x] PQ approximate distance evaluator: per-query precompute m tables (k floats), then table-sum per-candidate (SIMD-friendly).
   - [x] Binary prefilter: block streaming XOR + popcount (AVX2/POPCNT) vectorized implementation.
   - [x] Candidate collector & ranking: maintain small heap for top candidates.
   - [x] Final refine: exact L2 / inner-product on top-N, fetching full floats from arena when necessary.
   - [x] PQ & prefilter performance follow-ups (tunable, optional):
     - [x] vectorize pq_eval (SIMD gather or blocked sum) if profiling shows PQ evaluation is a bottleneck.
     - [x] micro-optimize prefilter: PSHUFB table popcount / AVX512 path (optional, guarded by runtime CPU checks).
     - [x] Ensure scalar fallbacks exist so code runs on weak machines without AVX2.

3. Storage & IO management
   - [x] mmap file manager: preallocate file, page-align, expose read/write wrapper, thread-safe IO mutex.
   - [x] Packed-on-disk format for demoted PQ (4-bit): pack/unpack helpers.
   - [x] Atomic append/update semantics for SoA arrays; write-ahead log if durability required.
   - [x] madvise/mmap flags & optional mlock support for pinning hot pages.
   - [ ] Atomic/concurrent access guarantees:
     - [x] Use std::atomic<uint64_t> or std::atomic_ref<uint64_t> (where available) for ids_block updates/reads to avoid torn reads while demote/promote occur.
     - [x] Audit call sites that update ids payloads and PPE flags; use atomic CAS when combining flag updates.

4. PPE predictor & background worker
   - [x] PPEPredictor per-vector: last_access_ms, hits (atomic or relaxed) + touch() and predictNext().
   - [x] Demoter/promoter thread: runs periodically, decides moves inline↔arena↔mmap, uses PPEPredictor predictions, logs actions.
   - [x] Parameterizable thresholds: hot_size_limit, promote_lookahead_ms, demote_threshold_ms.
   - [x] Async demotion architecture (recommended):
     - [x] Implement demote_blob_async() + bounded queue + DemoteWorker (batch append to segment files).
     - [x] Backpressure policy (max_pending_demotes + sync fallback).
     - [x] Remote id encoding by segment (segment_id<<32 | offset) and segment rotation logic.
     - [x] Expose demote metrics (queue length, tasks completed, bytes written, latency).

5. API / runtime
   - [ ] Ingest API: addVec(id, float*vec) -> encodes fingerprint + PQ + optionally store full float if hot, update PPE.
   - [ ] Search API: search(query, topK) -> does multi-stage pipeline (fingerprint -> prefilter -> pq_eval -> collect -> refine).
   - [ ] Update/remove APIs: safe handling for SoA and arena offsets.
   - [ ] Telemetry endpoints: expose metrics (latency histograms, memory RSS, pagefaults).
   - [ ] Integration & pipeline:
     - [ ] Implement end-to-end pipeline function with conservative defaults and knobs (so it can be tuned for weak machines).

6. Instrumentation & logging
   - [ ] Metrics: QPS, P50/P95/P99 latency, candidate counts after prefilter, PQ-sum time, refine time, page faults, IO throughput, memory RSS.
   - [ ] Use tools/libraries: Prometheus endpoints or simple text logs for benchmark.
   - [ ] Profiling hooks: optional perf events, heap snapshots.
   - [ ] Micro-benchmark harnesses for each stage (fingerprint, prefilter, pq_eval, refine, arena IO).

7. Tests & CI
   - [ ] Unit tests: PQ encode/decode, pack4/unpack, fingerprint, arena read/write, mmap consistency.
   - [ ] Integration tests: ingest 100k synthetic vectors + queries validate recall vs brute-force.
   - [ ] Regression tests for demoter/promoter correctness.
   - [ ] Benchmark automation (scripted runs + CSV output + plotting).

B. Benchmark kế hoạch chi tiết
Mục tiêu benchmark: đo và tối ưu từng stage (prefilter, PQ evaluate, refine, end-to-end) trên 3 kích thước dataset: 100k, 500k, 1M với D=768. So sánh nhiều hệ cấu hình PQ/fingerprint/threshold.

1. Phần cứng & môi trường
   - Target machine: 2 cores, 4 threads, 8 GB RAM (swap disabled) — gọi là “weak node”.
   - Tốt có 1‑2 test machines mạnh hơn (8 cores, 32GB) để baseline.
   - OS: Linux (Ubuntu/CentOS), kernel 4.15+.
   - Compile flags: -O3 -march=native (để bật AVX2), link libpthread.
   - Pin threads (CPU affinity) cho worker/search thread để đảm bảo ổn định.

2. Datasets
   - Synthetic:
     - N=100k, 500k, 1M
     - D=768, float in [0,1] uniform or gaussian (tune generator to match expected distribution).
     - Save as binary file for reproducibility.
   - Optional Real:
     - nếu có dataset thực (embedding từ BERT/others), dùng 100k - 1M subset.
   - Train PQ codebooks on representative samples:
     - n_train = min(20000, 10% of dataset) — run kmeans per subquant.

3. Configurations to test (matrix)
   - Fingerprint:
     - SimHash bits: 256, 512
     - Raw sign bits: (baseline) D bits (768)
   - PQ parameters:
     - m=48 (subdim=16), k=256 (baseline)
     - m=64 (subdim=12), k=256 (higher granularity)
     - 8-bit codes (RAM resident) vs 4-bit packed (disk) trade-off
   - Candidate prefilter thresholds:
     - POPCOUNT threshold values tuned to produce ~1M→20k, 10k, 5k candidates
     - Or top-X by popcount (e.g., top 20k)
   - Final refine size:
     - refine_top = 100, 200, 500
   - PPE settings:
     - hot fraction: 1%, 3%, 5% (keeps full float copies)
     - demote_threshold_ms: 1s, 5s, 30s
     - promote_lookahead_ms: 5s, 30s
   - Concurrency:
     - single query thread; multi-threaded queries (1..4 threads) to measure scalability.

4. Metrics to collect
   - Memory:
     - RSS (resident) and total mapped file size
     - RAM used by SoA arrays (sum)
     - Additional hot floats memory
   - Latency:
     - P50, P90, P95, P99 end-to-end search latency (ms)
     - Breakdown: prefilter time, PQ-eval time, refine time
   - Throughput:
     - Queries per second (steady-state)
   - Accuracy:
     - Recall@K (e.g., recall@10) vs brute-force exact (compute on small subset or use ground-truth)
     - Mean Reciprocal Rank (MRR) optional
   - IO/OS:
     - pagefaults/sec (minor/major)
     - bytes read/written to disk per second (fwrites during demotion)
   - Candidate stats:
     - average candidates after prefilter
     - candidate distribution histogram
   - PPE behavior:
     - number promoted/demoted per interval
     - hot set size over time

5. Measurement methodology (repeatable)
   - Warm-up:
     - Insert full dataset.
     - Warm PPE: run 1000 warm-up queries (or run demoter/promoter for a warmup period).
     - Ensure codebooks loaded and mmap pages touched (madvise WILLNEED).
   - For each test configuration:
     - Clear OS caches only if comparing cold vs warm (echo 3 > /proc/sys/vm/drop_caches).
     - Run T repeats (e.g., 5 runs) each with Q random queries (Q>=500–2000) drawn from held-out test set.
     - Measure latency per-query and compute percentiles across all runs.
     - Run brute-force for ground-truth on a small subset (1000 queries) or use FAISS brute-force to get exact top-K.
     - Record memory RSS at start and during run.
     - For P95 P99, ensure sample size large enough (≥1000 queries).
   - Statistical significance:
     - Report mean & 95% CI of latency across runs.
     - Use same random seed for query selection to ensure reproducibility.

6. Micro-benchmarks (component-level)
   - Fingerprint encode speed:
     - throughput vec/sec for SimHash encoding.
   - Prefilter throughput:
     - speed to scan 1M fingerprints and compute popcount (ms).
     - measure vectorized vs scalar.
   - PQ precompute:
     - time to compute m tables for a query (m * k floats), should be small (ms).
   - PQ evaluate on candidates:
     - time per candidate for table-sum, vectorized vs scalar.
   - Arena fetch speed:
     - time to read single vector via arena offset (cold vs hot).
   - Pack/unpack 4-bit IO:
     - time to read and unpack demoted codes for one id.

7. End-to-end scenarios (priority list)
   - Scenario A (Warm, RAM resident codes):
     - PQ 8-bit in RAM, fingerprints in RAM, hot set populated.
     - Expect best latency numbers; measure baseline.
   - Scenario B (Mixed, PPE active):
     - PQ 8-bit but many cold codes demoted (4-bit on-disk). PPE demoter active.
     - Measure P95 spikes due to page faults.
   - Scenario C (Cold startup):
     - OS cache empty, evaluate initial queries and warm-up effect.
   - Scenario D (High update rate):
     - Insert rate 10/s, 100/s, 1000/s (simulate) and measure how PQ staleness / index mutability affects recall and latency.
   - Scenario E (Scale):
     - Run N=100k → 500k → 1M and measure linearity of memory & latency.

8. Parameter sweep plan (automated)
   - For each N in {100k,500k,1M}:
     - For each fingerprint in {SimHash512, SimHash256}:
       - For each PQ in {m48_k256, m64_k256}:
         - For candidate_target in {5k,10k,20k}:
           - For refine_top in {100,200}:
             - Run full benchmark (warm-up + Q=1000 queries).
   - Automate via script to produce CSV per-run with columns: config, N, fingerprint, PQ, candidate_count, refine_top, P50,P90,P95,P99,R@10,R@100,memory_bytes,pagefaults.

9. Expected target numbers (guideline, not guaranteed)
   - Memory footprint (SoA + codebooks) for 1M:
     - fingerprint (96B) = ~96MB
     - PQ (48B) = ~48MB
     - ids/offsets = ~8MB
     - PPE array ~16–32MB
     - codebooks ~1MB
     - total ~170–200MB baseline (plus hot floats)
   - Latency on weak node (single threaded, good vectorized impl):
     - Prefilter: 4–12 ms to scan 1M bitpacked SimHash (vectorized) — depends on memory bandwidth and implementation.
     - PQ evaluate for 10k candidates (vectorized): ~2–10 ms
     - Refine on 100 vectors (L2): <1ms if floats in RAM; if need to page-in, add I/O penalty.
     - End-to-end warm P50: 10–30 ms; P95: 20–60 ms depending on page faults and candidate_count. Aim to keep P95 <50ms by tuning candidate_count and hot set size.
   - Recall: with good PQ/opq + moderate candidate_count (10k → 20k) expect recall@10 ~0.85–0.95 depending on dataset & PQ quality.

10. Reporting & dashboards
    - For each run produce:
      - CSV line with config and summary metrics (P50/P90/P95/P99, Rec@10, memRSS, pagefaults).
      - Histogram plots: latency distribution, candidate_count distribution.
      - Time series during run: memory, pagefaults, IO rate.
      - Heatmap: hot set size over time, demote/promote counts.
    - Visualize via simple scripts (python/matplotlib) or Prometheus+Grafana.

11. Troubleshooting checklist (if results bad)
    - High P95 spikes:
      - Check pagefaults/IO; increase hot set/pin most accessed pages or prefetch via madvise.
      - Reduce refine_top or candidate_count.
    - Low recall:
      - Increase candidate_count; improve PQ (OPQ rotation) or increase m or k.
      - Consider adding coarse IVF clustering prefilter (PPIVF) in addition to fingerprint.
    - High memory:
      - Ensure 4-bit packing on-disk for demoted items; move more to arena/disk.
      - Reduce full-float hot set.
    - Poor throughput:
      - Vectorize PQ sum & popcount, use prefetch, tune thread affinity.

12. Timeline & effort estimate (rough)
    - Prototype (single-threaded, end-to-end on synthetic data): 2–3 weeks.
    - Vectorize & optimize prefilter & PQ evaluation (AVX2): 1–2 weeks.
    - Demoter/promoter + mmap reliability + tests: 1–2 weeks.
    - Benchmarking & tuning (matrix runs + analysis): 1–2 weeks.
    - Total MVP to production-ready: ~6–10 weeks depending on team size & experience.

13. Deliverables cuối cùng
    - Reproducible benchmark scripts and dataset generator.
    - CSV results + plots for all configs.
    - Recommended production config (fingerprint bits, PQ m/k, candidate thresholds, PPE params).
    - Runbook: how to deploy, monitor, tune.

------------------------------------------------------------
PHẦN MỚI: MỞ RỘNG TODO — GIAI ĐOẠN 2 (ASYNC DEMOTE, BATCH IO, TỐI ƯU HEO)
Mục tiêu giai đoạn 2: chuyển cơ chế demote hiện tại (đang có các gọi ghi file đồng bộ trên hot path) sang kiến trúc bất đồng bộ, batch-friendly, giảm số file nhỏ, giảm tail-latency và tránh blocking I/O trong addPoint / addQuantizedPoint. Giai đoạn này là bắt buộc để Thaut65/PomaiLight hoạt động ổn định production trên máy yếu.

A. Tổng quan giải pháp
- Giữ chức năng demote nhưng thực hiện bất đồng bộ:
  - Thay synchronous demote_blob_data(...) thành demote_blob_async(...) trả về placeholder.
  - Background DemoteWorker thực hiện ghi file theo batch/segment, cập nhật remote_map_ khi xong.
- Thay model "1 file / 1 blob" bằng segment files (append-only) hoặc preallocated blob store để giảm metadata and syscalls.
- Sử dụng PPPQ mmap file model cho PQ-codes demotion; cải tiến PPPQ write/read để giữ fd mở và sử dụng pwrite/mmap.
- Expose metrics & backpressure (max pending demotes). Nếu queue đầy, chọn chính sách (block briefly / fail / force sync) theo config.

B. Tasks chi tiết (checkboxes) — P2 (ưu tiên & mapping file)
1. API & data structures
   - [ ] Thêm struct DemoteTask { uint64_t dst_data_ptr_offset_in_index_area, std::vector<char> payload_with_header, label, promise/optional callback } vào src/memory/arena.h/.cc
   - [ ] Thêm PomaiArena::demote_blob_async(const char *data_with_header, uint32_t total_bytes) -> returns placeholder_remote_id (e.g., special REMOTE_PENDING or reserved temp id)
   - [ ] Thêm PomaiArena::get_demote_queue_length() và config knobs: max_pending_demotes, demote_batch_size_bytes.
   - AFFECTED FILES: src/memory/arena.h, src/memory/arena.cc

2. Background demote worker
   - [ ] Implement demote worker thread inside PomaiArena that:
     - dequeues DemoteTask(s),
     - batches multiple tasks into a single segment file write (append),
     - for each task, compute final remote_id = blob_region_bytes_ + segment_id<<32 | offset_in_segment (or mapping structure),
     - atomically update the index payload (element's payload area) with remote_id and set PPE_FLAG_REMOTE.
   - [ ] Safety: write tmp segment or write with fsync at rotation; ensure rename/atomic commit semantics for segment updates if needed.
   - AFFECTED FILES: src/memory/arena.cc

3. Segment file format & index
   - [ ] Design remote_id encoding (proposal): remote_id = blob_region_bytes_ + (segment_id << 32) | offset (32 bits) — adjust if offset might exceed 32-bit.
   - [ ] Maintain in-memory map remote_map_[remote_id] -> (filename, segment_size) and persist small index file if durability needed.
   - [ ] File rotation: rotate when segment >= segment_size_limit (e.g., 512MB).
   - AFFECTED FILES: src/memory/arena.cc, maybe new src/memory/segment_store.h/.cc

4. Replace synchronous demote call sites
   - [ ] In src/ai/pomai_hnsw.cc:
     - replace calls to pomai_arena_->demote_blob_data(...) inside addPoint/addQuantizedPoint/restorePPEHeaders with pomai_arena_->demote_blob_async(...).
     - on fallback paths, do not throw; mark seed payload as REMOTE_PENDING or store placeholder remote_id and PPE_FLAG_INDIRECT (without REMOTE).
   - [ ] Ensure consumers of payload (readers) check for REMOTE_PENDING and behave (either treat as demoted and use PQ-only path, or return not-available until demote completes).
   - AFFECTED FILES: src/ai/pomai_hnsw.cc

5. Improve PPPQ file I/O
   - [ ] Change PPPQ::writePacked4ToFile/readPacked4FromFile to:
     - open persistent fd once (on PPPQ constructor), keep fd and use pwrite for writes and pread for reads; protect with mmap_file_mtx_.
     - Optionally memory-map the whole file and write into mapping (requires resize-on-start).
   - [ ] Avoid opening/closing stream on each write.
   - AFFECTED FILES: src/ai/pppq.cc/.h

6. Promotion: async promote & prefetch
   - [ ] Demoter/promoter should schedule promotions early for predicted-hot items (PPE).
   - [ ] Implement async promote_remote(remote_id) that reads segment file into arena in background and atomically updates payload to local offset.
   - [ ] Add madvise(MADV_WILLNEED) or mmap remap for hot segments.
   - AFFECTED FILES: src/memory/arena.cc, src/ai/pomai_hnsw.cc

7. Backpressure & policies
   - [ ] Add config knobs in src/core/config.h for:
     - demote_async_max_pending (default e.g., 10000)
     - demote_batch_bytes (e.g., 4MB)
     - demote_segment_size (e.g., 512MB)
     - demote_sync_fallback (bool: if queue full, block synchronously or fail)
   - [ ] Implement behavior in demote_blob_async: if queue > max_pending and demote_sync_fallback==true, perform synchronous demote (rare); if fallback==false, return error code or set REMOTE_PENDING but escalate metrics.
   - AFFECTED FILES: src/core/config.h/.cc, src/memory/arena.cc

8. Observability & metrics (must)
   - [ ] Add PomaiMetrics counters and gauges:
     - demote_tasks_enqueued, demote_tasks_completed, demote_tasks_failed
     - current_demote_queue_length, demote_bytes_written_total
     - demote_avg_latency_ms
   - [ ] Expose via OP_VMEM or separate OP_DEMOTE_STATS wire op for quick inspection.
   - AFFECTED FILES: src/core/metrics.h, src/memory/arena.cc, server.h

9. Tests & validation
   - [ ] Unit test for async demote: schedule demote, verify fast return, then poll until remote_map_ updated and payload area updated.
   - [ ] Stress test: saturate demote queue and verify behavior with demote_sync_fallback true/false.
   - [ ] Crash recovery test: simulate crash during demote and verify on restart remote_map_ can be reconstructed (scan remote_dir for segment files).
   - AFFECTED FILES/LOCATIONS: tests/, CI pipeline

10. Documentation & runbook
    - [ ] Document remote_id encoding, segment file layout, demote behavior and config knobs.
    - [ ] Add operation runbook: inspect demote queue; tune batch size; what to do if demote worker slow.

C. Acceptance criteria cho GĐ2
- [ ] addPoint/addQuantizedPoint không block trên disk I/O trong hầu hết trường hợp (phải trả về nhanh, đếm thời gian).
- [ ] Demote worker thực tế ghi blobs vào segment files, batch writes giảm số file nhỏ so với prior.
- [ ] Demote queue length, demote latency, demote rate được xuất metrics và trong giới hạn config.
- [ ] Tail-latency (P95) cho VSEARCH cải thiện (giảm spike) so với trước khi dùng async demote (measured trong benchmark).
- [ ] PPPQ demotion / read path không open/close file mỗi lần nữa, thay bằng fd/mmap reuse.

D. Timeline & effort estimate (GĐ2)
- Implement demote queue + worker + API + replace call sites: 4–7 days.
- Implement segment file format + mapping & rotate logic: 2–3 days.
- Update PPPQ FD reuse & pack/unpack optimization: 2–3 days.
- Tests, tuning, and benchmark runs: 3–5 days.
- Tổng GĐ2: ~2–3 tuần (bao gồm testing + small fixes).

E. Notes & trade-offs
- Batching reduces IOPS and metadata pressure, nhưng làm tăng latency "time-to-persist" cho từng blob (nhưng chấp nhận được vì hot path trả về nhanh và payload sẵn truy vấn PQ/approx path).
- Segment files cần giới hạn kích thước để tránh very large files; rotate & GC policy cần thiết.
- Nếu cần mạnh về crash-consistency/durability, thêm small index file per segment (atomic write) hoặc write-ahead log—là nâng cấp tiếp theo.

F. Gợi ý code starter (nếu bạn muốn PR skeleton)
- Tôi có thể tạo PR skeleton bao gồm:
  - PomaiArena: thêm demote queue struct và demote_worker thread skeleton.
  - pomai_hnsw.cc: thay synchronous demote -> demote_blob_async calls (stubs + comments).
  - PPPQ: change writePacked4ToFile/readPacked4FromFile to use persistent fd (skeleton).
  - Config knobs trong core/config.
- PR sẽ bao gồm unit-test stubs để dev tiếp tục.

------------------------------------------------------------
Kết luận: bạn nên giữ cơ chế demote (rất cần) nhưng chuyển sang mô hình bất đồng bộ và batch để tránh block trên hot path và giảm tail-latency. Nếu bạn đồng ý, mình sẽ bắt đầu tạo PR skeleton cho GĐ2 (demote queue + async API + replacements) — bạn chọn "tạo PR skeleton" hoặc "viết spec segment format" tiếp theo.