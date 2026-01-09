# Pomai VectorStore — TODO (priorities, tasks, acceptance criteria)

Mục tiêu: hoàn thiện VectorStore 10/10 (single & sharded modes), an toàn atomic trên mmap, mạnh về nén vector để chạy cả trên laptop yếu.

## P0 — An toàn atomic & ordering (must)

- [x] Audit atomicity & memory ordering (atomic_utils / std::atomic_ref)  
  - Mô tả: kiểm tra tất cả nơi đọc/ghi trường nằm trong mmap / placement-new (PPEHeader.flags, 64-bit payload(s), ids block).  
  - Files: `src/ai/atomic_utils.h`, `src/ai/ppe.h`, `src/ai/pomai_hnsw.cc`, `src/memory/mmap_file_manager.cc`, `src/ai/soa_ids_manager.cc`  
  - Acceptance: mọi viết-publish sequence payload->flag sử dụng pattern:
    - atomic_store_u64(payload_ptr, val)  (release)
    - atomic_set_flags(...) / atomic_store_u32(&flags, ...) (release/seq_cst as appropriate)  
  - Status: DONE (atomic_utils used; SoaIdsManager durable path updated to use atomic_store_u64; pomai_hnsw paths updated to use atomic_store_u64; restorePPEHeaders updated).

- [x] Replace std::atomic_ref stores in SoaIdsManager durable path with atomic_utils::atomic_store_u64  
  - Files: `src/ai/soa_ids_manager.cc`  
  - Status: DONE.

- [x] Ensure mmap_file_manager provides atomic helpers and callers use them  
  - Files: `src/memory/mmap_file_manager.h/.cc` (provides atomic_load_u64_at/atomic_store_u64_at).  
  - Acceptance: call-sites use atomic_utils (or MmapFileManager helpers) for 64-bit reads/writes in mapped region.  
  - Status: Partially done — tests for atomic mmap passed.

## P0 — PPPQ per-id atomic state

- [x] Replace `code_nbits_` and `in_mmap_` with atomic containers and use proper memory-ordering  
  - Files: `src/ai/pppq.h`, `src/ai/pppq.cc`  
  - Acceptance: `.store(..., std::memory_order_release)` on updates, `.load(..., std::memory_order_acquire)` when read. No torn/undefined behaviour; code compiles.  
  - Status: DONE (note: ensure vector<atomic<uint8_t>> initialized safely; use reserve+emplace_back or resize + explicit atomic init).

## P1 — Tests & verification

- [ ] Add unit tests for PPE flags ordering & publish patterns  
  - Mô tả: test that readers never observe flags=INDIRECT while payload is zero/partial (simulate writer ordering).  
  - Files: new test e.g. `tests/atomic/ppe_publish_test.cc`  
  - Acceptance: no inconsistent observations under stress (multi-threaded).

- [x] Run atomic mmap test and fix issues discovered  
  - Status: DONE (atomic_mmap_test PASS).

- [ ] Add tests for PPPQ demote/promote concurrent state (in_mmap/code_nbits)  
  - Mô tả: concurrent demote worker + readers should observe consistent code_nbits/in_mmap state when async demote in-flight.  
  - Files: `tests/pppq/pppq_demote_concurrency_test.cc`  
  - Acceptance: no inconsistent reads, statistics counters coherent.

## P1 — Robustness & durability

- [ ] WAL recovery end-to-end test for SoaIdsManager  
  - Mô tả: simulate crash after WAL write / before msync, ensure replay_wal_and_truncate recovers mapping.  
  - Files: tests for `src/ai/soa_ids_manager.*`  
  - Acceptance: mapping restored after replay.

- [ ] Add msync/flush tests for MmapFileManager append/flush paths (durable updates).

## P1 — Run/tests infra

- [x] Update test runner to discover & run tests recursively (include subfolders)  
  - File: `tests/run_tests.sh`  
  - Suggested change: use `find "${TEST_DIR}" -type f -name '*.cc' -o -name '*.cpp'` to collect tests recursively instead of only top-level.  
  - Acceptance: test-runner builds and runs all tests under `tests/` including nested dirs (atomic tests included).

## P1 — VectorStore features & behavior (functional)

- [ ] Ensure both modes work and have parity:
  - Single-map legacy mode (PomaiMap + PPHNSW + PomaiArena)
  - Sharded mode (ShardManager + PPSM + per-shard PPHNSW)
  - Files: `src/ai/vector_store.*`, `src/core/pps_manager.*`, `src/core/shard_manager.*`  
  - Acceptance: same API (upsert/search/remove) behaves equivalently for simple workloads; add integration tests covering both modes.

- [ ] Ensure insertion paths correctly set PPEHeader fields, label mapping, and PPPQ/PPIVF registration.

## P1 — Strong compression & laptop usability (performance)

Goal: allow running reasonably large indexes on low-RAM laptops by aggressive compression / demotion.

High-level tasks:

- [ ] Provide configurable quantization profiles & light-weight defaults for low-RAM:
  - Add config presets: `tiny`, `laptop`, `server` choosing PQ m,k, bits, arena size, M/ef.  
  - Files: `src/core/config.*`, `VectorStore::init` config paths.  
  - Acceptance: `laptop` preset keeps memory < ~4GB by default.

- [ ] Improve PPPQ demotion strategy:
  - Ensure async demote queue bounded and safe fallback to sync demote when queue full (config knobs).  
  - Already present: `demote_async_max_pending` and `demote_sync_fallback`. Verify behavior with tests.

- [ ] Optimize PQ code paths for low-memory:
  - Implement/polish packed-4 on-disk layout and lazy unpacking on query path (already partly implemented).  
  - Provide option to zero RAM codes after demotion to reduce memory footprint (done in async success path; ensure atomic visibility).

- [ ] Add lighter-weight SimHash/OPQSign presets (e.g., 128 bits) for fast prefilter on low-RAM.

- [ ] Micro-optimizations:
  - Use cache-friendly layouts, reduce per-element PPE overhead (if safe).
  - Add mlock hints only for small hot slices.

Acceptance: demonstrable end-to-end run on a low-RAM laptop (e.g., 8GB) with baseline dataset X (document exact dataset), with search latency within acceptable budget.

## P2 — Performance & SIMD

- [ ] Continue vectorization / AVX2/AVX512 paths (pq_eval, prefilter) and add runtime cpu feature gating (already present).  
  - Files: `src/ai/pq_eval.*`, `src/ai/prefilter.*`  
  - Acceptance: measurable speedup for pq_eval/prefilter on CPUs with AVX2.

## P2 — CI, docs, benchmarks

- [ ] Add CI workflow that builds, runs unit tests (including atomic), and runs a small benchmark harness for performance regressions.  
- [ ] Add developer docs describing publish-order expectations (payload->store->flag), mapping file formats, and how to tune runtime flags for laptops.  
- [ ] Provide benchmark scripts / reproducible dataset + instructions.

---

## Immediate next steps I can do for you

- Create the TODO.md file in the repo (I can open a PR if you want).  
- Update `tests/run_tests.sh` to recursively find tests and run them (I can provide the patch).  
- Add concurrency unit tests for PPE publish ordering and PPPQ demote/promote and run the full suite.  
- Run the full test suite locally and produce a short report.

Do you want me to:
- A) open a PR that adds TODO.md + updates `tests/run_tests.sh` (and optionally adds a PPE atomic test), or  
- B) just output the patch contents here so you can apply, or  
- C) something else?
