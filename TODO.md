Understood — below is a focused, actionable TODO list that integrates the unused “granule” files into the VectorStore pipeline. This is strictly about the integration plan, per-file tasks, tests, acceptance criteria, and a short rollout/verification checklist. Follow this plan as high-priority work — do one phase at a time and run the listed tests/benchmarks before moving on.

High-level goals (must be satisfied):
- Use all existing modules (simhash/fingerprint, prefilter, pq, pq_eval/pq_encoder, refine, soa/ids/arena, PPE/PPPQ/PPIVF, candidate_collector) so no file is left unused.
- Provide a working SoA-backed VectorStore mode that stores 1M vectors (dim 512) in ~600–700 MB with query latency <50 ms on a modest 2-core 8 GB machine (engineering target—measure per run).
- Pipeline for search: binary-prefilter -> PQ approximate -> refine top-K exact -> return merged results.
- Robustness: crash-recoverable SoA ids + WAL replay; atomic publish semantics already present must be used.

Priority phases (do in this order). Each phase includes precise file-level actions, tests, and acceptance criteria.

PHASE 1 — SoA mmap storage & ids integration (1 day)
- Objective: provide on-disk SoA storage (SoaMmapHeader layout) and Soa-backed read/write helpers so VectorStore can store binary fingerprints, packed PQ codes, and ids/offsets in a single mapped file.
- Files to implement/modify:
  - Add new files:
    - src/ai/vector_store_soa.h / src/ai/vector_store_soa.cc: SoA helper encapsulating MmapFileManager + SoaMmapHeader parsing + block pointer getters and small helpers to append/allocate entries.
    - tests/soa/soa_write_read_test.cc: simple create SoA, write a few vectors, reload and check blocks.
  - Use/modify:
    - src/ai/soa_mmap_header.h (already exists) — use layout fields.
    - src/memory/mmap_file_manager.{h,cc} — use existing APIs (atomic_store_u64_at, flush).
    - src/ai/soa_ids_manager.{h,cc} — ensure atomic_store/load usage and WAL integration is used when Soa writes ids.
- Implementation details:
  - SoA layout blocks: fingerprints (bitpacked), pq_packed4, ids_block (uint64_t per vector where IdEntry encodes local offset/label/remote), PPE block optional.
  - VectorStore_soa API (minimal): open_or_create(path, header_init), append_vector(fp_bytes, pq_packed_bytes, id_entry) -> returns index; read_block_ptrs().
  - Ensure writes use MmapFileManager::write_at / atomic helpers for 8-byte writes and ids_mmap_ flush when durable.
- Tests:
  - tests/soa/soa_write_read_test: create SoA with N=1024, write entries, unmap, reopen and verify content matches.
- Acceptance:
  - Tests pass in CI.
  - Reopen recovers mapping offsets and pointers successfully.

PHASE 2 — Binary fingerprint (SimHash) + prefilter integration (1 day)
- Objective: compute bitpacked fingerprints per vector on add; use prefilter to reduce candidate set for search.
- Files to use:
  - src/ai/simhash.h/.cc (exists) — compute fingerprints.
  - src/ai/fingerprint.h/.cc — make a thin wrapper factory around SimHash (already present).
  - src/ai/prefilter.h/.cc — use compute_hamming_all / collect_candidates_threshold / topk_by_hamming.
  - vector_store_soa.* (new) — store fingerprints in the binary block.
  - src/ai/vector_store.{h,cc} — add code path to compute fingerprint on upsert and to use prefilter on search when SoA mode enabled.
- Implementation details:
  - On addVec (SoA mode): compute fingerprint bytes() -> write to SoA binary block at index.
  - On search:
    - get fingerprint for query (fingerprint_->compute_words / compute())
    - call prefilter::collect_candidates_threshold with threshold tuned (config)
    - get vector of candidate ids (indices)
- Tests:
  - tests/prefilter/prefilter_integration_test.cc: insert N vectors with known clusters, check prefilter selects small candidate set with recall > 95 for top-100.
- Acceptance:
  - Prefilter reduces scan size by 10–100x for random test dataset.
  - Test shows recall > configured threshold for top-K (e.g. top-100).

PHASE 3 — PQ training, encoding, packed4 store + pq_eval integration (2 days)
- Objective: quantize vectors using ProductQuantizer, store packed4 codes in SoA, evaluate approximate distances on candidates with pq_eval SIMD paths.
- Files to use:
  - src/ai/pq.h/.cc, pq_encoder.h/.cc — PQ training / encoding.
  - src/external/pq_trainer.cpp — offline trainer utility (use for tests).
  - src/ai/pq_eval.h/.cc — batch evaluator (AVX2/gather path).
  - src/ai/quantize.h — helper quantization.
  - vector_store_soa.* — store packed4 codes.
  - src/ai/vector_store.{h,cc} — add PQ encode on upsert (if training available) and pq_eval use on search.
- Implementation details:
  - Training: offline or on small sample at init; store codebooks file path in SoA header or config.
  - Encoding:
    - On addVec: pq.encode(vec) -> pack4From8 -> write to SoA pq_packed block at index.
    - Keep small in-RAM codebook (ProductQuantizer object).
  - Searching:
    - For candidate set from prefilter, build distance tables with codebooks (per query).
    - Call pq_eval::pq_approx_dist_batch_packed4(tables, m, k, pq_packed_block + candidate_offset, n_candidates, out).
    - Choose top-N approximate candidates (e.g. 100) for refine.
- Tests:
  - tests/pq/pq_integration_test.cc: train small PQ (m=8/k=256) on random data, encode vectors, run pq_eval batch on candidate set and compare relative ordering vs exact L2 for top-100.
- Acceptance:
  - Approx pipeline returns top-100 that contains the true top-100 (recall > 90%) for test dataset.
  - pq_eval uses SIMD path on CPU with AVX2 available (optional but test must run scalar fallback if missing).

PHASE 4 — Refine exact final scoring (refine module) (0.5 day)
- Objective: refine top candidates exactly (L2/IP) by fetching full vectors using arena/ids offsets and using refine::refine_topk_*.
- Files to use:
  - src/ai/refine.h/.cc (exists) — uses atomic id loads and arena access.
  - src/ai/ids_block.h/.cc — id entry encoding/decoding for local/remote/label.
  - src/memory/arena.{h,cc} — resolve offsets, promote_remote, demote support.
  - vector_store_soa.* — ids block returns IdEntry values; SoA vector payload may be inline or arena offset (choose arena approach).
- Implementation details:
  - After PQ approximate top-N, call refine::refine_topk_l2(query, dim, candidate_ids, ids_block_ptr, arena, K).
  - If any IdEntry is remote or placeholder, arena.promote_remote will be used (as in current refine).
- Tests:
  - tests/refine/refine_end_to_end_test.cc: create small dataset, run full pipeline (prefilter->PQ->refine) and verify exact top-K.
- Acceptance:
  - refine returns exact top-K and the pipeline runtime remains reasonable for test sizes.

PHASE 5 — PPE adaptive demotion/promote & PPPQ integration (1 day)
- Objective: use PPE predictors to choose when to keep payloads inline/in-arena/demoted, instruct PPPQ to manage packed PQ demotion, and use PPE hints to pick 4-bit vs 8-bit.
- Files to use:
  - src/ai/ppe_predictor.h / ppe_array.h — predictor state and APIs.
  - src/ai/pppq.h/.cc — PPPQ demotion metrics & addVec.
  - src/ai/pomai_hnsw.* — background demoter integrates PPE (already integrated).
  - vector_store_soa.* — set PPE per-vector metadata in SoA header or separate PPE block.
- Implementation details:
  - On addVec: call PPE touch; set precision bits in PPEHeader if used.
  - PPPQ: call addVec for index labels where PPPQ is attached; PPPQ will demote/publish flags.
  - Demotion: when PPE marks cold, pack PQ codes to packed4 if not already and optionally demote to remote files (arena.demote_blob_async).
- Tests:
  - tests/pppq/pppq_demote_metrics_test.cc: push many vectors, call PPPQ::purgeCold(), ensure demote counts increment and in_mmap flags set.
- Acceptance:
  - PPE adapts bits and demote queue metrics behave as expected.

PHASE 6 — VectorStore wiring, modes parity & tests (1 day)
- Objective: expose the new SoA-backed VectorStore mode in existing VectorStore API with parity to single-map legacy mode; add integration tests for parity.
- Files to change:
  - src/ai/vector_store.h/.cc — add attach_soafile() or attach_shard_manager() to choose SoA/sharded modes; implement upsert/search/remove branching to SoA pipeline.
  - Add tests:
    - tests/integration/vector_store_parity_test.cc (exists but failing) — update to refer to correct Seed type namespace and build parity test that compares legacy single-map behavior vs SoA mode on small workloads.
    - tests/integration/vector_store_end_to_end_test.cc — pipeline e2e: add N vectors, query, assert results equal between modes (within tolerances).
- Implementation details:
  - VectorStore::init should detect sharded_mode_ vs SoA_mode_ and call the relevant init functions.
  - For SoA mode addVec must:
    1) Compute fingerprint (SimHash)
    2) PQ encode + pack4
    3) Append vector payload to arena or store inline depending on config
    4) Write ids block entry (IdEntry::pack_local_offset or pack_label)
    5) Store label mapping (if used) using existing map or label->key map
  - Add config toggles: enable_prefilter, enable_pq, pq_codebooks_path, soa_file_path
- Tests:
  - vector_store_parity_test: assert upsert/search/remove results match between single-map and SoA mode on small N (e.g. 2048).
- Acceptance:
  - Parity test passes on CI.

PHASE 7 — WAL, crash-recovery tests for SoaIdsManager (0.5 day)
- Objective: ensure WAL replay/truncate recovers mapping after simulated crash (your earlier TODO).
- Files:
  - tests/soa/soa_ids_manager_wal_recovery_test.cc (exists partial) — finish test, ensure it uses wal_manager_ or raw wal and then reopens and verifies ids map restored.
  - SoaIdsManager::replay_wal_and_truncate already implemented — ensure SoA open uses it.
- Tests:
  - Run the recovery test: append WalEntry directly to wal file, reopen SoaIdsManager and verify ids block contains applied value.
- Acceptance:
  - Test passes reliably.

PHASE 8 — Performance validation, profiling & CI adjustments (1 day)
- Objective: measure memory, latency, and tune thresholds; add CI smoke benchmarks (optional).
- Tasks:
  - Microbenchmarks:
    - scripts/bench/bench_pipeline.sh: generate N synthetic vectors, build SoA, run queries and measure end-to-end latency and memory.
  - Add basic CI job (non-blocking) that runs small-scale pipeline and ensures no regressions.
  - Document required CPU flags for SIMD acceleration (AVX2) and fallback behavior.
- Acceptance:
  - Benchmarks show candidate reduction and PQ approximate speedups vs naive baseline for representative N (e.g., 100k).
  - Memory usage measured via procfs must be less than target for a scaled test (document results).

Cross-cutting tasks & implementation notes
- Config options (src/core/config.h):
  - Add new runtime flags:
    - enable_soa_mode (bool)
    - soa_file_path
    - pq_codebooks_path
    - fingerprint_bits
    - prefilter_threshold
    - demote_async_max_pending (exists)
    - demote_sync_fallback (exists)
  - Make default conservative: disable PQ/prefilter by default until codebooks available.
- API stability:
  - Keep VectorStore public API unchanged for callers. Add attach_soafile() or enable_soa_mode() that is optional.
- Concurrency/atomicity:
  - Always use existing atomic_utils / MmapFileManager atomic helpers when touching mapped 8-byte ids or PPE payloads.
  - Use wal_manager_ append_record for durable updates (SoaIdsManager already calls wal_manager_ when enabled).
- Tests / CI:
  - Add unit tests under tests/ for each module integration point.
  - Add a top-level integration test tests/integration/vector_store_parity_test.cc (update existing failing file to refer to global Seed type or pomai::core::Seed as needed).
  - Add a small benchmark harness in tools/bench to measure latency and memory.
- Rollback plan:
  - Implement changes as feature-flagged (enable_soa_mode). Merge only after parity tests pass.
  - If performance or correctness regressions occur, revert the SoA mode default to off and fix in feature branch.

Checklist (short actionable items)
1. Create vector_store_soa.h/.cc (SoA read/write helpers + header management) — implement append/read; test with SoA unit test.
2. Wire SimHash + prefilter into VectorStore upsert/search for SoA mode.
3. Integrate PQ encode + pack4 store + pq_eval approx search over candidate lists.
4. Use refine module to exact-refine top candidates.
5. Hook PPE/PPPQ demotion/promotion: publish precision bits & in_mmap flags consistently (use atomic_utils).
6. Add/adjust VectorStore API to enable SoA mode and keep parity with legacy single-map.
7. Add parity/integration/recovery tests (including WAL recovery test).
8. Add small benchmarking scripts and run them; iterate on threshold values.
9. Update docs/README with config options & how to train/load PQ codebooks and fingerprints.
10. Add CI job to run small-scale pipeline tests.

Suggested tests to add (file names)
- tests/soa/soa_write_read_test.cc
- tests/soa/soa_ids_manager_wal_recovery_test.cc (finish existing)
- tests/prefilter/prefilter_integration_test.cc
- tests/pq/pq_integration_test.cc
- tests/refine/refine_end_to_end_test.cc
- tests/integration/vector_store_parity_test.cc (fix namespace issue: use pomai::core::Seed or forward-declare correctly)
- tools/bench/bench_pipeline.sh (not a unit-test but useful for validation)

Commands & quick dev notes
- Train PQ offline:
  - ./build/pq_trainer --random --n 20000 --dim 512 --m 32 --k 256 --iters 10 --out codebooks.bin
- Run SoA unit test:
  - cd build && make tests && ./tests/soa/soa_write_read_test
- Run integration parity test:
  - ./tests/intergration/vector_store_parity_test
  - If fail with "Seed is not a member", update includes: tests should include "src/core/seed.h" and refer to pomai::core::Seed or fully qualified if header declares a different namespace.

Performance target tuning hints
- Prefilter threshold: tune so prefilter reduces 1M → 5k–20k candidates depending on bits.
- PQ m (subquantizers) and k (centroids) vs packing: choose m so packed4BytesPerVec fits size targets (goal ~32 B per vec).
- Candidate top for refine: 64..256 depending on accuracy/latency trade-off.

Acceptance criteria summary (must meet before considering feature complete)
- All newly added tests pass on CI.
- Parity test: single-map vs SoA mode produce same (or nearly same within epsilon) top-K results for small workloads.
- WAL recovery test for SoaIdsManager passes.
- Bench runs show large candidate reduction from prefilter and PQ approximation reduces expensive refine work; measured memory for scaled test is within order-of-magnitude of target (document exact numbers).
- No undefined behavior: mapped atomic load/store using atomic_utils where required.

If you want, I can:
- Produce skeleton code for vector_store_soa.h/.cc and unit test files per the plan.
- Produce concrete edits to VectorStore (vector_store.cc) to call new SoA pipeline (small patches).
- Produce the parity test fix (the immediate error you saw: qualify Seed declaration with pomai::core::Seed or adjust include order).

Which do you want me to do next? (I will only act on steps related to this TODO.)