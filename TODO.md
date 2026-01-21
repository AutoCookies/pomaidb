PRIORITY ROADMAP (MUST-DO, ordered, each mục kèm lý do, rủi ro, estimated effort)

Phase 0 — Emergency correctness fixes (1–2 engineer-weeks) These must be done before anything else.

A. Fix mapping publish ordering (HIGH)

    Problem: append_only_arena.remap_locked currently publishes new mapping pointer before publishing new size; readers can observe pointer -> OOB access -> memory corruption / crash.
    Fix: publish mapped_size_ (new_size) first (store with release), then publish map_base_ (store pointer with release). Readers must be updated to load mapped_size_ first (acquire), then map_base_. Alternatively pack {ptr,size} into single atomic struct (C++20 atomic<struct> not portable) or use seq_cst ordering with documented load/store order.
    Files: append_only_arena.cc, append_only_arena.h, any reader helpers (offset_from_blob_ptr, blob_ptr_from_offset_for_map) — change load order to size then ptr.
    Risk: Medium — must be carefully synchronized and tested. Measure: reduces crash/UB risk to near zero.

B. Eliminate UB from unaligned atomic ops (HIGH)

    Problem: MmapFileManager::atomic_store_u64_at / write_at use __atomic_store_n on pointers that may be unaligned. This is UB on C++ standard and potentially crashes on some archs.
    Fix: require callers to only use atomic helpers when aligned; if unaligned, perform memcpy into local aligned uint64_t and use atomic_store on that aligned temporary into the destination using memcpy with proper memory_order via atomic_utils (or use byte-wise memcpy + msync). Better: enforce alignment for atomic fields (pad file layout) or implement atomic helpers that do memcpy under io_mu_ lock for unaligned addresses.
    Files: mmap_file_manager.cc, atomic_utils.h
    Risk: Medium. Measurable: eliminates rare data races/torn-writes.

C. msync/alignment correctness (HIGH)

    Problem: ShardArena::persist_range calls msync(addr, len) with unaligned addresses; append_only_arena.persist_range aligns but ShardArena not.
    Fix: always align msync arguments to page boundaries: page_off = floor(offset/page)*page; msync(base+page_off, page_end-page_off).
    Files: shard_arena.cc, append_only_arena.cc ensure consistent behavior.
    Risk: Low.

D. Single authoritative demote worker lifecycle (HIGH)

    Problem: demote worker started in multiple places (allocate_region and demote_blob_async) — potential double-start or race.
    Fix: centralize demote worker startup (single helper), use atomic flag + once semantics, and ensure destructor stops and joins.
    Files: arena.cc (PomaiArena allocate_region), arena_async_demote.cc
    Risk: Low.

Phase 1 — Testing/CI & safety hardening (2–4 engineer-weeks) E. Add unit tests & CI

    Write unit tests for:
        WAL: append_record, replay (including truncated/partial record cases), fsync behavior.
        remap ordering: simulate mapping publish order with mocks.
        MmapFileManager atomic helpers for aligned and unaligned cases.
        ZeroHarmony pack/unpack roundtrip for various dims and use_half_nonzero true/false.
        SimHash correctness and hamming distance.
    Add CI (GitHub Actions) to build on Linux x86_64, run tests, and run address sanitizer + UBSAN on subset.

F. End-to-end integration tests

    Small test that starts PomaiDB instance, creates membrance, inserts a few thousand vectors, checkpoint, restart, ensure recovered vector counts and exactness.

G. Add benchmarks harness

    Provide micro-benchmarks for insert throughput and search latency (we'll provide a Python benchmark harness per your earlier note). Measure P50/P99 and memory usage.

Phase 2 — Performance & operational (2–6 engineer-weeks) H. Replace std::async with threadpool for global scatter-gather (medium)

    File: orchestrator.h
    Rationale: reduce per-query thread spawn cost; expect 10–30% latency/throughput improvement under concurrency.
    Implement fixed threadpool sized to shards or CPU cores.

I. WAL batching and durable checkpoints (medium)

    Ensure WAL batching behavior is configurable and consistent across process exit. Add explicit fsync on graceful shutdown and durable checkpoint operation that truncates WAL only after snapshot consistent.

J. Observability (low-medium)

    Export metrics (Prometheus) and expose /health and /metrics endpoints or simple file dumps. Add periodic metrics reporter to log P99 of search latency (from WhisperGrain observation).

K. Documentation & ops runbook (low)

    Add README with sizing guidelines and recovery steps.

Phase 3 — Harder improvements (3+ weeks) L. Consider a safer remap approach for growing file-backed mapping:

    Use MmapFileManager abstraction everywhere; centralize mapping/resizing logic with atomic pointer+size publish (use std::atomic<uintptr_t> + size_t). Option: use a small versioned descriptor struct and publish with double-buffering semantics.

M. NUMA and shard-worker placement improvements

    Provide shard-manager pinning and per-shard worker threads for inserts/search to improve P99 on big machines.

MEASURABLE EXPECTATIONS (if roadmap applied)

    Correctness: crash/UB risk -> near zero after Phase 0 fixes (no pointer-size races; no unaligned atomics).
    Latency stability: P50/P99 improvement mainly from threadpool change and hotspot fixes; expect P50 stable, reduce tail by 20–50% for search under concurrency.
    Insert throughput: WAL batching + optimized hot-tier flush -> throughput increases 2x–4x for batch inserts.
    Memory: preallocation + hot-tier non-growing policy keeps memory bounded.

SPECIFIC BUGS / CODE LOCATIONS (evidence)

    append_only_arena.remap_locked: publishes pointer before size. (append_only_arena.cc, lines in remap_locked fallback)
    MmapFileManager::write_at / atomic_load_u64_at / atomic_store_u64_at: unaligned atomic operations (mmap_file_manager.cc).
    ShardArena::persist_range: msync called without page-align. (shard_arena.cc)
    GlobalOrchestrator::search: uses std::async per shard (orchestrator.h)
    In several places use of reinterpret_cast to uint64_t* for atomic_store on mmap memory that may not be 8-byte aligned. Audit required.
