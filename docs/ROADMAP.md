# Roadmap — Database Transformation

> **Authority**: This roadmap reflects PomaiDB's identity as defined in [docs/DB_CONTRACT.md](DB_CONTRACT.md).  
> **Branch**: `pomai-embeded`  
> **Focus**: Database-grade correctness, durability, and storage engine quality.

---

## Milestone DB-0: Database Constitution ✅

**Goal**: Lock PomaiDB's identity as an embedded database (not a search engine).

**Deliverables**
- ✅ [docs/DB_CONTRACT.md](DB_CONTRACT.md) — Final authority on identity and guarantees
- ✅ Implementation plan for 6-phase transformation
- ✅ Task breakdown for all phases

**Status**: Complete (2026-02-04)

---

## Milestone DB-1: Search Path Rewrite (DB-Grade Scan)

**Goal**: Replace search-engine-style scan with proper database merge scan.

**Acceptance criteria**
- `SearchLocalInternal` uses 1-pass merge scan (newest → oldest). (Source: `ShardRuntime::SearchLocalInternal`.)
- No per-query `unordered_set` allocations. (Source: shard-local reusable tracker.)
- Inline visibility checks (no multi-pass tombstone collection). (Source: integrated scan logic.)
- Correctness tests prove no tombstone leakage, newest-wins semantics. (Source: `tests/unit/search_correctness_test.cc`.)

**Tests to pass**
- `ctest --test-dir build -R search_correctness_test`
- All existing unit/integ tests remain passing

**Database Value**: Scan algorithm resembles LSM-tree iterators, not search engine loops.

---

## Milestone DB-2: Freeze & Manifest Atomicity (Durability Hardening)

**Goal**: Guarantee crash safety at any point during Freeze.

**Acceptance criteria**
- Manifest committed ONCE after all segments built. (Source: `ShardRuntime::HandleFreeze`.)
- Directory fsync after segment creation and manifest rename. (Source: `fsync_directory` calls.)
- Crash at any point results in either old snapshot or new snapshot (never partial). (Source: `tests/crash/freeze_crash_test.cc`.)
- WAL reset only after manifest commit succeeds. (Source: `HandleFreeze` sequencing.)

**Tests to pass**
- `ctest --test-dir build -R freeze_crash_test`
- All existing crash tests remain passing
- TSAN clean (no data races in snapshot publication)

**Database Value**: Atomic commit protocol like RocksDB/LevelDB manifest updates.

---

## Milestone DB-3: Segment Format (DB-Grade Storage)

**Goal**: Streaming segment writes, no full buffering in RAM.

**Acceptance criteria**
- `SegmentBuilder` writes records incrementally (no buffering). (Source: `SegmentBuilder::Add`.)
- Header + footer + checksum protocol. (Source: `SegmentBuilder::Finish`.)
- Fsync + atomic rename before segment is visible. (Source: `SegmentBuilder::Finalize`.)
- Optional: Block-level index in footer for faster seeks. (Source: segment footer format.)

**Tests to pass**
- `ctest --test-dir build -R segment_streaming_test`
- All existing segment tests remain passing

**Database Value**: Storage engine format like SSTable/LSM-tree segments.

---

## Milestone DB-4: Tombstones & Compaction (DB Moat)

**Goal**: Implement tombstone semantics and compaction that ANN engines never have.

**Acceptance criteria**
- Tombstones correctly block older versions across frozen memtables and segments. (Source: snapshot merge semantics.)
- Compaction purges tombstones and drops overwritten versions. (Source: `HandleCompact` merge logic.)
- Read amplification reduced after compaction. (Source: segment count metrics.)
- Correctness tests prove tombstone visibility invariants. (Source: `tests/unit/tombstone_visibility_test.cc`.)

**Tests to pass**
- `ctest --test-dir build -R tombstone_visibility_test`
- All existing compaction tests remain passing
- Verify tombstone purging during compaction

**Database Value**: GC and compaction logic fundamentally different from FAISS/ANN engines.

---

## Milestone DB-5: Data Engine Features (Not Search)

**Goal**: Make PomaiDB valuable as a data engine, even without ANN.

**Acceptance criteria**
- `SnapshotIterator` API for streaming iteration over snapshots. (Source: `include/pomai/iterator.h`.)
- Dataset export API with deterministic train/test/val splits. (Source: `DB::ExportDataset`.)
- Zero-copy iteration where possible. (Source: `SnapshotIterator` implementation.)
- Metadata-aware iteration (filter by timestamp, shard, etc.). (Source: iterator options.)

**Tests to pass**
- `ctest --test-dir build -R iterator_test`
- Export determinism tests (same seed = same split)
- All existing tests remain passing

**Database Value**: ML dataset export, streaming access, data engineering use cases.

---

## Milestone DB-6: Guardrails & Enforcement

**Goal**: Prevent drift toward search-engine identity via CI and contribution rules.

**Acceptance criteria**
- `CONTRIBUTING.md` updated with PR rejection criteria. (Source: DB_CONTRACT.md enforcement section.)
- CI checks reject PRs that:
  - Tune ANN recall as primary feature
  - Optimize SIMD kernels as architectural focus
  - Add FAISS-like features without DB guarantees
- Lint rules flag banned patterns (e.g., direct exposure of ef/M/nprobe). (Source: CI configuration.)

**Enforcement**
- CI pipeline checks for banned patterns
- Maintainers enforce via code review
- Design review required for breaking changes

**Database Value**: Permanent enforcement of database-first identity.

---

## Forbidden Roadmap Items

The following items will **NEVER** be on the roadmap:

- ❌ ANN recall benchmark optimization (e.g., competing on ANN-Benchmarks)
- ❌ HNSW/IVF parameter tuning as core API surface
- ❌ SIMD kernel optimization as primary focus
- ❌ Search-first features (fuzzy search, ranking functions, etc.)
- ❌ Distributed consensus or replication (embedded-only)

See [docs/DB_CONTRACT.md](DB_CONTRACT.md) for the complete rejection criteria.

---

## Long-Term Database Evolution (Post-Milestones)

### Storage Engine Hardening
- Multi-version concurrency control (MVCC) with transaction IDs
- Point-in-time recovery (PITR) via WAL archiving
- Incremental snapshots and checkpointing
- Format versioning and backward compatibility

### Operational Maturity
- Metrics and observability (snapshot age, compaction lag, etc.)
- Corruption detection and repair
- Online backup and restore
- Quota and resource management per membrane

### Data Engine Features
- Secondary indexes (metadata-based filtering)
- Bloom filters for existence checks
- Compression (per-segment or per-block)
- Tiered storage (hot/warm/cold)

### Performance (Database-Centric)
- Background compaction policies
- Write batching and group commit
- Read caching (block cache, not search cache)
- IO scheduling and prioritization

---

## Success Criteria

PomaiDB succeeds when:

1. ✅ **It makes sense without ANN**: Remove ANN entirely, PomaiDB is still valuable.
2. ✅ **WAL + Snapshot are central**: Core architecture revolves around durability/visibility.
3. ✅ **Cannot be mistaken for FAISS**: Design clearly diverges from search-first systems.
4. ✅ **Storage engine quality**: Crash safety, compaction, GC, format versioning are first-class.

PomaiDB fails when:

1. ❌ ANN tuning becomes the primary value proposition.
2. ❌ Search engine features dominate the roadmap.
3. ❌ Correctness is sacrificed for performance.
4. ❌ Durability or snapshot isolation are weakened.

---

**Version**: 2.0 (Database-Centric)  
**Previous Version**: 1.0 (Search-Hybrid, deprecated)  
**Last Updated**: 2026-02-04
