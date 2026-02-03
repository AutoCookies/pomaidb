# Roadmap — Alpha → Hardening

> This roadmap is scoped to the current branch (`pomai-embeded`).

## Milestone 1: Alpha Stabilization
**Goal**: Establish deterministic recovery and visibility semantics.

**Acceptance criteria**
- WAL replay tolerates truncated tails and passes crash tests. (Source: `Wal::ReplayInto`, `tests/crash/crash_replay_test.cc`.)
- `Freeze` flushes frozen memtables to segments and resets WAL. (Source: `ShardRuntime::HandleFreeze`.)
- Documentation reflects current consistency model and failure semantics. (Policy.)

**Tests/benchmarks to pass**
- `ctest --test-dir build -L unit`
- `ctest --test-dir build -L integ`
- `ctest --test-dir build -L crash`

## Milestone 2: Hardening — Visibility & Staleness Controls
**Goal**: Make bounded staleness configurable and observable.

**Acceptance criteria**
- Configurable freeze thresholds by ops/bytes/time. (Planned; not in code.)
- Metrics for snapshot age and queue depth exposed through an API or callback. (Planned.)
- RYW behavior documented clearly for all read paths. (Policy.)

**Tests/benchmarks to pass**
- New unit tests for threshold configuration.
- TSAN tests for snapshot publication and reads. (Source: `tests/tsan/*`.)

## Milestone 3: Hardening — ANN Search Path
**Goal**: Replace brute-force scan with an ANN index while preserving snapshot isolation.

**Acceptance criteria**
- IVF/HNSW search path used under a snapshot-safe scheme. (Planned; current code bypasses IVF.)
- Search uses metric selection from `MembraneSpec`. (Planned.)
- Recall tests pass against the ANN path. (Source: `tests/recall/recall_test.cc`.)

**Tests/benchmarks to pass**
- `ctest --test-dir build -L recall`
- `./build/bench_baseline` comparison runs with ANN enabled.

## Milestone 4: Hardening — On-disk Compatibility & Durability
**Goal**: Formalize versioning and directory fsync behavior across filesystems.

**Acceptance criteria**
- Explicit backward compatibility policy for WAL/segment/manifest formats. (Policy; see `docs/ON_DISK_FORMAT.md`.)
- Directory fsync behavior documented per filesystem; tests validate rename durability. (Planned; see `docs/FAILURE_SEMANTICS.md`.)
- Stable API versioning policy. (Planned; see `GOVERNANCE.md`.)

**Tests/benchmarks to pass**
- New crash tests for manifest rename/fsync failure windows.
- Regression tests for format parsing across versions.
