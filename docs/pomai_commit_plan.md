# Commit-by-Commit Plan

1) **Scan API + Membrane View**
   - Add ScanRequest/Response types.
   - Implement Membrane-level snapshot view + cursor validation.
   - Wire PomaiDB::Scan and add scan stats hooks.

2) **Compaction Scheduler**
   - Add compaction knobs to DbOptions.
   - Implement per-shard size-tiered compaction with immutable swaps.
   - Track compaction backlog + duration for observability.

3) **CLI Tools**
   - Add `pomai_dump` using Scan for streaming export.
   - Improve `pomai_verify` output paths if needed.

4) **Stats + Lifecycle Hooks**
   - Expose checkpoint LSN, WAL lag, scan throughput, search budget counters.

5) **Testing**
   - Scan correctness (cursor, filters, uniqueness).
   - Compaction correctness (IDs preserved + segments reduced).
   - Corruption detection (verify rejects corrupted files).
