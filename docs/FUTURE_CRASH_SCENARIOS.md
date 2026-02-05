# Future Crash Scenarios PomaiDB Should Catch

This list proposes **20 crash/failure situations** PomaiDB should detect and recover from in future milestones.
It is based on common incident patterns seen in production vector databases (e.g., abrupt power loss, WAL corruption, partial segment writes, compaction races, and OOM during indexing), and mapped to PomaiDB's current durability model and roadmap.

## 1) Power loss during WAL append
- **Pattern seen in large vector DBs**: process dies mid-record write.
- **Catch target**: tolerate torn/truncated WAL tail and replay valid prefix only.

## 2) Crash after WAL append but before fdatasync
- **Pattern**: acknowledged writes disappear after restart on weak durability settings.
- **Catch target**: mark durability boundary clearly and detect non-durable ACK windows.

## 3) Crash during memtable rotation
- **Pattern**: in-memory active/frozen pointers become inconsistent.
- **Catch target**: rebuild in-memory state from WAL replay deterministically.

## 4) Crash during segment build (partial file)
- **Pattern**: orphaned/corrupt segment file appears on disk.
- **Catch target**: verify segment integrity and avoid publishing partial segments.

## 5) Crash after segment fsync, before manifest commit
- **Pattern**: durable data file exists but metadata pointer is missing.
- **Catch target**: detect orphan segments and recover or garbage-collect safely.

## 6) Crash after manifest rename, before directory fsync
- **Pattern**: metadata commit durability differs by filesystem semantics.
- **Catch target**: startup validation that selects old/new manifest atomically.

## 7) Crash during WAL reset/truncation after freeze
- **Pattern**: duplicate replay or data reappearance after restart.
- **Catch target**: idempotent replay markers/checkpoints to avoid duplicate visibility.

## 8) Crash while updating per-membrane metadata
- **Pattern**: named collections disappear from catalog after restart.
- **Catch target**: durable membrane catalog scan/rebuild across restart.

## 9) Crash during manifest checksum write
- **Pattern**: checksum mismatch blocks startup.
- **Catch target**: fallback to last good manifest generation.

## 10) Crash while compaction output is being produced
- **Pattern**: mixed old/new segment sets lead to partial compaction visibility.
- **Catch target**: two-phase compaction publish with atomic manifest swap.

## 11) Crash after compaction publish, before obsolete-file cleanup
- **Pattern**: startup sees duplicate versions in old + new segments.
- **Catch target**: startup dedupe rules and safe deferred GC.

## 12) OOM kill during large batch ingest
- **Pattern**: process terminated by kernel, leaving partial in-memory progress.
- **Catch target**: replay-safe batching and bounded memory admission controls.

## 13) Crash during concurrent delete + put races
- **Pattern**: tombstone ordering bug resurrects deleted vectors.
- **Catch target**: strict newest-wins ordering across WAL/frozen/segments.

## 14) Crash during snapshot publication
- **Pattern**: readers observe half-published view.
- **Catch target**: lock-free but atomic snapshot pointer publication invariants.

## 15) Crash while mmap-based segment is being opened
- **Pattern**: invalid map or short file triggers SIGBUS/SIGSEGV on read.
- **Catch target**: pre-map file size/header validation and guarded open path.

## 16) Disk-full (ENOSPC) during freeze/flush
- **Pattern**: partial files + misleading success paths.
- **Catch target**: strict error propagation and rollback to pre-freeze snapshot.

## 17) I/O error (EIO) while fsync/rename appears to succeed partially
- **Pattern**: latent corruption discovered only at next restart.
- **Catch target**: end-to-end commit verification and startup consistency scan.

## 18) Crash during schema/format version transition
- **Pattern**: mixed-format files make node unbootable.
- **Catch target**: versioned manifest with upgrade fence and rollback path.

## 19) Crash in background worker thread (unhandled exception/abort)
- **Pattern**: main process stays up but state machine is deadlocked/incomplete.
- **Catch target**: watchdog + fatal escalation + restart-safe recovery sequence.

## 20) Host restart during high-concurrency shard writes
- **Pattern**: race-dependent metadata divergence across shards.
- **Catch target**: shard-level recovery invariants and deterministic startup reconciliation.

---

## Why these 20 are aligned with PomaiDB
PomaiDB already documents crash boundaries around WAL replay, segment persistence, manifest atomicity, and restart behavior.
These scenarios extend that model into compaction, catalog durability, mmap safety, and operational failure modes that commonly appear in larger vector DB deployments.


## Implementation status in this patch
- ✅ **Manifest corruption detection + fallback**: shard manifest now includes CRC32C and loader can fall back to `manifest.prev` if `manifest.current` is corrupted.
- ✅ **Manifest commit resilience**: commits keep a previous manifest generation to improve restart survivability.
- ⚠️ **Remaining scenarios**: most listed items still require dedicated fault-injection tests and runtime changes across WAL/reset/compaction/shard recovery paths.
