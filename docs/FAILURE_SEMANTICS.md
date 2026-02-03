# Failure semantics

## What it is
- Crash recovery via WAL replay into MemTables, followed by snapshot publication on startup. (Source: `pomai::storage::Wal::ReplayInto`, `pomai::core::ShardRuntime::Start`.)

## What it is not
- Not a distributed consensus or replicated log. (Assumption based on local file usage only.)

## Design goals
- Recover from process crash (`SIGKILL`) by replaying WAL prefixes. (Source: `Wal::ReplayInto` tolerates truncated tails.)
- Make segment/manifest updates crash-safe with atomic rename + dir fsync where implemented. (Source: `SegmentBuilder::Finish`, `ShardManifest::Commit`.)

## Non-goals
- Guarantee durability when `FsyncPolicy::kNever` is configured. (Source: `Wal::Flush` short-circuits when fsync is disabled.)

## Key invariants
- WAL replay stops on truncated tail without treating it as corruption. (Source: `Wal::ReplayInto`.)
- Snapshots represent a prefix of WAL history. (Source: `core/shard/invariants.h`.)

## Crash points and outcomes

> **Notation:** “Durable” means persisted to stable storage only when fsync is enabled (`FsyncPolicy::kAlways`) and the relevant fsync/SyncData call has completed.

### 1) Crash before WAL append
- **Outcome**: The write is lost. (Source: `ShardRuntime::HandlePut` appends WAL before MemTable update.)
- **Visibility**: The write never reaches any snapshot. (Source: `ShardRuntime::PublishSnapshot` uses frozen/segments only.)

### 2) Crash after WAL append, before snapshot publish
- **Outcome**: The write is replayed on restart if WAL record is durable. (Source: `Wal::AppendPut`, `Wal::ReplayInto`.)
- **Visibility after restart**: The replayed data is rotated to frozen at `ShardRuntime::Start` and becomes visible. (Source: `ShardRuntime::Start`, `RotateMemTable`.)
- **Risk**: If fsync is disabled, the WAL record may be lost. (Source: `Wal::Flush`, `FsyncPolicy::kNever`.)

### 3) Crash during soft freeze (MemTable rotation)
- **Outcome**: Rotation is in-memory only; on crash, the frozen/active state is reconstructed from WAL replay. (Source: `RotateMemTable` does not persist, `Wal::ReplayInto`.)
- **Visibility**: Depends on replay; entries not persisted to segments are still visible after startup rotation. (Source: `ShardRuntime::Start`.)

### 4) Crash during Freeze (segment write)
- **Outcome**: If segment file is incomplete, it may be missing from the manifest or fail to open on restart. (Source: `SegmentBuilder::Finish`, `SegmentReader::Open`.)
- **Visibility**: Only segments listed in `manifest.current` are loaded. (Source: `ShardManifest::Load`, `ShardRuntime::LoadSegments`.)

### 5) Crash after segment fsync, before manifest update
- **Outcome**: The segment file exists but is not referenced by `manifest.current`, so it is ignored on restart. (Source: `ShardManifest::Load` reads `manifest.current`.)
- **Data loss**: Writes flushed into that segment are lost from the visible state until a subsequent Freeze recreates them. (Assumption based on manifest being the only reference.)

### 6) Crash after manifest update, before dir fsync
- **Outcome**: The manifest rename may or may not be durable, depending on filesystem behavior. (Assumption; see `ShardManifest::Commit`.)
- **Visibility**: If manifest rename is lost, the segment may be ignored; if rename is durable, the segment is loaded. (Source: `ShardManifest::Commit`, `ShardRuntime::LoadSegments`.)

### 7) Crash after manifest update + dir fsync
- **Outcome**: The segment list is durable and will be loaded. (Source: `ShardManifest::Commit`, `ShardRuntime::LoadSegments`.)

### 8) Crash during WAL reset (after Freeze)
- **Outcome**: If WAL deletion is partial, replay may include already-flushed entries, which may reappear. (Assumption based on `Wal::Reset` deleting files without a transactional boundary.)
- **Visibility**: Replayed entries are rotated to frozen on startup, potentially duplicating data already in segments. (Source: `Wal::Reset`, `Wal::ReplayInto`, `ShardRuntime::Start`.)
- **Mitigation**: Rely on segment tombstones or compaction; formal de-duplication is not implemented. (Assumption based on absence of segment-level dedupe.)

## Platform assumptions
- **Atomic rename**: `std::filesystem::rename` is atomic on POSIX for same-directory renames. (Assumption; `ShardManifest::Commit` uses rename.)
- **fsync semantics**: `fdatasync`/`fsync` honor durability for file contents; directory `fsync` makes renames durable. (Assumption; `PosixFile::SyncData`, `FsyncDir`.)
- **Memory mapping**: `SegmentReader` uses `mmap` and assumes mapped data is stable after `Open`. (Source: `PosixFile::Map`, `SegmentReader::Open`.)

## Testing failure semantics
- **WAL truncation tests**: Use crash tests that kill the process mid-write and validate replay tolerance. (Source: `tests/crash/crash_replay_test.cc`.)
- **Manifest corruption tests**: Validate CRC and load failure behaviors. (Source: `tests/crash/manifest_corruption_test.cc`, `storage/manifest/manifest.cc`.)
- **Planned fault injection** (not implemented):
  - Inject crashes between segment fsync and manifest update.
  - Inject crashes between manifest rename and dir fsync.
  - Verify segment visibility and data loss scenarios. (Assumption based on current lack of fault injection hooks.)

## Operational notes
- Use `FsyncPolicy::kAlways` and explicit `Flush` to make WAL durable. (Source: `Wal::Flush`.)
- Call `Freeze` to persist frozen data into segments and reset WAL. (Source: `ShardRuntime::HandleFreeze`.)

## Metrics
- No built-in durability/flush metrics. (Source: public API lacks counters.)

## Limits
- The system provides best-effort durability when fsync is disabled. (Source: `Wal::Flush`.)

## Code pointers (source of truth)
- `pomai::storage::Wal::AppendPut` / `AppendDelete` — WAL durability boundary. (File: `src/storage/wal/wal.cc`.)
- `pomai::storage::Wal::ReplayInto` — replay with truncated tail handling. (File: `src/storage/wal/wal.cc`.)
- `pomai::core::ShardRuntime::Start` — rotate replayed data for visibility. (File: `src/core/shard/runtime.cc`.)
- `pomai::table::SegmentBuilder::Finish` — segment write + fsync + rename. (File: `src/table/segment.cc`.)
- `pomai::core::ShardManifest::Commit` — manifest update + dir fsync. (File: `src/core/shard/manifest.cc`.)
- `pomai::util::FsyncDir` — directory fsync implementation. (File: `src/util/posix_file.cc`.)
