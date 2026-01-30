# Pomai Database Lifecycle

Pomai is a pomegranate: immutable grains combined into a membrane view. All reads are snapshot-based, lock-free, and immutable.

## Lifecycle Phases

### 1) Ingest
- Upserts go into a per-shard WAL and memtable (`Seed`).
- The memtable is periodically frozen into an immutable segment (grain).
- The latest live snapshot is published via an atomic `ShardState` swap.

### 2) Immutable View
- Each shard maintains an atomic `ShardState` containing:
  - frozen segments (immutable snapshots + indexes)
  - current live snapshot
- Search and Scan read from shared immutable snapshots only.
- No locks on hot paths: readers only load shared_ptr snapshots.

### 3) Compaction (Silent, LSM-Style)
- Segments are assigned levels.
- When `compaction_trigger_threshold` segments exist at a level, they are merged into the next level.
- Compaction runs in the background and publishes a new `ShardState` only after:
  - merged vector data built
  - index rebuilt
  - quantization bounds recomputed
- Old segments are removed after publishing, and are deleted once refcounts drop.

### 4) Scan (DB-Grade Iteration)
- A scan captures a consistent snapshot (`ScanView`) at the membrane level.
- Cursor encodes:
  - snapshot epoch
  - grain index
  - row offset
- Scan continues safely across concurrent ingest/compaction because it holds immutable snapshots.

### 5) Export & Verify
- `pomai_dump` streams vectors and metadata using the Scan API.
- `pomai_verify` validates manifest, snapshot, dictionary, and index checksums.

### 6) Observability
- Exposed stats:
  - last_checkpoint_lsn
  - wal_lag_lsn per shard
  - compaction backlog
  - last compaction duration
  - scan throughput
  - search budget hit counters

### 7) Durability & Recovery
- WAL and checkpoint snapshots are authoritative.
- Recovery loads snapshots, verifies checksums, and replays WAL when needed.
