# PomaiDB Architecture

## Core Components

### 1. ShardRuntime (Actor)
The `ShardRuntime` is the heart of the system. It manages the lifecycle of data for a single shard partition.
- **Mailbox**: A `BoundedMpscQueue` receiving write commands (`Put`, `Delete`, `Freeze`).
- **Worker Thread**: A single thread processing the mailbox sequentially.
- **Snapshot Publication**: The worker thread publishes new atomic snapshots to `current_snapshot_`.

### 2. Read Path (Lock-Free)
Readers (`Search`, `Get`, `Exists`) bypass the mailbox entirely.
1. Load `std::shared_ptr<ShardSnapshot>` from atomic storage.
2. Query `FrozenMemTables` (Newest -> Oldest).
3. Query `Segments` (Newest -> Oldest).
4. Merge results.

**Note**: Since `ActiveMemTable` is NOT in the snapshot, reads have bounded staleness.

### 3. Write Path
1. `Put/Delete` -> Enqueue to Mailbox.
2. Worker pops command.
3. Append to WAL.
4. Update `ActiveMemTable`.
5. Check size > Threshold (5000).
6. If Threshold met: Call `RotateMemTable`.
    - Move `Active` -> `Frozen`.
    - Create new `Active`.
    - **Publish Snapshot** (New version visible).

### 4. Persistence
- **WAL**: Append-only log.
- **Segment V2**:
    - Header (Magic, Version, Count, Dim).
    - Entries (ID, Flags, Vector).
    - Tombstones are just entries with `kFlagTombstone`.
- **Manifest**: Atomic file swap (`manifest.new` -> `manifest.current`) tracking active segments.

## Invariants
See `src/core/shard/invariants.h` for formal listing.
1. Snapshots are immutable.
2. Deletes shadow older values (Tombstone wins).
3. Snapshot version is monotonic.
