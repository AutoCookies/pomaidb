# PomaiDB Failure Semantics & Crash Safety

PomaiDB is designed for mission-critical edge hardware where power loss and hardware failure are active threats. Our reliability layer is distilled from SQLite's battle-tested VFS architecture.

## 1. Atomic Commit Protocol

PomaiDB follows a "Strict Transition" model to guarantee that any modification is atomic, durable, and consistent across system crashes.

### Metadata Atomicity
1. **Prepare**: Data is written to a new segment or temporary workspace.
2. **Flush**: Data is pushed to persistent storage using `fdatasync()`.
3. **Commit**: The directory structure is updated via an atomic `rename()` operation.
4. **Visibility**: The parent directory is flushed using `fsync()` to ensure the rename is durable.

## 2. Durability Mechanisms

### Fsync Policies
- **Data Segments**: We use `fdatasync()` on write-ahead logs and segment files. This flushes the data to the storage controller while avoiding the overhead of metadata (e.g., access time) updates where safe.
- **Header Integrity**: Critical database headers are flushed using `fsync()` to ensure file size and block pointers are committed to the filesystem's journal.
- **Directory Synchronization (DIRSYNC)**: Inspired by SQLite's `unixSync`, PomaiDB performs an `fsync()` on the parent directory whenever a new file is created or renamed to prevent "missing files" after a power loss.

## 3. Crash Recovery

Upon restart, PomaiDB performs a "Warm Scan":
- **Segment Validation**: Checks trailing checksums for every segment.
- **Incomplete Work**: Any temporary files or orphan segments detected via the directory scan are safely discarded.
- **Strict Checkpointing**: The index only advances once the underlying data segments have been confirmed as durable.

## 4. Hardware Sympathy (PSOW)

PomaiDB assumes **Power-Safe Overwrite (PSOW)** capability is absent unless explicitly detected.
- **Sector Alignment**: All writes are aligned to `512` or `4096` byte sector boundaries to prevent partial-sector writes that can lead to hardware-level corruption during power failure.
- **Zero-Copy Safety**: Memory-mapped regions used for reads are never used for direct-write mutations to avoid "scribbling" on disk during a process crash.

## 5. Error Handling Rigor

PomaiDB treats all I/O errors as fatal to the current operation:
- **ENOSPC**: Graceful halt and rollback to the last known-good checkpoint.
- **EIO**: Immediate closure of the shard to prevent further corruption of the storage medium.
