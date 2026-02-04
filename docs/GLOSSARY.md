# Glossary

This document serves as the single source of truth for terminology in PomaiDB.

## Core Concepts

### Shard
A horizontal partition of the database.
- **Role**: Unit of write serialization (single writer thread) and consistency.
- **Persistence**: Has its own WAL and directory of segments.
- **Isolation**: No cross-shard ordering or atomicity.

### Membrane
A logical namespace or collection of vectors (analogous to a "Table" or "Collection" in other DBs).
- **Default Membrane**: The membrane used when no name is specified (internally `__default__`).
- **Isolation**: Membranes are physically separated on disk (separate subdirectories).
- **Status**: Currently, named membranes are **ephemeral** in the API (lost on restart) due to implementation gaps.

### VectorId
A 64-bit unsigned integer uniquely identifying a vector within a membrane.
- **Constraint**: Must be unique per membrane. Resubmitting an existing ID is an overwrite (Upsert).

## Persistence & Storage

### WAL (Write-Ahead Log)
Append-only log file receiving all writes before they are applied to memory.
- **Rotation**: Rotated to a new file when it exceeds a size threshold (default 64MB) or on Freeze.
- **Durability**: Configured via `FsyncPolicy`.

### Segment
Immutable on-disk file containing a sorted run of vector data.
- **Creation**: Created only during `Freeze`.
- **Format**: `pomai.seg.v1` (version 2 payload). contains header, data, and CRC.

### Manifest
File tracking the structural state of the database.
- **Global Manifest**: `MANIFEST` at root. Tracks membranes (currently unwired).
- **Shard Manifest**: `manifest.current` in shard dir. Tracks active segments.

## Visibility & Consistency

### Snapshot
A consistent view of the database at a point in time.
- **Composition**: A set of immutable segments + a frozen MemTable (if any).
- **Acquisition**: Readers acquire a snapshot at the start of an operation (`Search`, `Get`).
- **Ordering**: Monotonically increasing versions.

### Freeze
An operation that:
1. Rotates the active MemTable to valid frozen state.
2. Flushes frozen state to a new Segment.
3. Updates Manifest.
4. Resets WAL.
- **Semantics**: Provides durability (persisted to segment) and visibility (published to new snapshot).

### Soft Freeze
An automatic, in-memory rotation of the MemTable when it fills up (default 5000 items).
- **Semantics**: Makes recent writes visible to new snapshots, but does *not* persist to Segment.

### FsyncPolicy
Configuration controlling when `fsync` is called on WAL.
- `kNever`: Relies on OS page cache. Fast, unsafe.
- `kOnFlush`: `fsync` only on explicit `Flush()` call.
- `kAlways`: `fsync` on every `Put`/`Delete`. Slow, safe.

