# PomaiDB

PomaiDB is a local-first, single-machine, embedded vector database built in C++20. It implements a sharded, actor-model architecture with **lock-free snapshot isolation** for high-concurrency read workloads.

> [!NOTE]
> **Status**: Alpha (Hardening Phase).
> **Consistency Model**: Snapshot Isolation with Bounded Staleness.

## System Model & Guarantees

### 1. Concurrency & Isolation
PomaiDB separates Read and Write paths to ensure readers never block writers and vice-versa.

- **Writes (Put/Delete)**: Serialized per-shard via a bounded mailbox (Actor Model).
- **Reads (Search/Get/Exists)**: **Lock-Free**. Reads access an atomic `ShardSnapshot`.
- **Visibility (Bounded Staleness)**:
    - Writes go to an **Active MemTable** (invisible to readers).
    - Data becomes visible ONLY after a **Soft Freeze** (Snapshot update).
    - Soft Freeze triggers automatically (default: 5000 items) or via manual `Freeze()`.
    - **No Read-Your-Writes**: A thread writing data will not see it in `Get()` until the next freeze.

### 2. Storage Hierarchy
Data flows through immutable states:
$$ \text{Active (Mutable)} \xrightarrow{\text{Freeze}} \text{Frozen (Immutable)} \xrightarrow{\text{Flush}} \text{Segment (Disk)} $$

- **Active MemTable**: Mutable, write-only, invisible to readers.
- **Frozen MemTables**: Immutable, visible to readers (in Snapshot).
- **Segments**: Immutable on-disk files, visible to readers (in Snapshot).
- **WAL**: Durability log (Prefix Consistency).

### 3. Snapshot Semantics
A `ShardSnapshot` gives a consistent view of the database at a point in time.
- **Immutability**: Snapshots reference only immutable data structures (Frozen MemTables + Segments).
- **Ordering**: Snapshots represent a strictly monotonic prefix of the WAL history.
- **Lifetime**: Memory is automatically reclaimed when the last reader releases the snapshot (`std::shared_ptr`).

## Data Model
- **Vectors**: Dense Float32 only. No quantization (yet).
- **Distance**: Euclidean (L2).
- **Search**: exact re-ranking of results.
    - *Current Limitation*: Algorithm is brute-force scan over snapshot items (IVF bypass for correctness).

## Persistence & Durability
- **WAL**: All Puts/Deletes are appended to WAL before memory ack.
- **Crash Recovery**: On restart, WAL is replayed into MemTable.
- **Startup Visibility**: Replayed data is automatically rotated to Frozen on startup, making it immediately visible.

## Build & Test

### Dependencies
- C++20 Compiler
- CMake 3.20+

### Commands
```bash
# Build
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build -j

# Run Tests
ctest --test-dir build --output-on-failure
```

## Known Limitations
1. **Bounded Staleness**: Reads lag writes by ~5000 ops (configurable).
2. **Directory Sync**: Segment creation relies on OS/Filesystem for directory entry sync (strict `fsync` on dir not fully guaranteed on all platforms).
3. **IVF Bypass**: Search currently performs a linear scan of the snapshot for maximum correctness (TSAN clean), ignoring the IVF index.
4. **Single Tenant**: Designed for embedded, single-app use.
