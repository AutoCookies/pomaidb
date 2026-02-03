# PomaiDB

PomaiDB is a local-first, single-machine, embedded vector database built in C++20. It implements a sharded, actor-model architecture for concurrent vector storage and retrieval.

> [!WARNING]
> This project is currently in **Alpha** state. Some architectural claims from previous documentation (SQ8 quantization, lock-free reads, search quality contracts) are not yet implemented. This document reflects the current actual state of the codebase.

## System Model & Formal Definitions

### 1. Sharded Actor Model & Concurrency
The database is partitioned into $N$ **Shards**.
*   **Writes**: Serialized per shard via a **Bounded MPSC Queue** (Actor Model).
*   **Reads (Search/Get)**: **Non-blocking / Lock-Free**.
    *   Reads access a **Snapshot** (Atomic `shared_ptr`).
    *   Snapshots contain immutable Segments and Frozen MemTables.
    *   **Bounded Staleness**: Reads do not see the active mutable MemTable. Data becomes visible after a "Soft Freeze" (default: 5000 items).
*   **Search Scaling**: Uses a fixed **ThreadPool** for fan-out, independent of shard actor loops.

### 2. Storage Hierachy
Data within a Shard is organized into:
*   **Active MemTable**: Mutable, write-only.
*   **Frozen MemTables**: Immutable, read/write (flush pending). Part of Snapshot.
*   **Segments**: Immutable on-disk files. Part of Snapshot.
*   **WAL**: Durability log.

**State Transition**:
$$ \text{Active} \xrightarrow{\text{Soft Freeze}} \text{Frozen} \xrightarrow{\text{Flush}} \text{New Segment} $$

### 3. Vector Space & Distance
*   **Vectors**: Dense inputs $x \in \mathbb{R}^d$ (stored as `float32`).
*   **Distance**: Two metrics are supported in configuration, but currently `L2` (Euclidean) is the primary path.
    $$ \text{L2}(x, y) = \sum_{i=1}^{d} (x_i - y_i)^2 $$
*   **Score**: Negative L2 distance (higher is better, 0 is exact match).

### 4. Search Pipeline
Search is performed via a **Fan-out/Fan-in** pattern:
1.  **Request**: `DB::Search` fans out the query to all $N$ shards.
2.  **Local Search**: Each shard executes:
    *   **Candidate Generation**: Uses `IvfCoarse` (Centroid-based) to prune clusters, or falls back to Brute Force if not trained/empty.
    *   **Scoring**: Exact re-ranking using stored `float32` vectors.
    *   **Selection**: Maintains a local min-heap of top-$k$ results.
3.  **Merge**: The coordinating thread waits for all shards and merges `N` local top-$k$ lists into a global top-$k$.

**Note**: There is currently **no quantization** (SQ8/PQ). All vectors are stored and compared in full precision.

---

## Storage & Durability Model

### On-Disk Layout
```
<db_dir>/
  MANIFEST               # Global layout (membranes)
  membranes/
    <name>/
      MANIFEST           # Membrane configuration
      shards/
        <id>/
          manifest.current  # List of active segments
          wal_*.log         # Write-Ahead Logs
          seg_*.dat         # Immutable Segment Files
```

### Durability Guarantees
*   **WAL**: Operations (Put/Delete) are appended to the WAL and `fdatasync`'d (configurable via `FsyncPolicy`).
*   **Crash Consistency**:
    *   WAL uses CRC32C checksums per frame.
    *   Recovery replays valid frames until the first corruption/truncation (prefix consistency).
*   **Checkpoints (Segments)**:
    *   `Freeze` writes a new segment file -> `fsync` -> `close`.
    *   Updates `manifest.new` -> `fsync` -> Atomic Rename to `manifest.current`.
    *   **Limitation**: Directory entries for segments may not be strictly synced on all filesystems immediately after creation.

---

## Concurrency & Memory Guarantees

### Read/Write Isolation Model
* **Writes** are serialized per shard via the actor mailbox.
* **Reads** do NOT execute on the shard mailbox.
  * Reads load an immutable `ShardSnapshot` via atomic `shared_ptr`.
  * Snapshots include only immutable data structures.
* **Isolation Level**: Snapshot Isolation with bounded staleness.
  * Reads observe a consistent snapshot.
  * Newly written data becomes visible only after a Soft Freeze.

### Memory Safety
* Snapshots reference only immutable memory.
* Snapshot publication is atomic.
* Memory is reclaimed automatically when no readers hold references.
* No reader ever observes partially constructed state.

### Explicit Non-Goals
*   **No MVCC with per-transaction timestamps**: Simple "latest snapshot" semantics.
*   **No read-your-writes guarantees across shards**: Eventual consistency for reads.
*   **No multi-version historical queries**: Time-travel queries are not supported.
*   **No Multi-Tenancy**: Designed for single-user embedded workloads.

---

## Build & Run

### Prerequisites
*   C++20 Compiler (GCC 10+, Clang 11+, MSVC 19.28+)
*   CMake 3.20+

### Build
```bash
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build -j
```

### Testing
```bash
ctest --test-dir build --output-on-failure
```

---

## Roadmap vs Reality

| Feature | Claimed (Old) | Actual (Current) |
| :--- | :--- | :--- |
| **Quantization** | SQ8 / 4-bit | **Float32 Only** |
| **Concurrency** | Lock-free Readers | **Lock-Free Snapshot Reads** |
| **Search** | Quality vs Latency Modes | **Simple Top-K only** |
| **Durability** | Strict Atomic Checkpoints | **WAL-based (Manifest sync partial)** |

---

## Production Readiness
**Verdict**: **NO (Experimental, but Scalable Reads Implemented)**

* Snapshot-based, non-blocking reads are implemented and scale with CPU cores.
* Write path is serialized per shard and may become a bottleneck under heavy ingest.
* Quantization and advanced indexing are not yet implemented.
* Durability edge-cases (directory fsync) require hardening.
