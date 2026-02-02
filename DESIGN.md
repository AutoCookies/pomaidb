# PomaiDB Architecture & Design

## Overview
PomaiDB is an embedded C++20 vector database designed for:
- **Simplicity**: Single static library, no external runtime dependencies.
- **Performance**: Sharded architecture, lock-free queues, AVX-optimized distance calculations.
- **Durability**: Segmented WAL, periodic checkpoints, and crash-safe manifests.

## Core Concepts

### 1. Sharded Runtime
- **Actor Model**: The DB is partitioned into `N` shards. Each shard runs on a dedicated thread.
- **Mailbox**: Communication happens via bounded, lock-free queues (MPMC or MPSC) per shard.
- **State Isolation**: Each shard manages its own subset of data (MemTable, WAL, Segment files) completely independently to minimize contention.

### 2. Storage Layer
- **MemTable**: In-memory mutable structure for recent writes.
- **WAL (Write-Ahead Log)**: Sequential log for crash recovery. Writes are appended to WAL before acknowledging to the user.
- **Segments (SST-like)**: Immutable files storing older data. Periodically created via Checkpoints.
- **Manifest**: Metadata file tracking the set of active segments and configuration. Updated atomically.

### 3. Membranes (Namespaces)
- **Isolation**: Membranes provide logical separation of data (like collections/tables).
- **Metadata**: Each membrane has its own configuration (dimensions, distance metric, etc.).

### 4. Search
- **IVF Coarse**: Centroid-based routing to prune the search space.
- **Fanout**: Queries are fanned out to relevant shards (or all shards if brute-force).
- **TopK Merge**: Results from shards are merged deterministically.

## Data Flow

### Write Path (Put)
1. **API**: `DB::Put(id, vector)` is called.
2. **Routing**: `id` is hashed to determine the target shard.
3. **Queue**: Operation is pushed to the target shard's mailbox.
4. **Shard Processing**:
    - Append to WAL.
    - Update MemTable.
    - Acknowledge completion.

### Read Path (Get)
1. **API**: `DB::Get(id)` is called.
2. **Routing**: `id` hashed to target shard.
3. **Lookup**: Shard checks MemTable -> Immutable MemTables -> Segments (with bloom filter/index).

### Search Path
1. **API**: `DB::Search(query, k)` is called.
2. **Candidate Selection**: Use Coarse Index (if enabled) to select relevant centroids/shards.
3. **Fanout**: Send sub-queries to involved shards.
4. **Shard Processing**: Scan MemTable + Segments for candidates. Compute distances. Return local TopK.
5. **Merge**: Aggregator merges local TopKs into global TopK.

## Invariants & Safety
- **No Undefined Behavior**: All code compiles with -Wall -Wextra -Wconversion and runs under ASAN/UBSAN/TSAN.
- **Crash Safety**: `kill -9` at any point results in a consistent state upon recovery (replayed from WAL).
- **Thread Safety**: Public API is thread-safe. Internal classes assume single-threaded access (actor model).

## Directory Structure
```
root/
├── manifest/   # Global and per-membrane manifests
├── shard_0/
│   ├── wal/
│   ├── segments/
│   └── ...
├── shard_1/
└── ...
```
