# PomaiDB

PomaiDB is a local-first, single-machine, single-user vector database focused on deterministic recovery, immutable read views, and explicit search contracts for quality vs. latency. It is an embedded C++20 library with optional server mode and tools for verification and export.

## Table of Contents
- [What is Pomai in 30 seconds](#what-is-pomai-in-30-seconds)
- [Database contracts (non-negotiable)](#database-contracts-non-negotiable)
- [Architecture overview](#architecture-overview)
- [Math](#math)
- [On-disk layout](#on-disk-layout)
- [Build & Run](#build--run)
- [Benchmarking](#benchmarking)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Security & Safety](#security--safety)
- [Not implemented](#not-implemented)

## What is Pomai in 30 seconds
PomaiDB stores vectors on disk with WAL + checkpointing, serves lock-free read snapshots, and exposes a single explicit search contract: you must choose `SearchMode::Latency` or `SearchMode::Quality`. It is a **local-first** library; the optional server speaks a small binary protocol for PING/CREATE/UPSERT/SEARCH only. It is **not** a multi-tenant cloud service and **not** an ML framework.

## Database contracts (non-negotiable)
- **Local-first.** Data lives on the machine. No background “phone home,” no cloud dependency.
- **Durability & recovery invariants.** WAL + atomic checkpoints are authoritative. Recovery loads the latest checkpoint and replays WAL to a consistent state. A manifest describes the checkpoint, dictionary, and index artifacts.
- **Immutability (grain/membrane).**
  - **Grain**: an immutable segment built from a frozen memtable.
  - **Membrane**: a published, immutable view composed of grains and the current live snapshot.
- **No locks on hot read path.** Reads load immutable `ShardState` snapshots via atomic/shared_ptr swaps; no global mutex on search.
- **Quality vs. latency modes.** Search requests must declare `SearchMode::Quality` or `SearchMode::Latency`. Quality mode refuses to pretend success when filtered results are insufficient and returns `SearchStatus::InsufficientResults` with explicit flags.

## Architecture overview

### Data flow (ASCII)
```
             +---------------------+
             |  PomaiDB (API)      |
             +----------+----------+
                        |
                        v
             +----------+----------+
             | MembraneRouter      |  <-- query contract + budgets
             +----+----------+-----+
                  |          |
                  |          +--------------------------+
                  v                                     v
           +------+-----+                        +------+-----+
           |  Shard 0   | ...                    |  Shard N   |
           +------+-----+                        +------+-----+
                  |                                     |
                  v                                     v
     +------------+----------+              +-----------+-----------+
     | Immutable segments    |              | Immutable segments     |
     | + live snapshot       |              | + live snapshot        |
     +------------+----------+              +-----------+-----------+
                  |
                  v
            WAL + Checkpoint
```

### Membrane publish (ASCII)
```
  Seed (memtable) --> freeze --> Grain (immutable)
       |                               |
       +---- live snapshot ------------+
                     |
              atomic ShardState swap
                     |
               Membrane view
```

### Module map
- `include/pomai/api/*` – public API types and entrypoints.
- `include/pomai/core/*` – membrane, shard, seed (immutable grains), spatial router.
- `include/pomai/index/*` – `OrbitIndex` graph index.
- `include/pomai/storage/*` – WAL, snapshot, manifest, verification.
- `include/pomai/concurrency/*` – bounded queues, thread pools, memory manager.
- `include/pomai/util/*` – utility helpers (CPU kernels, search utils, fixed-topk).
- `src/*` mirrors the same structure, plus `src/tools` for CLI utilities.

## Math

### L2 distance
For vectors \(x, y \in \mathbb{R}^d\):
\[
\mathrm{L2}(x, y) = \sum_{i=1}^{d} (x_i - y_i)^2
\]
PomaiDB stores scores as **negative L2** for top-k ranking.

### SQ8 quantization (encode/decode)
For each dimension \(i\):
\[
q_i = \mathrm{clamp}_{[0,255]}\left(\mathrm{round}\left(\frac{x_i - \min_i}{\mathrm{scale}_i}\right)\right)
\]
\[
\hat{x}_i = \min_i + q_i \cdot \mathrm{scale}_i
\]
Where \(\mathrm{scale}_i = (\max_i - \min_i) / 255\). PomaiDB stores SQ8 values per grain and dequantizes during exact rerank.

### Rerank: candidates vs. exact
Search is two-stage:
1. **Candidate generation** from quantized data / graph traversal (fast, approximate).
2. **Exact rerank** on dequantized vectors for the final top-k.
The search budget controls how far candidate generation can expand.

### Filter selectivity & candidate amplification
Filtered search adapts `filtered_candidate_k` and `graph_ef` based on selectivity. If the filter passes only \(s\) of candidates, expected amplification to reach `topk` is roughly \(1 / s\). PomaiDB expands candidates up to configured caps and surfaces budget hits explicitly.

## On-disk layout
```
<db_dir>/
  MANIFEST
  centroids.bin                (optional)
  checkpoints/
    chk_<epoch>_<lsn>.pomai
  wal/
    shard-<n>.wal
  meta/
    dict_<epoch>_<lsn>.bin
  indexes/
    idx_<epoch>_<lsn>_<kind>.bin
```

### Atomic commit protocol (checkpoint)
1. Write snapshot to `checkpoints/<file>.tmp`.
2. `fdatasync` the snapshot file.
3. Atomic rename to `checkpoints/<file>`.
4. `fsync` the checkpoints directory.
5. Write dictionary and index artifacts (`meta/`, `indexes/`) using the same temp + rename + `fsync` protocol.
6. Write manifest to `MANIFEST.tmp`, `fdatasync`, then atomic rename to `MANIFEST` and `fsync` the db root.

### Checksum verification rules
- Snapshot and dictionary files are CRC64-checked on read.
- `pomai_verify` validates the manifest, snapshot, dictionary, and index CRCs.

## Build & Run

### Prerequisites
- C++20 compiler (GCC 10+ or Clang 11+)
- CMake 3.10+
- Linux (WSL acceptable)

### Build (Release/Debug)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

### Embedded benchmark
```bash
./build/bench_embedded --queries=100 --topk=10 --rerank_k=0 --graph_ef=0
```

### WAL benchmark
```bash
./build/bench_wal --threads 4 --batch 64 --dim 128 --duration 5 --wait-durable
```

### Server (optional)
The server is a lightweight binary protocol (PING/CREATE_COLLECTION/UPSERT_BATCH/SEARCH). It is **not** a SQL or HTTP server.
```bash
./build/pomai-server --config config/pomai.yaml
# or
./build/pomai-server config/pomai.yaml
```
Note: configuration file parsing is currently a stub; defaults from `pomai_server_main.cpp` are used when the file is missing or unparsed.

### Debug logging
- Embedded: set `DbOptions::debug_logging = true`.
- Server: set `log_level: debug` in the config.

### Tools
```bash
./build/pomai_verify <db_dir>
./build/pomai_dump <db_dir> <output.tsv>
```

## Benchmarking
Recommended commands:
```bash
./build/bench_embedded --queries=200 --topk=10
./build/bench_concurrent_search
./build/bench_wal --threads 4 --batch 64 --duration 5
```

Metrics:
- **recall@10**: fraction of true top-10 neighbors found.
- **p50/p95/p99**: latency percentiles (lower is better).
- **ingest ops/s**: vectors written per second.
- **filtered search**: read `filtered_budget_exhausted`, `filtered_missing_hits`, and retry counters to judge selectivity impact.

## Testing
```bash
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Sanitizers:
```bash
cmake -S . -B build-asan -DPOMAI_SANITIZE=ASAN
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

Convenience test runner:
```bash
./tests/test.sh
```

## Troubleshooting
- **OOM during benchmarks**: the embedded benchmark keeps a full copy of vectors for ground truth. Use fewer vectors or sampled ground truth; disable exact rerank if the budget is tight.
- **p99 spikes**: reduce `SearchMode::Quality` usage, tighten `filtered_candidate_k`/`graph_ef`, and ensure WAL/checkpoint I/O isn’t throttled.
- **Build failures**: confirm a C++20 compiler and CMake 3.10+; remove old build directories after toolchain changes.

## Roadmap
- Cosine metric search support (currently only L2 is implemented).
- Structured server config parsing.
- Compaction telemetry and backpressure knobs.

## Security & Safety
- Local-first data handling: PomaiDB does not exfiltrate data or require network access.
- Crash safety is explicit: WAL + checkpointing plus checksum verification are mandatory for recovery.
- The server exposes **no authentication**; run it on trusted hosts only.

## Not implemented
These items are **not** in the current codebase and should not be assumed:
- SQL or CLI interface.
- Multi-tenant cloud service.
- Dataset splitting, ML training pipelines, or streaming training data.
- SimHash or 4-bit compression.
- Cosine distance search (configuration exists, execution is L2-only today).
