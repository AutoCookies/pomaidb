# PomaiDB

<img src="./assets/logo.png">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**The predictable vector database for the edge of things.**

PomaiDB is an **embedded, single-threaded vector database** written in C++20. It is built for environments where stability, hardware longevity, and deterministic behavior matter more than theoretical peak throughput. Single-threaded event-loop execution, zero-copy reads, and an append-only storage model keep your edge devices predictable—and your SD cards alive.

---

## Why PomaiDB?

### SD-Card Savior

Most databases punish flash storage with random writes. Wear leveling and write amplification on SD cards and eMMC lead to early failure and unpredictable latency. PomaiDB is designed around **append-only, log-structured storage**: new data is written sequentially at the tail. Deletes and updates are represented as tombstones. No random seeks, no in-place overwrites. **The I/O pattern your storage was built for.**

### Single-Threaded Sanity

No mutexes. No lock-free queues. No race conditions or deadlocks. PomaiDB runs a **strict single-threaded event loop**—similar in spirit to Redis or Node.js. Every operation (ingest, search, freeze, flush) runs to completion in order. You get deterministic latency, trivial reasoning about concurrency, and a hot path optimized for CPU cache locality without any locking overhead.

### Zero-OOM Guarantee

PomaiDB integrates with **palloc**, a vector-first allocator that provides O(1) arena-style allocation and optional hard memory limits. Combined with the single-threaded design, you can bound memory usage and avoid the surprise OOMs that plague heap-heavy workloads on constrained devices.

---

## Technical Highlights

- **Architecture:** Shared-nothing, single-threaded event loop. One logical thread of execution; no worker threads, no thread pools in the core path.
- **Storage:** Log-structured, append-only. Tombstone-based deletion; sequential flush of in-memory buffer to disk. Optional explicit `Flush()` from the application loop.
- **Memory:** Powered by **palloc** (mmap-backed allocator). Core and C API use only palloc/mmap—no raw `malloc` or `new`. Arena-backed buffers for ingestion; optional hard limits for embedded and edge deployments.
- **I/O:** Sequential write-behind; **mmap** zero-copy reads for persisted segments. Designed for SD-card and eMMC longevity first, NVMe-friendly by construction.
- **Hardware:** Optimized for **ARM64** (Raspberry Pi, Orange Pi, Jetson) and **x64** servers. Single-threaded design avoids NUMA and core-pinning complexity.

---

## Installation & Usage

### Build

Requires a C++20 compiler and CMake 3.20+.

```bash
git clone --recursive https://github.com/YOUR_ORG/pomaidb.git
cd pomaidb
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Smaller clone (embedded / CI):** Use a shallow clone and slim the palloc submodule to skip unneeded directories (saves ~6MB+ and reduces history size):

```bash
git clone --depth 1 --recursive https://github.com/YOUR_ORG/pomaidb.git
cd pomaidb
./scripts/slim_palloc_submodule.sh
```

Then build as above. The script configures sparse checkout for `third_party/palloc` so that `media/`, `test/`, `bench/`, and `contrib/` are not checked out (pomaidb does not need them to build).

### Quick Start (C++20)

Create a database, ingest vectors, and run a search. Vectors are written through an arena-backed buffer and, when you choose, flushed sequentially to disk.

```cpp
#include "pomai/pomai.h"
#include <cstdio>
#include <memory>
#include <vector>

int main() {
    pomai::DBOptions opt;
    opt.path = "/data/vectors";
    opt.dim = 384;
    opt.shard_count = 1;
    opt.fsync = pomai::FsyncPolicy::kNever;

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) return 1;

    // Ingest: vectors are buffered in arena-backed storage
    std::vector<float> vec(opt.dim, 0.1f);
    st = db->Put(1, vec);
    if (!st.ok()) return 1;

    st = db->Put(2, vec);
    if (!st.ok()) return 1;

    // Flush buffer to disk when you're ready (e.g. from your event loop)
    st = db->Flush();
    if (!st.ok()) return 1;

    // Freeze memtable to segment for search (optional; enables segment-based search)
    st = db->Freeze("__default__");
    if (!st.ok()) return 1;

    // Query: zero-copy reads from mmap'd segments where possible
    pomai::SearchResult result;
    st = db->Search(vec, 5, &result);
    if (!st.ok()) return 1;

    for (const auto& hit : result.hits)
        std::printf("id=%llu score=%.4f\n", static_cast<unsigned long long>(hit.id), hit.score);

    db->Close();
    return 0;
}
```

Link against the PomaiDB static library and, when using the palloc integration, the palloc library. See the repository's build instructions for details.

### Run benchmarks (one by one)

From a configured build directory, run each benchmark in order:

```bash
# Build (benchmarks are built by default with the main targets)
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Run all benchmarks one by one (short workloads)
../scripts/run_benchmarks_one_by_one.sh
```

Or run individual executables: `./bench_baseline`, `./comprehensive_bench --dataset small`, `./ingestion_bench 10000 128`, `./rag_bench 100 64 32`, `./ci_perf_bench`, `./bin/bench_cbrs`, `./benchmark_a` (use `POMAI_BENCH_LOW_MEMORY=1` for a shorter run).

### Minimal clone (embedded)

For the smallest footprint on embedded devices:

1. **Shallow clone** to avoid full history: `git clone --depth 1 --recursive <repo>`.
2. **Slim palloc submodule** (saves ~6MB): after clone, run `./scripts/slim_palloc_submodule.sh` so `third_party/palloc` omits `media/`, `test/`, `bench/`, and `contrib/`.
3. **Optional sparse checkout of pomaidb**: for a production embedded build you can exclude `benchmarks/`, `examples/`, or `tools/` via your own sparse-checkout if you do not need them at build time.

### Edge deployments & failure semantics

For recommended settings on real edge devices (build flags, durability policies, backpressure, and how PomaiDB behaves on power loss), see:

- `docs/EDGE_DEPLOYMENT.md` — **edge-device configuration & failure behavior**
- `docs/FAILURE_SEMANTICS.md` — low-level WAL / manifest crash semantics

### Docker: run benchmarks

Build the image, then run benchmarks in constrained (IoT/Edge) or server-style containers:

```bash
docker compose build
docker compose up
```

Each service runs `benchmark_a` by default; when the benchmark finishes, the container exits. To run a single environment or a different benchmark:

```bash
# IoT (128 MB RAM, low-memory targets)
docker compose run --rm pomai-iot-starvation

# Edge (512 MB RAM)
docker compose run --rm pomai-edge-gateway

# Server (2 GB RAM, no CPU cap)
docker compose run --rm pomai-server-lite
```

Override the default command to run another benchmark (e.g. `ingestion_bench`, `comprehensive_bench`, `rag_bench`, `ci_perf_bench`):

```bash
docker compose run --rm pomai-iot-starvation ingestion_bench
```

To get a shell and run benchmarks manually:

```bash
docker compose run --rm pomai-iot-starvation sh
# inside container: benchmark_a, ingestion_bench, benchmark_a --list, etc.
```

Reports can be written to the host under `./bench` by overriding the command (see `docker-compose.yml` comment).

For very small containers (e.g. 128 MiB for `pomai-iot-starvation`), PomaiDB uses **memtable backpressure** to leave RAM
for the OS and other processes. You can tune this via:

- `POMAI_MEMTABLE_FLUSH_THRESHOLD_MB` – approximate upper bound for memtable RAM (e.g. `48` for ~48 MiB)
- `POMAI_BENCH_LOW_MEMORY=1` – shrink benchmark workloads and enable a conservative default threshold

When the memtable grows beyond this threshold, PomaiDB automatically runs a blocking `Freeze()`, which slows ingestion
just enough to avoid the OOM killer while keeping the OS responsive.

---

## Use Cases

- **Camera & object detection:** Embed frames or crops, run similarity search on-device. Single-threaded ingestion fits naturally into a camera pipeline; append-only storage avoids wearing out SD cards in 24/7 deployments.
- **Edge RAG:** Ingest document chunks and embeddings on the device; run retrieval-augmented generation with local vector search. Bounded memory and deterministic latency simplify deployment on Raspberry Pi, Orange Pi, and Jetson.
- **Offline semantic search:** Index documents or media on a NAS or edge node. Sequential writes and mmap reads are friendly to both SSDs and consumer flash; no need for a separate search server.

---

## Discovery Tags

**Keywords:** embedded vector database, single-threaded, C++20, append-only, log-structured, zero-copy, mmap, palloc, edge AI, IoT, Raspberry Pi, Orange Pi, Jetson, ARM64, SD card longevity, vector search, similarity search, RAG, semantic search.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
