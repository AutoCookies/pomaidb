# PomaiDB

<img src="./assets/logo.png">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**The predictable vector database for the edge of things.**

PomaiDB is an **embedded, single-threaded vector database** written in C++20, built for edge devices, IoT gateways, and environments where stability, hardware longevity, and deterministic behavior matter more than theoretical peak throughput. It combines **log-structured storage**, **zero-copy reads**, and an **offline-first Edge RAG pipeline** so you can run vector search and retrieval-augmented generation entirely on-device—no cloud APIs, no random writes, no surprise OOMs.

---

## What is PomaiDB?

PomaiDB is a **small-footprint, production-ready vector store** that runs natively on ARM64 (Raspberry Pi, Orange Pi, Jetson) and x86_64. It is designed to be **linked into your application** as a static or shared library and driven via a simple C++ API or a C API with Python bindings. Unlike distributed or server-oriented vector databases, PomaiDB assumes a **single process, single thread of execution**: one event loop, one storage engine, one logical database. That constraint is intentional. It yields predictable latency, trivial concurrency reasoning, and an I/O model that flash storage (SD cards, eMMC) can sustain for years.

**Core capabilities:**

- **Vector ingestion and search** — Put vectors (with optional metadata), run approximate nearest-neighbor search (ANN) with configurable index types (IVF, HNSW), batch search, and point queries. All with optional scalar quantization (SQ8) to cut memory and bandwidth.
- **Membranes** — Logical collections (vector-only or RAG) with separate dimensions, sharding, and indexes. Create multiple membranes in one database (e.g. `default` for embeddings, `rag` for document chunks).
- **Offline-first Edge RAG** — Ingest documents (chunk → embed → store) and retrieve context (embed query → search → format chunk text) entirely on-device. Zero-copy chunking, a pluggable `EmbeddingProvider` (mock for tests; ready for a small local model), and strict memory limits so the pipeline fits in 64–128 MB RAM.
- **Virtual File System (VFS)** — Storage and environment operations go through abstract `Env` and file interfaces. Default backend is POSIX; an in-memory backend (`InMemoryEnv`) supports tests and non-POSIX targets. No direct `<unistd.h>` or `<fcntl.h>` in core code.
- **Zero-OOM philosophy** — Bounded memtable size, backpressure (auto-freeze when over threshold), and optional integration with **palloc** for arena-style allocation and hard memory caps. The RAG pipeline enforces max document size, max chunk size, and batch limits so ingestion never grows without bound.

PomaiDB does **not** aim to replace distributed vector DBs or to maximize throughput under heavy concurrency. It aims to be the **reliable, embeddable vector and RAG engine** for edge AI: cameras, gateways, NAS, and custom OSes where you need search and RAG without the cloud and without killing your storage.

---

## Why PomaiDB?

### SD-Card Savior

Most databases punish flash storage with random writes. Wear leveling and write amplification on SD cards and eMMC lead to early failure and unpredictable latency. PomaiDB is designed around **append-only, log-structured storage**: new data is written sequentially at the tail. Deletes and updates are represented as tombstones. No random seeks, no in-place overwrites. **The I/O pattern your storage was built for.**

### Single-Threaded Sanity

No mutexes. No lock-free queues. No race conditions or deadlocks. PomaiDB runs a **strict single-threaded event loop**—similar in spirit to Redis or Node.js. Every operation (ingest, search, freeze, flush) runs to completion in order. You get deterministic latency, trivial reasoning about concurrency, and a hot path optimized for CPU cache locality without any locking overhead.

### Zero-OOM Guarantee

PomaiDB integrates with **palloc** (and compatible allocators) for O(1) arena-style allocation and optional hard memory limits. Combined with single-threaded design and configurable backpressure, you can bound memory usage and avoid the surprise OOMs that plague heap-heavy workloads on constrained devices. The Edge RAG pipeline respects max document size, max chunk size, and batch limits so that under 64–128 MB RAM the system never exceeds a safe threshold.

### Offline-First RAG

The built-in RAG pipeline needs **no external API**. You ingest documents (text → chunk → embed → store) and retrieve context (query → embed → search → formatted text) entirely inside PomaiDB. A mock embedding provider is included for tests and demos; the interface is designed so a small local model (e.g. GGML/llama.cpp) can be plugged in later without changing pipeline code.

---

## Technical Highlights

- **Architecture** — Shared-nothing, single-threaded event loop. One logical thread; no worker threads or thread pools in the core path. Full DB implementation (`DbImpl`) with membrane manager supports both vector and RAG membranes; C API and Python bindings use this same engine.
- **Storage** — Log-structured, append-only. Tombstone-based deletion; sequential flush of in-memory buffer to disk. Optional explicit `Flush()` from the application loop. VFS abstraction (`Env`, `SequentialFile`, `RandomAccessFile`, `WritableFile`, optional `FileMapping`) so core code has no OS-specific includes.
- **Memory** — Optional **palloc** (mmap-backed or custom allocator). Core and C API can use palloc for control structures and large buffers; RAG pipeline uses configurable limits and batch sizes. Arena-backed buffers for ingestion; optional hard limits for embedded and edge deployments.
- **I/O** — Sequential write-behind; zero-copy reads (mmap where available via VFS, or buffered I/O). Designed for SD-card and eMMC longevity first, NVMe-friendly by construction.
- **RAG** — Zero-copy chunking (`std::string_view`), `EmbeddingProvider` interface, optional chunk text storage in RAG engine, and a unified `RagPipeline` with `IngestDocument` and `RetrieveContext`. C API: `pomai_rag_pipeline_create`, `pomai_rag_ingest_document`, `pomai_rag_retrieve_context` (and buffer-based variant); Python: `ingest_document`, `retrieve_context`.
- **Hardware** — Optimized for **ARM64** (Raspberry Pi, Orange Pi, Jetson) and **x64** servers. Single-threaded design avoids NUMA and core-pinning complexity.

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

**Tests:** `cmake .. -DPOMAI_BUILD_TESTS=ON` then build and run the test targets (e.g. `rag_chunking_test`, `rag_embedding_test`, `rag_pipeline_test`, `c_api_test`, `env_test`).

**Smaller clone (embedded / CI):** Use a shallow clone and slim the palloc submodule to skip unneeded directories (saves ~6MB+ and reduces history size):

```bash
git clone --depth 1 --recursive https://github.com/YOUR_ORG/pomaidb.git
cd pomaidb
./scripts/slim_palloc_submodule.sh
```

Then build as above.

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

    std::vector<float> vec(opt.dim, 0.1f);
    st = db->Put(1, vec);
    if (!st.ok()) return 1;
    st = db->Put(2, vec);
    if (!st.ok()) return 1;

    st = db->Flush();
    if (!st.ok()) return 1;
    st = db->Freeze("__default__");
    if (!st.ok()) return 1;

    pomai::SearchResult result;
    st = db->Search(vec, 5, &result);
    if (!st.ok()) return 1;

    for (const auto& hit : result.hits)
        std::printf("id=%llu score=%.4f\n", static_cast<unsigned long long>(hit.id), hit.score);

    db->Close();
    return 0;
}
```

### Quick Start (Edge RAG, C++)

Create a RAG membrane, ingest a document through the pipeline (chunk → embed → store), and retrieve context for a query—all offline.

```cpp
#include "pomai/pomai.h"
#include "pomai/rag/embedding_provider.h"
#include "pomai/rag/pipeline.h"
#include <memory>
#include <string>

int main() {
    pomai::DBOptions opt;
    opt.path = "/tmp/rag_db";
    opt.dim = 4;
    opt.shard_count = 2;
    std::unique_ptr<pomai::DB> db;
    if (!pomai::DB::Open(opt, &db).ok()) return 1;

    pomai::MembraneSpec rag;
    rag.name = "rag";
    rag.dim = 4;
    rag.shard_count = 2;
    rag.kind = pomai::MembraneKind::kRag;
    if (!db->CreateMembrane(rag).ok() || !db->OpenMembrane("rag").ok()) return 1;

    pomai::MockEmbeddingProvider provider(4);
    pomai::RagPipelineOptions pipe_opts;
    pipe_opts.max_chunk_bytes = 512;
    pomai::RagPipeline pipeline(db.get(), "rag", 4, &provider, pipe_opts);

    std::string doc = "Your document text here. It will be chunked and embedded locally.";
    if (!pipeline.IngestDocument(1, doc).ok()) return 1;

    std::string context;
    if (!pipeline.RetrieveContext("your query", 5, &context).ok()) return 1;
    // Use context for your local LLM or downstream task.

    db->Close();
    return 0;
}
```

### Quick Start (Python)

After building, set `POMAI_C_LIB` to the path of `libpomai_c.so` (or `.dylib` on macOS). Then use the offline RAG flow:

```python
import pomaidb

db = pomaidb.open_db("/tmp/rag_db", dim=4, shards=2)
pomaidb.create_rag_membrane(db, "rag", dim=4, shard_count=2)

# Ingest document (chunk + embed + store, no external API)
pomaidb.ingest_document(db, "rag", doc_id=1, text="Your document text here.")

# Retrieve context for a query
context = pomaidb.retrieve_context(db, "rag", "your query", top_k=5)

# Low-level: put_chunk / search_rag also available
pomaidb.close(db)
```

See `examples/rag_quickstart.py` and `scripts/rag_smoke.py` for full examples.

### Run benchmarks

From a configured build directory:

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
../scripts/run_benchmarks_one_by_one.sh
```

Or run individual executables: `./bench_baseline`, `./comprehensive_bench --dataset small`, `./ingestion_bench 10000 128`, `./rag_bench 100 64 32`, `./ci_perf_bench`, `./benchmark_a` (use `POMAI_BENCH_LOW_MEMORY=1` for a shorter run).

**Python CIFAR-10 benchmark (end-to-end):** Create a venv, install from `requirements.txt`, build the C library, then run the benchmark (uses ctypes against `libpomai_c.so`; downloads real CIFAR-10 by default):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target pomai_c
.venv/bin/python benchmarks/python_cifar10_feature_bench.py
```

Use `--no-download` if the dataset is already under `data/`; use `--allow-fake-fallback` only to fall back to synthetic data when offline.

### Edge deployments & failure semantics

For recommended settings on real edge devices (build flags, durability policies, backpressure, and how PomaiDB behaves on power loss), see:

- `docs/EDGE_DEPLOYMENT.md` — edge-device configuration & failure behavior  
- `docs/FAILURE_SEMANTICS.md` — low-level WAL / manifest crash semantics  

### Docker

Build the image, then run benchmarks in constrained (IoT/Edge) or server-style containers:

```bash
docker compose build
docker compose up
```

To run a single environment or a different benchmark:

```bash
docker compose run --rm pomai-iot-starvation
docker compose run --rm pomai-edge-gateway
docker compose run --rm pomai-server-lite
```

Override the default command to run another benchmark (e.g. `ingestion_bench`, `rag_bench`). For small containers (e.g. 128 MiB), PomaiDB uses memtable backpressure; tune via `POMAI_MEMTABLE_FLUSH_THRESHOLD_MB` and `POMAI_BENCH_LOW_MEMORY=1`.

---

## Use Cases

- **Camera & object detection** — Embed frames or crops, run similarity search on-device. Single-threaded ingestion fits naturally into a camera pipeline; append-only storage avoids wearing out SD cards in 24/7 deployments.
- **Edge RAG** — Ingest document chunks and embeddings on the device; run retrieval-augmented generation with local vector search and formatted context. No external embedding or search API; bounded memory and deterministic latency for Raspberry Pi, Orange Pi, and Jetson.
- **Offline semantic search** — Index documents or media on a NAS or edge node. Sequential writes and zero-copy reads are friendly to both SSDs and consumer flash; no separate search server required.
- **Custom & bare-metal OSes** — The VFS layer (`Env`, file abstractions) allows swapping the POSIX backend for an in-memory or custom backend, so PomaiDB can be adapted to non-POSIX or bare-metal environments without changing core storage or RAG logic.

---

## Discovery Tags

**Keywords:** embedded vector database, single-threaded, C++20, append-only, log-structured, zero-copy, mmap, palloc, edge AI, IoT, Raspberry Pi, Orange Pi, Jetson, ARM64, SD card longevity, vector search, similarity search, RAG, semantic search, offline RAG, VFS, virtual file system.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
