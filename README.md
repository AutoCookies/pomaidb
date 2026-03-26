<div align="center">

# PomaiDB

<img src="./assets/logo.png" width="200px">

**The predictable edge-native database for multimodal AI memory.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=for-the-badge&logo=c%2B%2B)](https://en.wikipedia.org/wiki/C%2B%2B20)

---

[What is PomaiDB?](#-what-is-pomaidb) • [Why PomaiDB?](#-why-pomaidb) • [Architecture](#-technical-highlights) • [Installation](#-installation--usage) • [Quick Start](#-quick-start-c20) • [Benchmarks](#-performance) • [Contributors](#-contributors)

</div>

---

## 🚀 What is PomaiDB?

PomaiDB is a **small-footprint, production-ready vector store** that runs natively on ARM64 (Raspberry Pi, Orange Pi, Jetson) and x86_64. It is designed to be **linked into your application** as a static or shared library and driven via a simple C++ API or a C API with Python bindings. Unlike distributed or server-oriented vector databases, PomaiDB assumes a **single process, single thread of execution**: one event loop, one storage engine, one logical database. That constraint is intentional. It yields predictable latency, trivial concurrency reasoning, and an I/O model that flash storage (SD cards, eMMC) can sustain for years.

**Core capabilities (current):**

- **Vector ingestion and search** — Put vectors (with optional metadata), run approximate nearest-neighbor search (ANN) with configurable index types (IVF, HNSW), batch search, and point queries. All with optional scalar quantization (SQ8) to cut memory and bandwidth.
- **Membranes** — Logical collections with separate dimensions, sharding, and indexes. Supported kinds: `kVector`, `kRag`, `kGraph`, `kText`, `kTimeSeries`, `kKeyValue`, `kMeta`, `kSketch`, `kBlob`, `kSpatial`, `kMesh`, `kSparse`, `kBitset`.
- **Mesh LOD manager** — `kMesh` now supports asynchronous multi-LOD generation (high-poly + decimated levels) driven by `TaskScheduler`, with latency-first auto selection and explicit high-detail override per mesh query.
- **Typed membrane APIs** — Native C++ + C APIs across all supported membrane kinds with strict low-RAM caps.
- **Hybrid & multimodal search** — `QueryOrchestrator` supports vector + lexical + graph traversal paths with heuristic execution ordering, bounded frontier RAM, and metadata partition hints (`device_id`/`location_id`).
- **ObjectLinker (Phase 2)** — shared GID links across vector/graph/mesh (`LinkObjects`), so vector hits can expand into linked graph vertex + mesh ids.
- **Edge-native connectivity (Phase 3)** — embedded HTTP endpoints (`/health`, `/metrics`, `/ingest/meta/...`, `/ingest/vector/...`) and lightweight MQTT/WebSocket-style ingestion listener for direct edge sensor intake, with optional bearer-token/token auth, rate limiting, JSON error model, idempotency keys, and ACK/ERR replies.
- **Edge analytical aggregates (mini-OLAP)** — `Sum`, `Avg`, `Min`, `Max`, `Count`, `Top-K` over planner/runtime post-filter result sets.
- **Virtual time-travel** — query constraints by `as_of_ts` and `as_of_lsn` for replay/debug style reads.
- **Hardware wear-aware maintenance** — write-byte counters at WAL/segment layers and endurance-aware compaction biasing for flash longevity.
- **No-train AI dispatch** — deterministic heuristic inference path can classify/score all current membrane datatypes without any training step.
- **Offline-first Edge RAG** — Ingest documents (chunk → embed → store) and retrieve context (embed query → search → format chunk text) entirely on-device. Zero-copy chunking, a pluggable `EmbeddingProvider` (mock for tests; ready for a small local model), and strict memory limits so the pipeline fits in 64–128 MB RAM.
- **Virtual File System (VFS)** — Storage and environment operations go through abstract `Env` and file interfaces. Default backend is POSIX; an in-memory backend (`InMemoryEnv`) supports tests and non-POSIX targets. No direct `<unistd.h>` or `<fcntl.h>` in core code.
- **Zero-OOM philosophy** — Bounded memtable size, backpressure (auto-freeze when over threshold), and optional integration with **palloc** for arena-style allocation and hard memory caps. Runtime caps include lifecycle entries, text docs, query frontier, KV entries, sketch entries, and blob bytes.
- **Edge security hardening** — AES-256-GCM encryption-at-rest on WAL path, key manager primitives (arm/wipe), and anomaly-triggered key wipe hooks in orchestrated query flow.

PomaiDB does **not** aim to replace distributed vector DBs or to maximize throughput under heavy concurrency. It aims to be the **reliable, embeddable vector and RAG engine** for edge AI: cameras, gateways, NAS, and custom OSes where you need search and RAG without the cloud and without killing your storage.

---

## ❓ Why PomaiDB?

### 🛡️ SD-Card Savior

Most databases punish flash storage with random writes. Wear leveling and write amplification on SD cards and eMMC lead to early failure and unpredictable latency. PomaiDB is designed around **append-only, log-structured storage**: new data is written sequentially at the tail. Deletes and updates are represented as tombstones. No random seeks, no in-place overwrites. **The I/O pattern your storage was built for.**

### 🧵 Single-Threaded Sanity

No mutexes. No lock-free queues. No race conditions or deadlocks. PomaiDB runs a **strict single-threaded event loop**—similar in spirit to Redis or Node.js. Every operation (ingest, search, freeze, flush) runs to completion in order. You get deterministic latency, trivial reasoning about concurrency, and a hot path optimized for CPU cache locality without any locking overhead.

### 🛑 Zero-OOM Guarantee

PomaiDB integrates with **palloc** (and compatible allocators) for O(1) arena-style allocation and optional hard memory limits. Combined with single-threaded design and configurable backpressure, you can bound memory usage and avoid the surprise OOMs that plague heap-heavy workloads on constrained devices. The Edge RAG pipeline respects max document size, max chunk size, and batch limits so that under 64–128 MB RAM the system never exceeds a safe threshold.

### 🌐 Offline-First RAG

The built-in RAG pipeline needs **no external API**. You ingest documents (text → chunk → embed → store) and retrieve context (query → embed → search → formatted text) entirely inside PomaiDB. A mock embedding provider is included for tests and demos; the interface is designed so a small local model (e.g. GGML/llama.cpp) can be plugged in later without changing pipeline code.

---

## 🛠️ Technical Highlights

- **🏗️ Architecture** — Shared-nothing, single-threaded event loop. One logical thread, deterministic sequencing. `DbImpl` + `MembraneManager` + `QueryPlanner/QueryOrchestrator` provide typed multi-membrane execution for C++, C, and Python bindings.
- **💾 Storage** — Log-structured, append-only. Tombstone-based deletion; sequential flush of in-memory buffer to disk. Optional explicit `Flush()` from the application loop. VFS abstraction (`Env`, `SequentialFile`, `RandomAccessFile`, `WritableFile`, optional `FileMapping`) so core code has no OS-specific includes.
- **🧠 Memory** — Optional **palloc** (mmap-backed or custom allocator). Core and C API can use palloc for control structures and large buffers; RAG pipeline uses configurable limits and batch sizes. Arena-backed buffers for ingestion; optional hard limits for embedded and edge deployments.
- **🔌 I/O** — Sequential write-behind; zero-copy reads (mmap where available via VFS, or buffered I/O). Designed for SD-card and eMMC longevity first, NVMe-friendly by construction.
- **🔍 RAG** — Zero-copy chunking (`std::string_view`), `EmbeddingProvider` interface, optional chunk text storage in RAG engine, and a unified `RagPipeline` with `IngestDocument` and `RetrieveContext`. C API: `pomai_rag_pipeline_create`, `pomai_rag_ingest_document`, `pomai_rag_retrieve_context` (and buffer-based variant); Python: `ingest_document`, `retrieve_context`.
- **🛡️ Hardening** — Stress/soak, crash-replay, power-loss, SD-fault injection, endurance-aware write tracking, and encryption overhead benchmarks are part of the repository test/bench matrix.
- **💻 Hardware** — Optimized for **ARM64** (Raspberry Pi, Orange Pi, Jetson) and **x64** servers. Single-threaded design avoids NUMA and core-pinning complexity.

---

## 📊 Performance

PomaiDB is engineered for predictable ingestion, retrieval, and maintenance on constrained edge hardware.

> [!NOTE]
> **Latest benchmark run:** via `./scripts/run_benchmarks_one_by_one.sh` (full suite, exit code `0`).

**Run Device (Edge-class laptop):**
- **Model:** HP ProBook 450 G5
- **CPU:** Intel(R) Core(TM) i7-8550U @ 1.80GHz (8 cores)
- **RAM:** 16GB
- **Storage:** SATA SSD

| Benchmark | Workload | Latest Result |
| :--- | :--- | :--- |
| **Comprehensive Search** | 10K vecs / 1K queries / top-k=10 | Mean **18.55 ms**, P99 **28.52 ms**, QPS **53.89**, Recall@10 **100%** |
| **Ingestion Throughput** | 10K vectors @ 128-dim | **31,004 vectors/sec** (~15.14 MB/s), avg **32.25 us/vector** |
| **RAG Lexical** | Chunked retrieval pipeline | **0.068 ms** |
| **RAG Hybrid** | Lexical + vector candidates | **0.064 ms** |
| **CI Perf Gate** | 2K vecs / 300 queries (3 iters) | Ingest **56,847.3 qps**, Search p50 **2509.65 us**, p99 **3470.43 us** |
| **CBRS (single)** | Routed query profile | Query **30.9 qps**, p99 **21820.6 us**, Recall@10 **100%** |
| **Low-memory Edge Churn** | 5 cycles constrained run | Throughput **1844.43 vec/s**, Peak RSS **51.2 MiB**, PASS |
| **Quantization** | Float/SQ8/FP16/1-bit comparison | Throughput: Float **56.37**, SQ8 **56.90**, FP16 **56.42**, 1-bit **49.41** |
| **Mesh LOD** | 4,096 triangles / 5K volume ops | Auto-LOD **276.005 ms**, High-detail **1664.509 ms** (~6.0x faster auto path) |


*Note: Benchmarks are device/profile dependent. Throughput and latency vary by CPU class, storage medium, fsync policy, and memory limit profile.*

---

## ⚙️ Installation & Usage

### 🛠️ Build

Requires a C++20 compiler and CMake 3.20+.

**Vulkan headers:** the Khronos bundle (Vulkan-Hpp + `vulkan/*.h`) is expected under **`third_party/vulkan/include/`** in the repo root. CMake prints `[pomai] Vulkan headers: …` at configure time.

**Examples:** see **`examples/README.md`** (C++, C ABI, Python, Go, JS/TS, RAG quickstart).

```bash
# Clone the repository
git clone --recursive https://github.com/pomagrenate/pomaidb
cd pomaidb

# Build the project
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Tests (full suite):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

**Smaller clone (embedded / CI):**

```bash
git clone --depth 1 --recursive https://github.com/pomagrenate/pomaidb
cd pomaidb
./scripts/slim_palloc_submodule.sh
```

Then build as above.

---

## 🏁 Quick Start

### 🔹 Quick Start (C++20)

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

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) return 1;

    std::vector<float> vec(opt.dim, 0.1f);
    db->Put(1, vec);
    db->Flush();
    db->Freeze("__default__");

    pomai::SearchResult result;
    db->Search(vec, 5, &result);

    for (const auto& hit : result.hits)
        std::printf("id=%llu score=%.4f\n", (unsigned long long)hit.id, hit.score);

    db->Close();
    return 0;
}
```

### 🔹 Quick Start (Edge RAG, C++)

```cpp
#include "pomai/pomai.h"
#include "pomai/rag/embedding_provider.h"
#include "pomai/rag/pipeline.h"

int main() {
    pomai::DBOptions opt;
    opt.path = "/tmp/rag_db";
    opt.dim = 4;
    std::unique_ptr<pomai::DB> db;
    pomai::DB::Open(opt, &db);

    pomai::MembraneSpec rag;
    rag.name = "rag";
    rag.kind = pomai::MembraneKind::kRag;
    db->CreateMembrane(rag);

    pomai::MockEmbeddingProvider provider(4);
    pomai::RagPipeline pipeline(db.get(), "rag", 4, &provider);

    pipeline.IngestDocument(1, "Your document text here.");
    std::string context;
    pipeline.RetrieveContext("your query", 5, &context);

    db->Close();
    return 0;
}
```

### 🔹 Quick Start (Python)

```python
import pomaidb

db = pomaidb.open_db("/tmp/rag_db", dim=4)
pomaidb.create_rag_membrane(db, "rag", dim=4)

# Ingest document (chunk + embed + store)
pomaidb.ingest_document(db, "rag", doc_id=1, text="Your document text here.")

# Retrieve context for a query
context = pomaidb.retrieve_context(db, "rag", "your query", top_k=5)
pomaidb.close(db)
```

---

## 💡 Use Cases

- **📸 Camera & object detection** — Embed frames or crops, run similarity search on-device. Single-threaded ingestion fits naturally into a camera pipeline.
- **🧠 Edge RAG** — Ingest document chunks and embeddings on the device; run retrieval-augmented generation with local vector search and formatted context.
- **🔍 Offline semantic search** — Index documents or media on a NAS or edge node. Sequential writes and zero-copy reads are friendly to both SSDs and consumer flash.
- **🖥️ Custom & bare-metal OSes** — The VFS layer allows swapping the POSIX backend for an in-memory or custom backend.

---

## 📖 Documentation

- **📑 Edge release criteria**: [`docs/EDGE_RELEASE.md`](docs/EDGE_RELEASE.md)
- **🌍 Edge deployment**: [`docs/EDGE_DEPLOYMENT.md`](docs/EDGE_DEPLOYMENT.md)
- **💥 Failure semantics**: [`docs/FAILURE_SEMANTICS.md`](docs/FAILURE_SEMANTICS.md)
- **🐍 Python ctypes API**: [`docs/PYTHON_API.md`](docs/PYTHON_API.md)
- **📌 ABI versioning**: [`docs/VERSIONING.md`](docs/VERSIONING.md)

## 👥 Contributors

<table style="border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td align="center" style="border: none; padding: 10px;">
      <a href="https://github.com/pomagrenate">
        <img src="https://github.com/pomagrenate.png" width="80px;" alt="pomagrenate" style="border-radius: 50%;"/><br />
        <sub><b>pomagrenate</b></sub>
      </a>
    </td>
    <td align="center" style="border: none; padding: 10px;">
      <a href="https://github.com/quanvanskipli">
        <img src="https://github.com/quanvanskipli.png" width="80px;" alt="quanvanskipli" style="border-radius: 50%;"/><br />
        <sub><b>quanvanskipli</b></sub>
      </a>
    </td>
    <td align="center" style="border: none; padding: 10px;">
      <a href="https://github.com/claude">
        <img src="https://github.com/claude.png" width="80px;" alt="claude" style="border-radius: 50%;"/><br />
        <sub><b>claude</b></sub>
      </a>
    </td>
    <td align="center" style="border: none; padding: 10px;">
      <a href="https://github.com/Roto0flame">
        <img src="https://github.com/Roto0flame.png" width="80px;" alt="Roto0flame" style="border-radius: 50%;"/><br />
        <sub><b>Roto0flame</b></sub>
      </a>
    </td>
  </tr>
</table>

## ⭐ Star History

<div align="center">
  <a href="https://star-history.com/#pomagrenate/pomaidb&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pomagrenate/pomaidb&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pomagrenate/pomaidb&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pomagrenate/pomaidb&type=Date" width="100%" />
    </picture>
  </a>
</div>

---

## 🏷️ Discovery Tags

**Keywords:** embedded vector database, single-threaded, C++20, append-only, log-structured, zero-copy, mmap, palloc, edge AI, IoT, Raspberry Pi, Orange Pi, Jetson, ARM64, SD card longevity, vector search, similarity search, RAG, semantic search, offline RAG, VFS, virtual file system.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
