<div align="center">

# 🍇 PomaiDB

<img src="./assets/logo.png" alt="PomaiDb Logo"/>

### **The vector database that runs on the edge — not in the cloud.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![Platforms](https://img.shields.io/badge/Platforms-Linux%20%7C%20ARM64%20%7C%20x86__64-333333?logo=linux)](https://github.com/AutoCookies/pomaidb)
[![Python](https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white)](python/README.md)

**[⭐ Star](https://github.com/AutoCookies/pomaidb/stargazers)** · **[🍴 Fork](https://github.com/AutoCookies/pomaidb/fork)** · **[📖 Docs](docs/)** · **[🤝 Contribute](CONTRIBUTING.md)**

</div>

---

**PomaiDB** is a **lean, embedded vector database** in pure C++20. No servers. No API keys. No internet.  
It runs **in-process** on your device — Raspberry Pi, phone, laptop, IoT — with a tiny footprint, crash-safe storage, and SIMD-accelerated search.  
Built for **offline RAG**, **on-device agents**, and **private embedding search**.

> *"A database should be like a pomegranate: atomic grains of data, each protected by an immutable membrane."*

---

## ✨ Why PomaiDB?

| You want… | PomaiDB gives you |
|-----------|-------------------|
| **Privacy** | Data never leaves the device. No cloud, no telemetry. |
| **Offline-first** | Works without the internet. Survives power loss and reboots. |
| **Small & fast** | ~2–5 MB static, ARM64/NEON and x86 SIMD. Real-time search on low-power hardware. |
| **Embedded** | Single binary, in-process. No daemon, no Docker, no K8s. |
| **Crash-safe** | WAL + atomic manifest. Recover after battery death or SD corruption. |
| **Simple** | C++ and C API; Python via `pip install pomaidb`. No heavy runtime. |

**Ideal for:** edge AI, **personal RAG** (hybrid lexical + vector), local semantic search, IoT embeddings, on-device agents, and anywhere you need **vector search without the cloud**. Use a RAG membrane for chunk-level ingest and search (`create_rag_membrane`, `put_chunk`, `search_rag` in C and Python).

---

## 🚀 Quick Start

### C++

```cpp
#include <pomai/pomai.h>

int main() {
    pomai::DBOptions opt;
    opt.path = "./my-vectors.pdb";
    opt.dim = 384;   // e.g. sentence-transformers
    opt.shard_count = 1;

    std::unique_ptr<pomai::DB> db;
    pomai::DB::Open(opt, &db);

    std::vector<float> vec(384, 0.42f);
    db->Put(1, vec);
    db->Freeze("__default__");

    pomai::SearchResult res;
    db->Search(vec.data(), 10, &res);
    for (const auto& hit : res.hits)
        std::cout << hit.id << " " << hit.score << "\n";

    db->Close();
    return 0;
}
```

### Python

```bash
# Build the C library first (see Build below), then:
pip install ./python
export POMAI_C_LIB=/path/to/build/libpomai_c.so   # or .dylib on macOS
```

```python
import pomaidb

db = pomaidb.open_db("/tmp/my_db", dim=128, shards=1)
pomaidb.put_batch(db, ids=[1, 2, 3], vectors=[[0.1]*128, [0.2]*128, [0.3]*128])
pomaidb.freeze(db)
results = pomaidb.search_batch(db, queries=[[0.15]*128], topk=5)
pomaidb.close(db)
```

[Full Python API →](docs/PYTHON_API.md)

---

## 📦 Build & Test

```bash
git clone https://github.com/AutoCookies/pomaidb.git
cd pomaidb
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the cross-engine benchmark (PomaiDB vs hnswlib, FAISS):

```bash
./benchmark_all.sh
```

---

## 🏗️ What’s inside

- **Single-process, sharded** — lock-free reads, one writer per shard.
- **Atomic Freeze** — readers see a consistent snapshot; no torn reads.
- **WAL + manifest** — durable commits; recovery from crash or power loss.
- **HNSW + segments** — graph index and on-disk segments; batch search with configurable parallelism.
- **SimSIMD** — NEON (ARM64) and AVX (x86) for fast distance (L2, inner product, cosine).
- **Membranes** — separate namespaces (e.g. `VECTOR`, `RAG`) in one DB. **RAG** membranes support chunk ingest (token IDs + optional embedding) and hybrid search (lexical + vector rerank).
- **C + C++ API** — easy FFI for Python, Node, or any language that talks C.

[Versioning & API stability →](docs/VERSIONING.md) · [Production & embedded assessment →](docs/PRODUCTION_AND_EMBEDDED_ASSESSMENT.md)

---

## 🛡️ Edge-first, not cloud-first

Most vector DBs assume servers and networks. PomaiDB assumes **your device**:

- ✅ Runs **offline** — no API keys, no latency, no vendor lock-in  
- ✅ **Crash-resilient** — WAL replay, manifest fallback  
- ✅ **Low write amplification** — gentle on SD cards and flash  
- ✅ **Small memory** — thousands of vectors on modest RAM  
- ✅ **ARM64-optimized** — NEON kernels for phones, Pi, Jetson  

---

## 🤝 Contributing

We care about **stability**, **correctness**, and **real edge hardware**.  
Whether it’s a bug fix, a benchmark on a Raspberry Pi, or a new binding — we’d love your help.

👉 **[CONTRIBUTING.md](CONTRIBUTING.md)** — how to contribute, what we prioritize, and how we work.

---

## 📜 License

[Apache 2.0](LICENSE) — use, modify, and distribute freely.

---

<div align="center">

**If you’re building private, fast, local AI — give us a ⭐ and share the repo.**

*PomaiDB · Made for the edge.*

</div>
