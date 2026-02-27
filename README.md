# PomaiDB — Edge Vector Database

<div align="center">
    <img src="./assets/logo.png" alt="PomaiDB Logo"/>
</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/Standard-C%2B%2B20-red.svg)](https://en.cppreference.com/w/cpp/20)
[![Platforms](https://img.shields.io/badge/Platforms-ARM64%20%7C%20x86__64-orange.svg)]()
[![GitHub stars](https://img.shields.io/github/stars/AutoCookies/pomaidb?style=social)](https://github.com/AutoCookies/pomaidb)

**PomaiDB** is a **lean, high-performance embedded vector database** built in pure C++20 — designed specifically for **Edge AI** on resource-constrained devices: phones, Raspberry Pi, IoT boards, embedded systems, and even browsers via WASM.

No servers. No cloud dependencies. No unnecessary layers.  
Just fast, private, local vector search that runs directly on your device.

> “A database should be like a Pomegranate: atomic grains of data, each protected by an immutable membrane.”

## 🎯 Purpose & Core Philosophy

In the world of on-device AI, personal agents, offline RAG, and private long-term memory, existing vector databases are often too heavy, too server-oriented, or too memory-hungry.

**PomaiDB exists to solve exactly that problem:**

- Be **truly embedded** — runs in-process, single binary, tiny footprint (~2–5 MB static possible)
- Deliver **real-time performance** on low-power ARM64 hardware (Raspberry Pi, phones, Jetson Nano)
- Guarantee **privacy & safety** — no network calls, crash-resilient, power-loss tolerant
- Offer **zero-copy efficiency** — data moves from storage to search kernel without redundant copies
- Stay **simple and predictable** — deterministic behavior, no background threads eating battery

PomaiDB is built for developers who want **local-first, offline-capable AI** without compromising speed or reliability.

## 💎 Key Design Pillars

- **Single-process embedded core** — no server, no external services
- **Sharded actor model** — lock-free reads, dedicated writer per shard
- **Atomic Freeze semantics** — readers always see a consistent, published snapshot
- **Native ARM64 / NEON SIMD** — optimized brute-force distance computation
- **Typed membranes** — `VECTOR` for embeddings, `RAG` for hybrid text + vector
- **WAL + atomic manifest** — crash-safe, survives sudden power loss
- **Minimal dependencies** — pure C++20 + CMake (FAISS optional for advanced indexing)

## ⚡ Quick Start (C++)

```cpp
#include <pomai/pomai.h>
#include <vector>
#include <iostream>

int main() {
    pomai::DBOptions opt;
    opt.path = "./my-vault.pdb";
    opt.dim = 384;                             // e.g. sentence-transformers/all-MiniLM-L6-v2
    opt.shard_count = std::thread::hardware_concurrency();

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Open failed: " << st.ToString() << "\n";
        return 1;
    }

    // Ingest a vector
    std::vector<float> embedding(384, 0.42f); // your model output
    db->Put(1337, embedding.data());

    // Make data visible (atomic snapshot)
    db->Freeze("__default__");

    // Search
    pomai::SearchResult res;
    db->Search(embedding.data(), 10, &res);

    for (const auto& hit : res.hits) {
        std::cout << "Hit: ID=" << hit.id << " | Score=" << hit.score << "\n";
    }

    return 0;
}
```

## 🛡️ Why Edge-First Matters

Most vector databases are built for cloud or powerful servers.  
PomaiDB is built for **your device**:

- Runs offline — no internet, no API keys
- Survives battery death or sudden reboot
- Minimizes SD card / flash wear (low write amplification)
- Uses tiny memory footprint even with thousands of vectors
- Optimized for ARM64 — native NEON for distance calculations

## 📦 Build & Run

```bash
git clone https://github.com/AutoCookies/pomaidb
cd pomaidb
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Run tests:
```bash
ctest
```

## 🤝 Contributing

We welcome every idea that helps make PomaiDB more stable, faster, and more useful on real edge hardware.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📜 License

Apache License 2.0 — free to use, modify, and distribute.

---

<p align="center">
Made with ❤️ for builders who want <b>private, fast, local AI</b> on every device.<br/>
<b>Star ⭐ if you're building the future of Edge AI!</b>
</p>