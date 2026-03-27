# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Makes PomaiDB Unique

PomaiDB is designed as a **multimodal embedded database for edge devices** with several distinctive features:

- **True Multimodal Storage**: Unlike traditional vector databases, PomaiDB supports 12+ membrane types (vectors, RAG, graphs, text, time series, key-value, metadata, sketches, spatial data, meshes, sparse vectors, bitsets) allowing storage and querying of diverse data modalities in a single database.

- **Edge-Native Single-Threaded Architecture**: Designed for deterministic latency on constrained hardware with no mutexes, lock-free queues, or race conditions - similar to Redis/Node.js event loop but optimized for flash storage longevity.

- **Zero-OOM Guarantee**: Integrated with palloc for arena-style allocation with hard memory limits, combined with backpressure mechanisms to prevent out-of-memory crashes on edge devices.

- **Offline-First Edge RAG**: Complete retrieval-augmented generation pipeline that runs entirely on-device (ingest → chunk → embed → store → retrieve) without external APIs, featuring zero-copy chunking and pluggable embedding providers.

- **Multimodal Query Orchestration**: Hybrid search across different membrane types (vector + lexical + graph traversal) with heuristic execution ordering and bounded frontier RAM.

- **Flash-Optimized Storage**: Append-only, log-structured design with tombstone-based deletion that minimizes random writes and extends SD/eMMC card lifespan.

- **Built-in Edge Features**: HTTP endpoints, MQTT/WebSocket-style ingestion, hardware wear-aware maintenance, encryption-at-rest, and mini-OLAP analytical aggregates.

## Development Commands

### Build
```bash
# Standard release build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Build with tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Edge-optimized build (smaller binary, less debug)
cmake .. -DCMAKE_BUILD_TYPE=Release -DPOMAI_EDGE_BUILD=ON
```

### Tests
```bash
# Run full test suite
cd build
ctest --test-dir . --output-on-failure

# Run a single test (replace TestName with actual test name)
cd build
ctest -R TestName --output-on-failure

# Run tests with verbose output
cd build
ctest --verbose
```

### Benchmarks
```bash
# Build benchmarks
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPOMAI_BUILD_BENCH=ON
cmake --build build -j$(nproc)

# Run all benchmarks
./scripts/run_benchmarks_one_by_one.sh

# Run specific benchmark
cd build
ctest -R bench_ --output-on-failure
```

### Code Quality
```bash
# Check for malloc/new usage (should use palloc)
./scripts/check_no_malloc_new.sh

# Verify API contract
./scripts/check_api_contract.sh
```

### Quick Start Examples
See `examples/` directory for language-specific examples:
- C++: `examples/quickstart_cpp/`
- C: `examples/quickstart_c/`
- Python: `examples/quickstart_python/`
- RAG: `examples/rag_quickstart/`

## Architecture Overview

### What Makes PomaiDB a Multimodal Embedded Database

PomaiDB's architecture is specifically designed to be a **multimodal embedded database for edge devices**, combining several unique capabilities:

- **Unified Multimodal Storage**: Single database engine that handles vectors, text, graphs, time series, spatial data, 3D meshes, and more through its membrane system, enabling true multimodal AI applications on edge devices.

- **Deterministic Edge Performance**: Single-threaded event loop eliminates concurrency complexity while providing predictable latency crucial for real-time edge applications.

- **Flash-Optimized for Longevity**: Append-only, log-structured storage minimizes write amplification on SD/eMMC storage, extending device lifespan.

- **Hardware-Aware Resource Management**: Integration with palloc provides hard memory caps and arena-style allocation, preventing OOM crashes on memory-constrained devices.

- **Complete On-Device AI Pipeline**: Offline-first RAG with zero-copy chunking and pluggable embedding providers enables local AI without cloud dependencies.

### Core Design Principles
- **Single-threaded event loop**: All operations (ingest, search, freeze, flush) run to completion in order, providing deterministic latency and trivial concurrency reasoning.
- **Shared-nothing architecture**: One logical thread, one storage engine, one logical database per process.
- **Zero-OOM philosophy**: Bounded memtable size, backpressure (auto-freeze when over threshold), and optional integration with palloc for arena-style allocation and hard memory caps.
- **Edge-native storage**: Append-only, log-structured storage with tombstone-based deletion, designed for SD-card and eMMC longevity.
- **Virtual File System (VFS)**: Storage and environment operations go through abstract `Env` and file interfaces. Default backend is POSIX; an in-memory backend supports tests and non-POSIX targets.

### Key Components
- **DbImpl**: Main database implementation handling core operations (Put, Get, Search, Flush, Freeze, Close).
- **MembraneManager**: Manages logical collections (membranes) with separate dimensions, sharding, and indexes. Supported kinds include kVector, kRag, kGraph, kText, kTimeSeries, kKeyValue, kMeta, kSketch, kSpatial, kMesh, kSparse, kBitset.
- **QueryPlanner/QueryOrchestrator**: Plans and executes hybrid/multimodal searches across membranes with heuristic execution ordering, bounded frontier RAM, and metadata partition hints.
- **Storage Engine**: Log-structured, append-only storage with sequential flush of in-memory buffer to disk. Uses WAL for crash recovery.
- **RAG Pipeline**: Zero-copy chunking (`std::string_view`), `EmbeddingProvider` interface, and unified `RagPipeline` with `IngestDocument` and `RetrieveContext` methods.
- **Memory Management**: Optional palloc (mmap-backed or custom allocator) for O(1) arena-style allocation and hard memory limits. Core and C API can use palloc for control structures and large buffers.
- **I/O Layer**: Sequential write-behind; zero-copy reads (mmap where available via VFS, or buffered I/O). Designed for SD-card and eMMC longevity first, NVMe-friendly by construction.

### Membrane Types (Multimodal Capabilities)
Each membrane kind enables specific multimodal capabilities:
- `kVector`: Vector storage with ANN search (IVF, HNSW) - for embeddings and similarity search
- `kRag`: Retrieval-Augmented Generation pipeline storage - for document retrieval and context augmentation
- `kGraph`: Graph storage for relationships and linkages - for knowledge graphs and entity relationships
- `kText`: Raw text storage - for storing and querying unstructured text
- `kTimeSeries`: Time-series data storage - for sensor data, metrics, and temporal analysis
- `kKeyValue`: Simple key-value store - for configuration and metadata storage
- `kMeta`: Metadata storage - for flexible schema-less data tagging
- `kSketch`: Probabilistic data structures (HyperLogLog, CountMinSketch) - for approximate counting and frequency estimation
- `kSpatial`: Geospatial data storage - for location-based services and mapping applications
- `kMesh`: 3D mesh storage with LOD management - for AR/VR, robotics, and spatial computing
- `kSparse`: Sparse vector storage - for efficient storage of high-dimensional sparse data
- `kBitset`: Bitmask operations and filtering - for fast set operations and feature flags

### Important Directories
- `src/`: Core implementation (C++)
- `include/`: Public headers
- `sdk/`: Language bindings (Python, etc.)
- `tests/`: GoogleTest unit and integration tests
- `benchmarks/`: Performance benchmarks
- `scripts/`: Utility scripts for building, testing, and benchmarking
- `examples/`: Quickstart examples in multiple languages
- `docs/`: Detailed documentation (edge release, deployment, failure semantics, etc.)

## Common Development Tasks

### Adding a New Membrane Type (Extending Multimodal Capabilities)
1. Define the membrane kind in `include/pomai/types.h` (add to `MembraneKind` enum)
2. Implement the membrane class in `src/core/membranes/` (follow existing patterns like `VectorMembrane` or `RagMembrane`)
3. Register the membrane in `MembraneManager::CreateMembrane()` and `MembraneManager::GetMembrane()`
4. Add appropriate tests in `tests/membranes/` (test basic operations, edge cases, and integration with query orchestrator)
5. Update documentation if needed (mention in membrane types overview)
6. Consider if the new membrane type should participate in hybrid queries (update `QueryOrchestrator` if needed)

### Developing for Edge Devices
1. **Memory Optimization**: Use `palloc` for arena-style allocation; avoid unbounded growth
2. **Flash Longevity**: Favor sequential writes; minimize random I/O operations
3. **Deterministic Latency**: Avoid blocking operations; keep operations bounded and predictable
4. **Power Efficiency**: Profile CPU usage; leverage SIMD instructions when available
5. **Testing on Target Hardware**: Use scripts in `benchmarks/` to validate performance on actual edge devices
6. **Verify No System Calls**: Run `./scripts/check_no_malloc_new.sh` to ensure proper memory practices

### Modifying Storage Layer (Edge-Optimized)
1. Changes typically affect `src/core/storage/`
2. Ensure WAL consistency and crash-recovery properties are maintained (critical for power-loss resilience)
3. Run soak/power-loss tests (see `tests/storage/`)
4. Verify append-only property and tombstone handling (no random writes)
5. Test with `./scripts/edge_release_print_sizes.sh` to check binary footprint
6. Validate performance characteristics with `./scripts/run_benchmarks_one_by_one.sh`

### Working with RAG Pipeline (On-Device AI)
1. Core logic in `src/core/rag/`
2. Embedding provider interface in `include/pomai/rag/embedding_provider.h` (implement for local models)
3. Chunking strategies in `src/core/rag/chunking/` (optimize for zero-copy on edge)
4. Test with `scripts/rag_smoke.py`
5. Verify memory limits are respected (palloc integration)
6. Test end-to-end pipeline: ingest → embed → store → retrieve → generate

### Building Language Bindings for Edge
- Python: Located in `sdk/python/` - uses ctypes by default (minimal footprint)
- For richer APIs on edge, consider minimal pybind11 or custom C-compatible wrappers
- Follow existing patterns in `sdk/` for new bindings
- Consider creating lightweight bindings for microcontrollers if needed

### Adding Edge-Specific Features
1. **Connectivity**: HTTP endpoints in `src/edge_connectivity/` (health, metrics, ingestion)
2. **Authentication**: Token-based auth mechanisms in edge connectivity layer
3. **Hardware Features**: Utilize platform-specific optimizations (NEON, AVX2, etc.)
4. **Wear Leveling**: Extend write-byte counters in storage layer for flash endurance

## Testing Guidelines
- Unit tests: GoogleTest framework in `tests/`
- Integration tests: Focus on cross-membrane operations and recovery scenarios
- Benchmark tests: Located in `benchmarks/` - measure throughput and latency
- Sanitizer builds: Enable ASan/UBSan/TSan via CMake for CI-like testing locally
- Always run `./scripts/check_no_malloc_new.sh` to ensure proper memory allocation practices

## Logging and Diagnostics
- Structured logging in `src/util/logging.*`
- Log levels: DEBUG, INFO, WARN, ERROR
- Use `POMAI_LOG(level)` macros for conditional logging
- Inspect builds: `src/core/inspect.cc` provides runtime introspection
- Health checks: Built-in HTTP endpoint (`/health`) when Edge connectivity features are enabled
