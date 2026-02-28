# Cross-Engine Vector Benchmark

Compares PomaiDB with hnswlib, FAISS, and optional engines (Qdrant, Milvus) on ingestion, query throughput, disk usage, and recall.

## Embedded / single-node tuning

For **embedded** or **single-node** use (smaller disk, higher QPS):

- **PomaiDB** is run with **1 shard**. With multiple shards and no IVF routing, every query is sent to every shard and results are merged, which multiplies work and disk. One shard gives one HNSW index and one set of segments, comparable to a single hnswlib/FAISS index.
- For **scale-out** (many shards + optional IVF routing), increase `opts.shards` in `engine_worker.py` and ensure routing is trained so queries are routed to a subset of shards.

## Improving throughput, disk, and memory

- **Query throughput**: Single shard (above), and ensuring the C API uses batch search (`pomai_search_batch`) as in this benchmark. Further gains: lower `efSearch` for speed/recall tradeoff, or routing so each query hits fewer shards.
- **Disk**: One shard reduces segment and index count. For even smaller footprint, consider FP16 or scalar quantization in segments (PomaiDB supports these); post-freeze WAL can be truncated in future if desired.
- **Peak RSS**: The benchmark reports process peak RSS for the whole run (all engines in one process). For per-engine memory, run engines in separate processes or use a memory profiler.

## Reference repos for refactoring

Useful when tuning HNSW, indexing, or storage:

- **hnswlib** (https://github.com/nmslib/hnswlib) – single-index HNSW, simple API, good baseline for query latency and recall.
- **FAISS** (https://github.com/facebookresearch/faiss) – flat and HNSW indices, batching, GPU options.
- **SimSIMD** (https://github.com/ashvardanian/SimSimd) – already vendored under `third_party/simd` for distance kernels; use for new backends or extra metrics.

PomaiDB’s HNSW layer lives in `third_party/pomaidb_hnsw/` and is called from `src/core/index/hnsw_index.cc`; distance and batching are in `src/core/distance.cc` and the vector engine in `src/core/vector_engine/`.
