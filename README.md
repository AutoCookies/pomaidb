# Pomai Cache — Production-Grade AI-Native In-Memory Data Platform

Pomai Cache is a hybrid in-memory data platform engineered for modern AI and real-time systems. It unifies key-value caching, vector search, time-series, graph relationships, and matrix operations in a single binary with adaptive runtime tuning, predictive eviction, and production-grade persistence and clustering features.

This document is a complete operational and technical reference intended for engineers, SREs, and platform teams responsible for deploying, operating, benchmarking, or contributing to Pomai Cache.

## Contents
- [Executive Summary](#executive-summary)
- [Design Principles](#design-principles)
- [Architecture Overview](#architecture-overview)
- [Core Components and Data Models](#core-components-and-data-models)
- [Algorithms and Internals](#algorithms-and-internals)
  - [PPE (Pomegranate Predictive Eviction)](#ppe-pomegranate-predictive-eviction)
  - [PIE (Pomai Intelligent Eviction — RL)](#pie-pomai-intelligent-eviction--rl)
  - [PQSE (Probabilistic Quantum Sampled Eviction)](#pqse-probabilistic-quantum-sampled-eviction)
  - [PLG and PLBR (Membrane Graph and Burst Replication)](#plg-and-plbr-membrane-graph-and-burst-replication)
  - [Vector Engine Tuning and Adaptive ef_search](#vector-engine-tuning-and-adaptive-ef_search)
  - [PGUS / VirtualStore](#pgus--virtualstore)
  - [PIC Compression](#pic-compression)
- [Configuration (ENV & CLI) and Priority Rules](#configuration-env--cli-and-priority-rules)
- [Startup Examples and Recommended Production Flags](#startup-examples-and-recommended-production-flags)
- [Persistence Modes and Durability Tradeoffs](#persistence-modes-and-durability-tradeoffs)
- [Benchmarking: Scenarios, Measurement, and Interpretation](#benchmarking-scenarios-measurement-and-interpretation)
- [Observability: Metrics, Logs, and Profiling](#observability-metrics-logs-and-profiling)
- [Operational Guidance and Checklist](#operational-guidance-and-checklist)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Development, Testing, and Contribution Notes](#development-testing-and-contribution-notes)
- [License and Contacts](#license-and-contacts)
- [Appendix: Quick Reference Commands](#appendix-quick-reference-commands)

## Executive Summary

Pomai Cache treats cached items as contextual objects rather than isolated key-value pairs. It preserves clusters of related keys ("membranes") and uses predictive algorithms and adaptive runtime controls so the system remains responsive under diverse workloads: retrieval-augmented generation (RAG), conversational session caches, vector embedding caches, real-time telemetry, and combined workloads.

Key benefits:
- Multi-model platform: KV, vectors, graphs, time-series, matrices in one binary.
- Predictive adaptive eviction to keep context rather than isolated hot keys.
- Emergency, bounded-latency reclaim path to avoid unbounded stalls under memory pressure.
- Adaptive runtime for container and bare-metal environments (auto-detect cgroups).
- Production-grade persistence options: snapshot and WAL with configurable write-behind.

## Design Principles

- Predictability: Keep tail latency and recoverability bounded; provide emergency reclaim.
- Low GC pressure: Avoid large in-heap objects; use chunking, compression, and mmap where appropriate.
- Workload-aware adaptation: Run-time parameters adjust automatically based on measured metrics.
- Observability and operability: Expose metrics, pprof, and admin endpoints for troubleshooting.
- Single-binary portability: Run on Linux, macOS, and Windows; behave sensibly in containers without manual tuning.

## Architecture Overview

Components:
- Control plane: HTTP admin server (health, metrics, admin endpoints).
- Data plane: High-performance TCP binary protocol (gnet).
- Tenancy: A tenants manager creates per-tenant Store instances.
- Store: Sharded stores with pluggable shard implementations and eviction manager.
- Subsystems: Vector index, matrix engine, time-series, PLG (membrane), PGUS/VirtualStore for large values, PIC (prompt chains).
- Persistence adapters: File-based snapshot and WAL (write-ahead log).
- Cluster & replication: Gossip-based discovery and replication manager.

Data flow (high level):
- Client connects via binary TCP protocol and issues commands.
- Store routes requests to a shard; read/write operations are mostly lock-reduced.
- Eviction manager (PPE/PIE) observes accesses and decides reclaim when needed.
- Persistence layer (if enabled) writes snapshots or WAL entries; optional write-behind batches writes.
- Replication manager optionally replicates hot keys or WAL entries for durability.

## Core Components and Data Models

Store and shards:
- Shards are power-of-two sized; implementations may vary (lock-based, lock-free stripes).
- Each shard maintains map entries, optional LRU list, and counters. Eviction interacts with shard APIs to delete items safely.
- Entry metadata is compact: size, expireAt, optional hints for PLG.

Vector engine:
- Default index "default_embeddings".
- Indexing: HNSW-like structure using contiguous arena allocations to reduce pointer-chasing and GC.
- Search parameters: ef_search, ef_construction, index size are tunable. The AutoTuner (PIE/PIE Bandit) adjusts ef_search at runtime.

PLG (membrane):
- Lightweight adjacency stored as short hint lists per key to represent likely co-access neighbors. Not a full graph DB; intended to preserve context clusters.

PGUS and VirtualStore:
- PGUS: Chunk/granule deduplicated storage (granules compressed with Snappy), reference counted.
- VirtualStore: Mmap-backed allocation to keep very large items out of Go heap with demotion/promotion capabilities.

TimeStream:
- Append-only time series blocks for windowed aggregation and analytics; optimized for append-heavy workloads.

PIC:
- Prompt Inference Chain storage optimized for sequential LLM conversation storage using delta compression.

## Algorithms and Internals

### PPE (Pomegranate Predictive Eviction)
- Per-item predictor is an EMA of inter-arrival intervals giving a predicted next-access timestamp.
- Cluster centrality (membrane) is a multiplicative preservation factor; more central keys get a lower eviction risk.
- Candidate scoring (conceptual):
  Score = alpha * predicted_time + beta * size_penalty + gamma * (1 / cluster_strength) + delta * frequency_penalty
- Lower score => candidate for eviction.
- PPE scanning: Samples from LRU tail across shards, computes scores, sorts candidates, and evicts until target reclaimed bytes.

### PIE (Pomai Intelligent Eviction — RL)
- A multi-armed-bandit (UCB1) agent selects parameter arms (arm = {ef_search, evict_sample}) periodically.
- Reward = hitRate - latencyPenalty where latencyPenalty scales with measured average latency.
- PIE updates bandit estimates every interval (default 30s) and applies the configuration of chosen arm to the store.
- PIE reduces manual tuning and adapts to changing workload characteristics.

### PQSE (Probabilistic Quantum Sampled Eviction)
- Emergency algorithm for critical memory reclaim.
- Random/probabilistic selection using a deterministic zero-allocation hash function and frequency estimate.
- Guarantees bounded latency; works when PPE is insufficient or too slow.

### PLG and PLBR (Membrane Graph and Burst Replication)
- PLG: Lightweight membrane graph to model key relationships.
- PLBR: When a key shows trending behavior (high access + predicted next-access), PLBR probabilistically replicates it to other nodes to reduce hotspot pressure.
- Replication is fire-and-forget; used to spread load early rather than wait for replication after saturation.

### Vector Engine Tuning and Adaptive ef_search
- Vector search latency is tracked via EMA.
- If EMA exceeds target threshold, AutoTuner will reduce ef_search to restore latency at the cost of some recall.
- PIE's bandit may also choose arms with different ef_search settings.

### PGUS / VirtualStore
- PGUS decomposes large objects into fixed-size granules with deduplication and reference counting.
- VirtualStore uses mmap for overflow handling, with promotion/demotion based on access patterns.

### PIC Compression
- Delta compression for sequential LLM chains, with predictive next-prompt hints.

SIMD / Parallel numeric paths:
- For heavy numeric operations (matrix multiply, vector dot), parallel loop partitioning is used with worker counts tuned to CPU and vector size.
- Optimized loops reduce overhead on modern CPUs; algorithms are adaptive (use single-thread for small sizes).

Predictive pruning (PPPC):
- A background cleaner samples keys biased by predicted next-access times and expands small clusters to remove likely-cold clusters proactively.
- Sampling and expansion respect time budgets to avoid interfering with foreground ops.

## Configuration (ENV & CLI) and Priority Rules

Priority rules:
1. CLI flags override environment variables.
2. Environment variables are used when the corresponding CLI flag is not provided.
3. If no explicit memory limit is set (via CLI or `GOMEMLIMIT`/`MEM_LIMIT`), ApplySystemAdaptive auto-detects container or host memory limits and adjusts runtime settings.

Important environment variables (canonical list):
- `PORT` (default "8080") — HTTP admin/metrics port
- `TCP_PORT` (default "7600") — TCP binary protocol port
- `CLUSTER_MODE` (default "false") — enable gossip/cluster
- `NODE_ID` (default "node-<timestamp>") — node identifier
- `GOSSIP_PORT` (default "7946") — gossip UDP port
- `CLUSTER_SEEDS` (default "") — comma-separated seed nodes
- `TCP_MAX_CONNECTIONS` (default 10000) — max concurrent TCP connections
- `CACHE_SHARDS` (default 2048) — number of keyspace shards (power-of-two)
- `PER_TENANT_CAPACITY_BYTES` (default 0) — per-tenant memory quota
- `PERSISTENCE_TYPE` (default "none") — "none", "file", or "wal"
- `DATA_DIR` (default "./data") — persistence directory
- `WRITE_BUFFER_SIZE` (default 1000) — write-behind buffer size
- `FLUSH_INTERVAL` (default 5s) — write-behind flush interval
- `HTTP_READ_TIMEOUT` / `HTTP_WRITE_TIMEOUT` / `HTTP_IDLE_TIMEOUT`
- `SHUTDOWN_TIMEOUT` (default 30s)
- `ENABLE_CORS` (default false)
- `ENABLE_METRICS` (default true)
- `ENABLE_DEBUG` (default false) — exposes pprof and debug routes
- `GOGC` (default -1) — GC percent override
- `GOMAXPROCS` (default 0) — if >0, override runtime.GOMAXPROCS
- `GOMEMLIMIT` / `MEM_LIMIT` (default 0) — set Go runtime soft memory limit (supports "4GB" etc.)

CLI flags:
- All of the above are exposed as CLI flags with hyphen names (for example, --http-port, --tcp-port, --persistence, --data-dir, --mem-limit, --cache-shards, --write-buffer, --flush-interval, --gomaxprocs, --gogc).
- Use `./pomai-server --help` to see all current flags and their descriptions.

Examples (flag precedence):
- Environment:
  ```
  export PORT=8081
  ./pomai-server
  ```
- Flags override:
  ```
  ./pomai-server --http-port=9090
  ```

## Startup Examples and Recommended Production Flags

Local development (fast start):
```
./pomai-server --persistence=none --cache-shards=512
```

Single-node production (WAL + bounded memory):
```
./pomai-server --persistence=wal --data-dir=./data --mem-limit=16GB --write-buffer=2000 --flush-interval=200ms --cache-shards=2048
```

Cluster example (seed + peer):
Node A (seed):
```
./pomai-server --cluster --node-id=nodeA --tcp-port=7601 --http-port=8081 --gossip-port=7946 --persistence=wal --data-dir=/data/nodeA
```
Node B (peer):
```
./pomai-server --cluster --node-id=nodeB --tcp-port=7602 --http-port=8082 --gossip-port=7947 --seeds=127.0.0.1:7946 --persistence=wal --data-dir=/data/nodeB
```

Docker example:
```
docker run -d --name pomai \
  -p 8080:8080 -p 7600:7600 \
  --memory=8g \
  -v $(pwd)/data:/data \
  pomai-cache:latest \
  ./pomai-server --persistence=wal --data-dir=/data --mem-limit=8GB
```

Kubernetes hints:
- Always set `resources.limits.memory` and `resources.requests.memory`.
- Mount a persistent volume for DATA_DIR.
- Use a PodDisruptionBudget and readiness checks for graceful upgrades.

## Persistence Modes and Durability Tradeoffs

Modes:
- none: no persistence. Fastest, but no recovery after process crash.
- file (snapshot): periodic or on-demand full snapshots. Good when dataset is moderate and frequent snapshots are acceptable.
- wal: write-ahead log provides crash recovery. WAL is replayed on startup to rebuild state.

Write-behind buffer:
- When persistence != none, a write-behind buffer batches writes to disk; tune WRITE_BUFFER_SIZE and FLUSH_INTERVAL:
  - Larger buffer => higher throughput, larger window of potential loss on crash.
  - Smaller buffer => lower latency/durability loss, higher IO.

Crash recovery guidance:
- Test restore times regularly; WAL replay may be slower than snapshot restore for large states.
- Consider periodic snapshot + WAL rotation: snapshot reduces WAL replay window.

## Benchmarking: Scenarios, Measurement, and Interpretation

Tool: `pomai-bench` (located at `cmd/pomai-bench`).
Build:
```
go build -o pomai-bench ./cmd/pomai-bench
```

Representative scenarios:
- KV read-heavy: measure get p50/p95/p99 and throughput.
- KV write-heavy: measure sustained write throughput and WAL impact.
- Vector search: measure query throughput and p99 latency with different ef_search.
- Matrix ops: measure latency of matrix set/get and matrix arithmetic.
- Mixed RAG-like: mix KV/vectors/streams with skewed access pattern simulating LLM pipelines.

What to collect:
- Application: Prometheus metrics for requests, errors, eviction counters, vector EMA.
- Profiles: CPU and heap via pprof (enable --debug).
- System: top/htop, iostat, vmstat, netstat.
- Disk: WAL write latency and fsync stats.

Interpreting results:
- High p99 latency + low CPU: check lock contention and blocking IO.
- High CPU + low throughput: check hotspots in loops or serialization overhead.
- Frequent PQSE triggers: system under memory pressure; consider increasing memory or reducing per-tenant capacity.

Recommended presets:
- Latency-oriented small machine (2 vCPU / 8GB):
  ```
  ./pomai-server --mem-limit=7GB --gomaxprocs=2 --persistence=none --cache-shards=512
  ```
- Throughput-oriented large machine (32 vCPU / 256GB):
  ```
  ./pomai-server --mem-limit=240GB --gomaxprocs=32 --persistence=wal --write-buffer=5000 --flush-interval=200ms --cache-shards=4096
  ```

## Observability: Metrics, Logs, and Profiling

Metrics:
- Expose Prometheus metrics by default on the HTTP control port (if ENABLE_METRICS=true). Key metrics:
  - requests_total, requests_errors_total
  - requests_latency_seconds (histogram: p50/p90/p99)
  - eviction_bytes_total, eviction_count_total
  - vector_search_latency_ms (EMA)
  - shard_bytes, shard_items_per_shard
  - wal_write_latency_ms, wal_pending_entries

Logs:
- Structured logs include subsystem tags: [NETWORK], [STORAGE], [EVICTION], [PERSISTENCE], [REPLICATION]
- Critical events logged: WAL failures, replication delays, emergency PQSE activity, final snapshot creation on shutdown.

Profiling:
- Enable debug (`--debug` or ENABLE_DEBUG) to expose /debug/pprof endpoints.
- Use `go tool pprof` to capture CPU and heap profiles:
  - `go tool pprof http://localhost:8080/debug/pprof/profile?seconds=30`
  - `go tool pprof http://localhost:8080/debug/pprof/heap`

## Operational Guidance and Checklist

Before production rollout:
- Validate WAL directory on reliable storage (SSD, NVMe).
- Set container memory limits (if in k8s/Docker).
- Tune cache_shards to expected concurrency.
- Configure per-tenant capacity to avoid noisy neighbor conditions.
- Increase ulimit (nofile) to accommodate high concurrency.
- Enable metrics and pprof for observability.

During operation:
- Monitor eviction trends: rising evictions can indicate under-provision.
- Watch vector_search EMA: rising values indicate ef_search or resource issues.
- Monitor WAL write latency and disk utilization.
- Measure PQSE invocation frequency; frequent PQSE indicates memory pressure.

Upgrade and maintenance:
- Use rolling upgrades with readiness/liveness probes.
- Test backups and restores regularly.
- Keep WAL rotation and retention policies in place.

## Troubleshooting

Out-of-memory (OOMKilled):
- Check container/pod limits.
- Increase MEM_LIMIT or reduce per-tenant capacity.
- Tune eviction thresholds or add nodes to cluster.

High tail latency (p99):
- Capture CPU/profile, examine mutex/memory stalls.
- Inspect vector_search EMA and adjust ef_search or AutoTuner settings.
- Check disk IO and WAL fsync latency.

WAL corruption or failed restore:
- Ensure WAL files are stored on reliable disk.
- If WAL corrupt, restore from latest snapshot and replay subsequent WALs when possible.

## Security Considerations

- Network: Binary protocol is plain TCP; place behind network controls or use mTLS/TLS tunneling at network layer.
- Multi-tenant: Ensure per-tenant capacity limits to avoid data leakage through resource exhaustion.
- Persistence protection: Restrict access to DATA_DIR and WAL paths.
- Administrative endpoints: Restrict HTTP admin endpoints via firewall, network policies, or identity-aware proxies.

## Development, Testing, and Contribution Notes

Code layout:
- `internal/engine` contains the runtime engines (core, ttl, eviction, replication).
- `internal/adapter` contains protocol adapters (tcp, http, persistence).
- `shared` contains data structures and algorithms (vector, ds, al).
- `cmd` contains server and bench tooling.

Testing:
- Unit tests for eviction scoring, bucket sampling, PGUS deduplication.
- Integration tests for WAL + snapshot restore, vector index correctness, and concurrency stress tests.
- Performance regression tests using `pomai-bench`.

Contribution workflow:
- Fork, create topic branch, add tests, open PR with description and benchmarks.
- Maintain backward compatibility for persistence formats (document versioning).

## License and Contacts

Check LICENSE at repository root for license details. For questions, operational support, or contribution guidance, open an issue and tag maintainers.

## Appendix: Quick Reference Commands

Start server:
```
./pomai-server --persistence=wal --data-dir=./data --mem-limit=8GB
```

Run bench:
```
./pomai-bench -addr=127.0.0.1:7600 -mode=kv -clients=100 -requests=200000
```

Capture CPU profile:
```
go tool pprof http://localhost:8080/debug/pprof/profile?seconds=30
```

Scrape metrics:
```
curl http://localhost:8080/metrics
```

Pomai Cache — Built for the AI Era. © 2026 PomaiDB Team.