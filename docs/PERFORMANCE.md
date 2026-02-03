# Performance

## What it is
- An explanation of current complexity and bottlenecks based on the implemented algorithms. (Source: `ShardRuntime::SearchLocalInternal`, `MemTable`, `SegmentReader`.)

## What it is not
- Not a claim of distributed or approximate nearest neighbor performance. (Assumption based on brute-force search.)

## Design goals
- Keep reads lock-free by using snapshots. (Source: `ShardRuntime::Search`.)
- Keep writes serialized per shard to reduce contention. (Source: `ShardRuntime::RunLoop`.)

## Non-goals
- ANN acceleration is not enabled in the read path. (Source: `ShardRuntime::SearchLocalInternal` bypasses IVF.)

## Key invariants
- Search uses a single snapshot for candidate generation and scoring. (Source: `core/shard/invariants.h`.)

## Complexity analysis

### Ingest (Put/Delete)
- **WAL append**: O(1) amortized per record. (Source: `Wal::AppendPut`, `AppendDelete`.)
- **MemTable update**: O(1) average (hash map). (Source: `MemTable::Put`, `MemTable::Delete`.)
- **Soft freeze**: O(1) to rotate and publish snapshot; no disk I/O. (Source: `RotateMemTable`, `PublishSnapshot`.)

### Get/Exists
- **Frozen MemTables**: O(F) where F is the number of frozen tables (hash map lookup per table). (Source: `ShardRuntime::GetFromSnapshot`.)
- **Segments**: O(S * log N) due to binary search per segment. (Source: `SegmentReader::Find`.)

### Search
- **Current algorithm**: brute-force dot-product scan over all vectors in snapshot. (Source: `ShardRuntime::SearchLocalInternal`, `core/distance::Dot`.)
- **Complexity**: O(M * D) per shard where M is total vectors in frozen tables + segments and D is vector dimension. (Source: `SearchLocalInternal`.)

## Bottlenecks and mitigations
- **Brute-force scan** is the dominant cost for search. (Source: `SearchLocalInternal`.)
- **Mitigation today**: Use sharding to parallelize across CPU cores. (Source: `Engine::Search` uses thread pool.)
- **Planned**: Re-enable IVF/HNSW search path with thread-safe snapshots. (Assumption; see roadmap.)

## Benchmark methodology
- **Baseline benchmark**: `bench_baseline` scans vectors using the current search path. (Source: `tests/bench_baseline.cc`.)
- **Reproduce**:
  ```bash
  cmake -S . -B build
  cmake --build build -j
  ./build/bench_baseline
  ```

## Metrics
- There are no built-in performance counters; add external instrumentation if needed. (Source: public API.)

## Limits
- Search latency scales linearly with snapshot size due to brute-force scan. (Source: `SearchLocalInternal`.)
