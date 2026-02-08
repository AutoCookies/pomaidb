# PomaiDB

<p align="center">
  <img src="./assets/logo.png" width="200" alt="Demo">
</p>

## Executive summary
PomaiDB is a single-process, embedded vector database implemented in C++20. 
It targets applications that need local vector search without external services. 
Data is partitioned into shards; each shard has a single writer thread. 
Readers are lock-free and use immutable snapshots for concurrency safety. 
Writes go through a write-ahead log (WAL) before mutating memory. 
Reads observe a published snapshot, not the active mutable MemTable. 
Snapshots are monotonically versioned and represent a prefix of WAL history. 
Visibility is bounded by MemTable rotation (soft freeze) or explicit Freeze. 
By default, soft freeze triggers after 5000 items per shard. 
The database is crash-recoverable by replaying WAL into MemTables. 
On startup, replayed data is rotated to frozen tables for immediate visibility. 
Segments are immutable on-disk files written during Freeze. 
Segment lists are persisted via shard manifests with atomic rename + dir fsync. 
Durability depends on the configured fsync policy for WAL and segment files. 
The default fsync policy is `kNever`, so durability is best-effort. 
Search currently uses brute-force scanning over snapshot items for correctness. 
IVF is present in code but bypassed in the read path today. 
Metrics are not yet exposed; operators must instrument externally. 
PomaiDB is not distributed and does not implement replication. 
It is intended for embedded, single-tenant deployments only. 
Membranes provide logical separation with persisted manifests. 
Membranes are now typed: VECTOR for embeddings and RAG for chunk/token storage. 
The API exposes explicit Freeze and Flush for visibility and durability control. 
Documentation is written to match the current code on branch `pomai-embeded`. 

## What it is
- An embedded, single-process vector database with sharded, actor-style writers.
- A lock-free snapshot reader model for Search/Get/Exists.
- A WAL + immutable segment storage engine for crash recovery.

## What it is not
- Not distributed or replicated.
- Not multi-tenant or cloud-managed.
- Not a general-purpose SQL/OLAP database.

## Quick links (authoritative docs)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Consistency model: [docs/CONSISTENCY_MODEL.md](docs/CONSISTENCY_MODEL.md)
- Failure semantics: [docs/FAILURE_SEMANTICS.md](docs/FAILURE_SEMANTICS.md)
- On-disk format: [docs/ON_DISK_FORMAT.md](docs/ON_DISK_FORMAT.md)
- Operations: [docs/OPERATIONS.md](docs/OPERATIONS.md)
- Performance: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- Performance tuning: [docs/PERFORMANCE_TUNING.md](docs/PERFORMANCE_TUNING.md)
- Benchmarking: [docs/BENCHMARKING.md](docs/BENCHMARKING.md)
- RAG membrane SOT: [docs/SOT_RAG_MEMBRANE.md](docs/SOT_RAG_MEMBRANE.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Glossary: [docs/GLOSSARY.md](docs/GLOSSARY.md)
- FAQ: [docs/FAQ.md](docs/FAQ.md)

## Hello World (embedded C++)
```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "pomai/pomai.h"

int main() {
    pomai::DBOptions opt;
    opt.path = "./demo_db";
    opt.dim = 4;
    opt.shard_count = 2;
    opt.fsync = pomai::FsyncPolicy::kAlways; // optional: durability boundary for WAL

    std::unique_ptr<pomai::DB> db;
    pomai::Status st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Open failed: " << st.message() << "\n";
        return 1;
    }

    float v1[] = {1, 0, 0, 0};
    float v2[] = {0, 1, 0, 0};
    db->Put(42, v1);
    db->Put(7, v2);

    // Publish a snapshot so reads can see recent writes.
    db->Freeze("__default__");

    pomai::SearchResult out;
    db->Search(std::span<const float>(v1, 4), 2, &out);
    for (const auto &hit : out.hits) {
        std::cout << "id=" << hit.id << " score=" << hit.score << "\n";
    }

    db->Close();
    return 0;
}
```

## API overview
- **Create/Open**: `pomai::DB::Open(DBOptions, ...)`
- **Upsert/Delete**: `PutVector`, `PutChunk`, `Delete` (per-membrane and default membrane overloads)
- **Read**: `Get`, `Exists`, `SearchVector`, `SearchRag`
- **Freeze**: publish visibility + flush frozen tables to segments
- **Flush**: WAL durability boundary (subject to fsync policy)
- **Close**: `Close` closes all membranes and shards

## Guarantee matrix
| Operation | Durability | Visibility | Isolation | Ordering | Notes |
| --- | --- | --- | --- | --- | --- |
| Upsert (Put/PutBatch/Delete) | WAL appended; durable if WAL `fsync=kAlways` or after `Flush` | Not visible until soft freeze (5000 items/shard) or explicit `Freeze` | Snapshot isolation per shard | Per-shard order via mailbox | Active MemTable is mutable and not yet published to readers. |
| Search | Reads a single immutable snapshot | Snapshot state only (published frozen + segments) | Snapshot isolation | Per-shard snapshot ordering only | Brute-force scan over published snapshot data. |
| Get/Exists | Reads a single immutable snapshot | Snapshot state only (published frozen + segments) | Snapshot isolation | Per-shard snapshot ordering only | Read-your-writes requires waiting for/pushing Freeze. |
| Freeze | Segment files + shard manifest updated; WAL reset | Publishes a new snapshot after segment update | Readers move to new snapshot atomically | Per-shard only | Segment rename + manifest fsync; no global barrier. |
| Flush | WAL `fdatasync` if `fsync != kNever` | No visibility change | N/A | N/A | Flush is a durability boundary only. |
| Recovery | WAL replay into MemTable + rotate to frozen | Replayed data visible after startup rotation | Snapshot isolation post-open | Per-shard only | Truncated WAL tail tolerated. |

## Build, test, benchmark
```bash
# Build library + baseline benchmark
cmake -S . -B build
cmake --build build -j

# Build and run tests
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# Run baseline benchmark
./build/bench_baseline
```

## Benchmarking

PomaiDB includes a comprehensive benchmark suite measuring industry-standard metrics:

```bash
# Build and run benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target comprehensive_bench
./build/comprehensive_bench --dataset small  # or medium, large
```

**Metrics measured**:
- **Search Latency** (P50/P90/P99/P999 in microseconds)
- **Throughput** (QPS - queries per second)
- **Recall@k** (accuracy vs brute-force ground truth)
- **Build Time** (indexing performance)
- **Memory Usage**

**Sample results** (10K vectors @ 128 dims):
```
SEARCH LATENCY:
  P50:   973 µs
  P99:   3.1 ms

THROUGHPUT:
  QPS:   866 queries/sec

ACCURACY:
  Recall@10:  100%
```

See [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for full guide and comparison with other systems.


## Docker (Ubuntu/Linux)

PomaiDB is an embedded library, so the container image is designed for **build + test** workflows (not running a standalone DB server):

```bash
# Build image
docker build -t pomaidb/dev:local .

# Run tests in container
docker run --rm pomaidb/dev:local

# Or via docker compose
docker compose up --build pomaidb-dev
```

Compose files provided:
- `docker-compose.yml` (primary)
- `docker-compose.yaml` (compatibility copy)

## CI platform policy

GitHub Actions CI is Linux-only (`ubuntu-latest`) for this branch.
Windows/macOS build jobs were removed intentionally to match the current support target and keep CI signal focused.

## Known limitations (current code)
1. **Bounded staleness via soft freeze** (5000 items per shard) or explicit `Freeze`.
2. **Search uses brute-force scan**; IVF is bypassed for correctness.
3. **Metric selection is not wired into Search** (dot product is used for scoring).
4. **No built-in metrics or logging**; operators must instrument externally.
5. **Membrane manifests exist on disk but are not wired into DB::Open**.
6. **Directory fsync behavior depends on OS/filesystem** for segment renames.

## License
Apache-2.0 (see [LICENSE](LICENSE)).
