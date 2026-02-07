# CBR-S in PomaiDB

This document describes PomaiDB's CBR-S implementation: **Global-Centroid Shard Assignment + Routed Query + Dual-Epoch Probe**.

## Routing table format

Engine-level table contains:
- `epoch` (`u64`)
- `k` (`u32`)
- `dim` (`u32`)
- `centroids[k][dim]` (`float32`)
- `owner_shard[k]` (`u32`)
- `counts[k]` (`u64`)

Persistence files at DB root:
- `ROUTING`
- optional `ROUTING.prev`

File contents are binary payload + trailing CRC32C. Load validates magic, shape, and CRC.

## Crash safety

Routing persistence uses atomic write pattern:
1. write `ROUTING.tmp`
2. fsync file data
3. rename to `ROUTING`
4. fsync parent directory

Corrupt/truncated files are treated as missing (warmup fallback), never fatal at startup.

## Warmup to ready lifecycle

Modes:
- `DISABLED`: feature off
- `WARMUP`: collect bounded reservoir `M = k * warmup_mult`
- `READY`: table available for query routing and ingest routing

Initialization uses deterministic kmeans-lite:
- KMeans++ seeding
- 5 Lloyd iterations
- deterministic seed
- owner mapping `owner_shard[g] = g % shard_count`

After ready, centroids receive online incremental updates using count-based EMA.

## Routed query and dual-epoch probe

When READY:
1. compute nearest centroids to query
2. choose probe count (`routing_probe`, default 2), with margin heuristic expanding to 3 when top distances are close
3. map centroid IDs to shard IDs and deduplicate
4. if previous table exists in memory, compute probe set there too
5. union current + previous shard sets and query only those shards

This dual-epoch probe mitigates misses during routing transitions.

## Point lookup strategy

Because routed ingest no longer follows `id % shard`, engine point operations (`Get/Exists/Delete`) fan out across shards for correctness.

Rationale:
- safest minimal change
- no per-vector in-memory postings or directory added
- preserves WAL/shard actor model and correctness under new placement
