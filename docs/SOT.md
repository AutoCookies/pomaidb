# PomaiDB Canonical Semantics (SOT)

This document is the single source of truth for runtime semantics. If behavior diverges, fix code or update this file in the same change.

## 1) Core identity and invariants

- Sharded architecture, single writer per shard (mailbox/actor).
- Per-shard WAL is the durability log for mutations.
- Readers use immutable snapshots and never mutate snapshot state.
- Read/write visibility follows **newest-wins layer merge**.
- Tombstones in newer layers hide older values.

## 2) Lifecycle and error semantics

- Constructors do not perform fallible IO.
- Open/Start operations perform IO and return `Status` on failure.
- Close/Stop operations are explicit and idempotent.
- No silent error swallowing on persistence or manifest transitions.

## 3) Write path semantics

For a shard mutation (`Put`, `Delete`, batch put):
1. Append to shard WAL.
2. Apply to active memtable.
3. Mutation becomes visible to subsequent reads via layer merge.

`Freeze` persists frozen memtables into immutable segments and commits manifest atomically before WAL reset.

## 4) Canonical read path (Get / Exists / Search)

Canonical layer order (newest to oldest):
1. Active memtable.
2. Frozen memtables (reverse age order).
3. Segment readers (manifest order, newest first).

Visibility rules:
- First matching record in the order above defines result.
- Tombstone at any newer layer returns NotFound / false and blocks older data.
- `Get` and `Exists` share the same canonical lookup semantics.
- `Search` deduplicates by id with newest-wins behavior and excludes tombstoned records.

## 5) Multi-shard policy

- Shard-local writes are fail-fast.
- Multi-shard search may return partial results if configured by higher layers; shard-level behavior always returns explicit status.

## 6) Performance constraints

- Keep ingest and search hot paths allocation-aware.
- Do not add extra vector copies on single `Put` path.
- Batch ingestion can copy for ownership safety, but changes must be deliberate and benchmarkable.

## WAL-Sketched Block Routing (WSBR) — single search algorithm
PomaiDB uses exactly one ANN routing algorithm: WSBR. Segment search uses a block sketch sidecar (`.wsbr`) with deterministic 64-bit block signatures and exact scoring within selected blocks.

Determinism rules:
- Query signature generation is deterministic and seed-fixed.
- Block ranking order uses `(hamming_distance, block_id)`.
- Cross-shard/segment merges remain newest-wins with tombstone dominance and stable tie breakers.

Failure semantics:
- Missing/corrupt WSBR sidecar returns `Status::Corruption` (fail-closed).
- No fallback to IVF/HNSW/PQ or alternate index family.

Tuning knobs (`IndexParams`):
- `wsbr_block_size`
- `wsbr_top_blocks`
- `wsbr_widen_blocks`
