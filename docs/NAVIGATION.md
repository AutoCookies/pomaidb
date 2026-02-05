# PomaiDB: How to navigate the repo

- `src/api/db.cc`: public DB facade and membrane-scoped API wiring.
- `src/core/engine/engine.{h,cc}`: database engine lifecycle, shard orchestration, cross-shard fanout.
- `src/core/shard/runtime.{h,cc}`: shard actor runtime, write handlers, freeze/compact, local search.
- `src/core/shard/layer_lookup.{h,cc}`: canonical newest-wins id lookup over active/frozen/segment layers.
- `src/core/shard/snapshot.h`: immutable shard snapshot payload used by readers.
- `src/storage/wal/*`: per-shard write-ahead log append/replay/reset behavior.
- `src/core/shard/manifest.*` + `src/storage/manifest/*`: segment manifest atomic commit/load.
- `src/table/memtable.*`: mutable in-memory table and tombstones.
- `src/table/segment.*`: immutable segment format, reader, and builder.
- `tests/integ/*`: end-to-end semantics tests (visibility, consistency, persistence).
- `tests/unit/*`: local component correctness tests.

Start from `docs/SOT.md` first, then trace from API -> engine -> shard runtime.
