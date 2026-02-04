# Operations

## What it is
- Operational guidance for embedding PomaiDB as a library in a single process. (Source: `pomai::DB` API in `include/pomai/pomai.h`.)

## What it is not
- Not a server deployment guide; there is no built-in server binary. (Assumption based on build targets.)

## Design goals
- Make durability and visibility explicit via `Flush` and `Freeze`. (Source: `pomai::core::ShardRuntime::HandleFlush`, `HandleFreeze`.)

## Non-goals
- Zero-ops observability; metrics are not yet exported. (Source: public API has no metric endpoints.)

## Operational notes
- **Open** a DB with `DB::Open` and set `DBOptions.path`, `dim`, `shard_count`. (Source: `pomai::DB::Open`.)
- **Durability**: set `DBOptions.fsync = kAlways` and call `Flush` to enforce WAL durability. (Source: `Wal::Flush`.)
- **Visibility**: call `Freeze` to publish snapshots and persist segments. (Source: `ShardRuntime::HandleFreeze`.)

## Recommended deployment modes
1. **Embedded library (primary)**
   - Link against `libpomai` and call the C++ API. (Source: `CMakeLists.txt`, `include/pomai/pomai.h`.)
2. **Embedded server**
   - Not provided in this branch. (Assumption based on build targets.)

## Resource tuning
- **Shard count**: Choose `DBOptions.shard_count` based on CPU cores and write concurrency; each shard has a dedicated writer thread. (Source: `Engine::OpenLocked`, `ShardRuntime`.)
- **Memory budget**:
  - Active MemTable uses an arena with 1 MiB blocks and a large hash map reserve. (Source: `MemTable::MemTable`.)
  - Frozen MemTables accumulate until `Freeze` is called. (Source: `ShardRuntime::RotateMemTable`, `HandleFreeze`.)
- **Freeze threshold**: Soft freeze triggers at 5000 entries per shard (not configurable). (Source: `ShardRuntime::HandlePut`.)
- **WAL size**: WAL rotates at 64 MiB segments. (Source: `Engine` constant `kWalSegmentBytes`.)

## Observability
- **Metrics**: None exported today. (Source: public API lacks counters.)
- **Logs**: None emitted by default. (Assumption based on absence of logging facilities.)

## Metrics (if you add instrumentation)
If you add custom instrumentation, recommended metrics include:
- `pomai.shard.queue_depth` (count) — mailbox depth per shard. (Source: `ShardRuntime::GetQueueDepth`.)
- `pomai.shard.ops_processed` (count) — operations processed per shard. (Source: `ShardRuntime::GetOpsProcessed`.)
- `pomai.snapshot.version` (count) — snapshot version per shard. (Source: `ShardSnapshot::version`.)
- `pomai.snapshot.age_ms` (ms) — age of snapshot (`now - created_at`). (Source: `ShardSnapshot::created_at`.)

## Troubleshooting

### Slow ingest
- **Possible cause**: WAL fsync on every append (`FsyncPolicy::kAlways`). (Source: `Wal::AppendPut`.)
- **Mitigation**: Use `kNever` during bulk ingest and call `Flush` periodically. (Assumption based on fsync behavior.)

### High snapshot lag
- **Possible cause**: Soft freeze threshold not reached and `Freeze` not called. (Source: `ShardRuntime::HandlePut`.)
- **Mitigation**: Call `Freeze` explicitly after a batch. (Source: `ShardRuntime::HandleFreeze`.)

### Large WAL growth
- **Possible cause**: No `Freeze`, so WAL is never reset. (Source: `ShardRuntime::HandleFreeze`, `Wal::Reset`.)
- **Mitigation**: Call `Freeze` to flush to segments and reset WAL. (Source: `ShardRuntime::HandleFreeze`.)

### Recovery time too slow
- **Possible cause**: WAL segments are large or numerous. (Source: `Wal::ReplayInto`, `Engine::OpenLocked`.)
- **Mitigation**: Call `Freeze` periodically to reset WAL. (Source: `Wal::Reset`.)

## Limits
- Visibility lag is bounded by 5000 entries per shard; there is no time-based bound. (Source: `ShardRuntime::HandlePut`.)
