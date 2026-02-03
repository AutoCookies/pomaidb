# FAQ

## Is this a database or a search engine?
- It is an embedded vector database with a storage engine (WAL + segments) and a vector search API. (Source: `pomai::DB` API, `Wal`, `SegmentBuilder`.)

## Why not distributed?
- The code is single-process and uses local filesystem APIs only; there is no network protocol or replication logic. (Assumption based on `src/` contents.)

## Why no read-your-writes (RYW)?
- Reads use `ShardSnapshot`, which excludes the active MemTable. Writes are only visible after a soft/hard freeze. (Source: `ShardRuntime::PublishSnapshot`, `RotateMemTable`.)

## Why snapshots?
- Snapshots allow lock-free reads and single-writer concurrency without blocking. (Source: `ShardRuntime::GetSnapshot`, `ShardRuntime::RunLoop`.)

## What’s the bounded staleness guarantee?
- By default, at most 5000 writes per shard can be invisible (active MemTable threshold). (Source: `ShardRuntime::HandlePut`.)

## How do I choose shard count?
- Each shard has a dedicated writer thread; choose a shard count near CPU cores and expected write parallelism. (Source: `Engine::OpenLocked` and `ShardRuntime` threading model.)

## Is it safe on ext4/xfs?
- Segment and manifest updates rely on `rename` + directory fsync. Durability depends on filesystem semantics. (Source: `ShardManifest::Commit`, `util::FsyncDir`.)

## How do I upgrade on-disk formats?
- Check version headers in WAL/segment/manifest formats and follow compatibility guidance in `docs/ON_DISK_FORMAT.md`. (Source: `SegmentReader::Open`, `Manifest::LoadRoot`.)
