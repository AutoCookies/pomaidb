# On-disk format

## What it is
- A per-membrane directory containing WAL files, shard directories, shard manifests, and segment files. (Source: `pomai::core::Engine::OpenLocked`, `Wal::SegmentPath`, `ShardManifest`.)

## What it is not
- Not a distributed storage layout or shared remote filesystem format. (Assumption based on local filesystem APIs.)

## Design goals
- Allow crash recovery by replaying WAL. (Source: `Wal::ReplayInto`.)
- Ensure manifest updates are atomic with rename + dir fsync. (Source: `ShardManifest::Commit`.)

## Non-goals
- Cross-version compatibility guarantees beyond the documented version headers. (Assumption based on explicit version checks.)

## Directory layout

Given a membrane path `DBOptions.path` (for default membrane) or `DBOptions.path/membranes/<name>` (for named membranes):

```
<membrane_path>/
  wal_<shard_id>_<gen>.log
  shards/
    <shard_id>/
      manifest.current
      manifest.new
      seg_<timestamp>_<ptr>.dat
```

- **WAL files** live directly under the membrane path. (Source: `Wal::SegmentPath`.)
- **Shard directories** live under `shards/<id>`. (Source: `Engine::OpenLocked`.)
- **Shard manifests** (`manifest.current`) list active segment filenames. (Source: `ShardManifest::Load`.)

> **Note:** Membrane manifests (`MANIFEST` files) exist in `storage::Manifest` but are not used by `DB::Open` yet. (Source: `storage/manifest/manifest.cc`, `core/membrane/manager.cc`.)

## WAL record format
- **File name**: `wal_<shard_id>_<gen>.log`. (Source: `Wal::SegmentPath`.)
- **Frame**:
  - `FrameHeader { uint32 len }` where `len` is bytes after header. (Source: `Wal::FrameHeader`.)
  - `RecordPrefix { uint64 seq, uint8 op, uint64 id, uint32 dim }`. (Source: `Wal::RecordPrefix`.)
  - Optional vector payload (`dim * 4` bytes for PUT). (Source: `Wal::AppendPut`.)
  - CRC32C of the body (prefix + payload). (Source: `Wal::AppendPut`, `AppendDelete`.)
- **Replay**: Stops on truncated tail; CRC mismatch is corruption. (Source: `Wal::ReplayInto`.)

## Segment file format
- **Header**: `SegmentHeader` with magic `pomai.seg.v1` and `version=2`. (Source: `SegmentHeader`, `SegmentReader::Open`.)
- **Entry layout** (per vector):
  - `uint64 id`
  - `uint8 flags` + 3 bytes padding
  - `float[dim]` vector (zeroed for tombstones)
  (Source: `SegmentBuilder::Finish`.)
- **CRC**: CRC32C of the file payload written at the end. (Source: `SegmentBuilder::Finish`.)
- **Ordering**: Entries are sorted by ID. (Source: `SegmentBuilder::Finish`.)

## Manifest format
### Shard manifest (`manifest.current`)
- **Location**: `<membrane_path>/shards/<id>/manifest.current`.
- **Content**: newline-separated list of segment filenames. (Source: `ShardManifest::Commit`.)
- **Atomic update strategy**:
  1. Write `manifest.new`.
  2. `fsync` file.
  3. Rename to `manifest.current`.
  4. `fsync` directory. (Source: `ShardManifest::Commit`.)

### Membrane manifest (`membranes/<name>/MANIFEST`)
- **Format**: `pomai.membrane.v2` with fields for name, shard count, dim, metric, index params. (Source: `storage::Manifest::WriteMembraneManifest`.)
- **Status**: Defined in code but not currently used by `MembraneManager`. (Source: `core/membrane/manager.cc`.)

## Backward compatibility policy
- **WAL**: Frames are parsed strictly; unknown op codes fail replay. (Source: `Wal::ReplayInto`.)
- **Segments**: `SegmentReader` only accepts `version == 2`. (Source: `SegmentReader::Open`.)
- **Manifest**: Root manifest requires `pomai.manifest.v3`. (Source: `storage::Manifest::LoadRoot`.)
- **Policy**: Backward compatibility is best-effort and must be explicitly implemented per format version. (Assumption based on explicit version checks.)

## Code pointers (source of truth)
- `pomai::storage::Wal::SegmentPath` — WAL file naming and location. (File: `src/storage/wal/wal.cc`.)
- `pomai::storage::Wal::ReplayInto` — WAL parsing and CRC verification. (File: `src/storage/wal/wal.cc`.)
- `pomai::table::SegmentBuilder::Finish` — segment layout and CRC. (File: `src/table/segment.cc`.)
- `pomai::table::SegmentReader::Open` — segment header validation and version checks. (File: `src/table/segment.cc`.)
- `pomai::core::ShardManifest::Commit` — shard manifest atomic update. (File: `src/core/shard/manifest.cc`.)
- `pomai::storage::Manifest::WriteMembraneManifest` — membrane manifest format. (File: `src/storage/manifest/manifest.cc`.)
