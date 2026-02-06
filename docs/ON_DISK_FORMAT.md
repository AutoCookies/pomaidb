# On-disk format

## Scope
PomaiDB stores per-membrane state on local filesystem paths. This document defines the stable versioned fields for persistence compatibility and the contract for handling incompatible/corrupt files.

## Directory layout
Given membrane path `DBOptions.path` (default membrane) or `DBOptions.path/membranes/<name>`:

```
<membrane_path>/
  wal_<shard_id>_<gen>.log
  shards/
    <shard_id>/
      manifest.current
      manifest.new
      seg_<timestamp>_<ptr>.dat
```

## File formats and version fields

### WAL (`wal_<shard_id>_<gen>.log`)
- **File header (16 bytes, required for newly created files):**
  - `magic[12] = "pomai.wal.v1"`
  - `uint32 version = 1`
- **Frame stream (after header):**
  - `FrameHeader { uint32 len }`
  - `RecordPrefix { uint64 seq, uint8 op, uint64 id, uint32 dim }`
  - payload
  - `uint32 crc32c`

Validation rules:
- If magic matches `pomai.wal.v1`, version **must** equal `1`; otherwise replay returns `CORRUPTION`.
- CRC mismatch or malformed frame lengths return `CORRUPTION`.
- Truncated tail is tolerated and ignored (clean stop), never crash.

### Segment (`seg_*.dat`)
- `SegmentHeader.magic = "pomai.seg.v1"`
- `SegmentHeader.version = 3`
- Segment reader validates expected version and header invariants before reading entries.

### Shard manifest (`manifest.current`)
- First line header: `pomai.shard_manifest.v2`
- Subsequent lines: segment file names.
- Updated atomically by write-temp + fsync + rename + dir fsync.

### Root/membrane manifest
- Root manifest header: `pomai.manifest.v3`
- Contains `version` field and membrane specs.

## Compatibility policy

### Backward/forward guarantees
- **Patch releases:** no on-disk breaking changes allowed.
- **Minor releases:** may add optional fields only if old readers keep working.
- **Major format change:** requires version bump in file header and explicit migration path.

### Reader behavior for incompatible input
- Unknown/unsupported version for known magic => return `CORRUPTION` (or `INVALID_ARGUMENT` for API-level malformed inputs), never undefined behavior.
- Missing required headers in files that claim a versioned magic => return `CORRUPTION`.
- Corrupt checksums/lengths => return `CORRUPTION`.

### Migration rules
- Any format bump must:
  1. increment the relevant on-disk version field,
  2. preserve read path for previous supported version(s) or document one-way migration,
  3. add regression tests for version mismatch and corruption handling,
  4. update this document in same PR.

## What breaks when
- Changing WAL/segment/manifest magic or version without dual-read support is a breaking on-disk change.
- Reinterpreting existing field semantics without version bump is prohibited.

## Source-of-truth references
- WAL header + replay validation: `src/storage/wal/wal.cc`
- Segment header validation: `src/table/segment.cc`
- Shard manifest format + atomic commit: `src/core/shard/manifest.cc`
- Root manifest format: `src/storage/manifest/manifest.cc`

## WSBR sketch sidecar (`.wsbr`)
Each segment `<seg>.dat` has sidecar `<seg>.dat.wsbr`.

Binary layout (v1):
- Header
  - magic[16] = `pomai.wsbr.v1`
  - version (u32) = 1
  - block_size (u32)
  - block_count (u32)
  - reserved (u32)
- Block entries (`block_count`)
  - signature (u64)
  - start_index (u32)
  - count (u32)
  - reserved (u32)
- crc32c (u32) over header + block entries

Validation:
- magic/version/size/CRC must match.
- Reader is fail-closed on mismatch (`Status::Corruption`).
