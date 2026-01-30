# Pomai Storage Contract Spec

## File Layout

```
<db_dir>/
  MANIFEST
  MANIFEST.tmp
  checkpoints/
    chk_<epoch>_<lsn>.pomai
  indexes/
    idx_<epoch>_<lsn>_<kind>.bin
  meta/
    dict_<epoch>_<lsn>.bin
  wal/
    shard_<i>.wal
```

## Atomic Commit Steps

1. Write snapshot temp file (`checkpoints/chk_...pomai.tmp`).
2. `fsync(snapshot.tmp)`.
3. `rename(snapshot.tmp, snapshot.final)`.
4. `fsync(checkpoints/)`.
5. Write dictionary and index temp files, `fsync` each, `rename`, `fsync` their directories.
6. Write `MANIFEST.tmp` with checksum.
7. `fsync(MANIFEST.tmp)`.
8. `rename(MANIFEST.tmp, MANIFEST)`.
9. `fsync(db_dir)`.

## Snapshot Format

* Magic bytes: `POMAI_SNAP`
* Version: `uint32`
* Endianness marker: `uint32`
* Flags: `uint32`
* Header checksum: CRC64
* Sections:
  * Section header: `section_type (u32), section_size (u64), section_crc64 (u64)`
  * Section data (streamed)
* Footer: `total_size (u64), total_crc64 (u64), footer_magic`

Sections:
1. **Schema**: dimension, metric, shard count, index kind.
2. **Shard**: shard id, segment count, segment states, live state.
3. **Dictionary**: namespace/tag id lists.

Each seed state encodes ids, quantized vectors, quantization metadata, and tag mappings.

## Recovery Rules

1. Read MANIFEST and verify checksum.
2. If MANIFEST is corrupt/missing, scan `checkpoints/` for newest valid snapshot.
3. Verify snapshot header/sections checksums and footer total CRC.
4. Load shard snapshots into memory.
5. Replay WAL entries with `LSN > checkpoint_lsn` only.
6. Publish Membrane view atomically after load completes.

## Commit Plan (Commit-by-Commit)

1. **Storage primitives:** add snapshot format, dictionary/index checksums, atomic commit function, verify utilities.
2. **Engine integration:** wire recovery + checkpoint orchestration into Membrane/Shard, use WAL replay boundaries.
3. **Tooling & tests:** add `pomai_verify` and crash/consistency tests.
