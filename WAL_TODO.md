# WAL_TODO — Thiết kế & Công việc để xây WAL "chuẩn chỉnh"

Mục đích: triển khai một Write‑Ahead Log (WAL) production‑grade cho Pomai — không phải giải pháp tạm bợ. WAL phải an toàn (durable), idempotent, có thể replay, testable, và có cấu trúc mở rộng để lưu nhiều loại event (ids updates, demote/promote, PPE publish/unpublish, v.v).

Hướng dẫn: đánh dấu checkbox khi hoàn thành; mỗi mục có mô tả, tiêu chí chấp nhận, test cần có.

---

## Tổng quan (Goals)
- [ ] Xác định yêu cầu chức năng & phi chức năng cho WAL
  - Mô tả: durability, atomicity ordering (WAL fsync trước khi áp dụng), replay correctness, truncated-on-success.
  - Chấp nhận: bộ test replay end‑to‑end cho SoaIdsManager pass.

## Thiết kế định dạng file WAL
- [x] Định nghĩa header file WAL
  - [x] `magic` (8 bytes), `version` (u32), `header_size` (u32), reserved
  - Tiêu chí: WAL mở được, phát hiện format mismatch/version.
- [x] Định nghĩa record header & payload
  - [x] Record header fields:
    - `rec_len` (u32) — payload length
    - `rec_type` (u16)
    - `flags` (u16)
    - `seq_no` (u64) — monotonic
  - [x] Payload (rec_len bytes)
  - [x] `crc32` hoặc `crc64` trên (header + payload)
  - Tiêu chí: replay lặp được, detect partial/truncated record, validate checksum.
- [x] Định nghĩa record types ban đầu
  - [x] TYPE_IDS_UPDATE (payload: idx:u64, value:u64)
  - [x] TYPE_CHECKPOINT (optional) — marks consistent snapshot point
  - [x] (Mở rộng sau cho PPPQ, PPE, Arena...)
- [x] Document endian/packing (use native little-endian; document if cross-platform)

---

## WalManager: API & impl
- [x] Tạo `WalManager` class
  - Methods:
    - [x] `open(path, create_if_missing=true, config)` — mở WAL
    - [x] `append_record(type, payload, len) -> seq_no` — append (sync behavior configurable)
    - [x] `fsync_log()` — flush WAL to disk
    - [x] `replay(apply_cb)` — iterate valid records, call apply callback
    - [x] `truncate_to_zero()` — truncate WAL after successful replay
    - [x] `close()`
  - [x] Internal: fd, seq counter, background group-commit thread (optional)
  - Chấp nhận: API đơn giản, thread-safe append, replay returns deterministic result.

---

## Ordering semantics & Recovery protocol
- [ ] Document chính sách ordering (the durable protocol)
  - Append WAL -> fsync WAL -> atomic_store to mmap (use atomic_utils) -> msync data -> truncate WAL
  - Chấp nhận: nếu crash ở bất kỳ bước nào, replay WAL (which has durable records) tái tạo state.
- [ ] Implement logic cho SoaIdsManager:
  - [ ] Replace/centralize current wal logic bằng WalManager
  - [ ] Ensure atomic_store + msync used for ids updates

---

## Durability modes (configurable)
- [ ] Implement modes:
  - [ ] ALWAYS (fsync per append) — default, simplest & safest
  - [ ] GROUP_COMMIT (batch fsync by background thread) — performance optimization
  - [ ] ASYNC (no fsync) — for best-effort mode (documented)
- [ ] Testing & metrics for each mode

---

## Replay behavior & robustness
- [ ] Replay only valid records (validate checksum + full header+payload length)
- [ ] On encountering partial/truncated record: truncate WAL to valid prefix (or to zero) and continue/abort per policy
- [ ] Idempotence: apply records so that replay multiple times has same final mapping (last-write wins)
- [ ] Ignoring out‑of‑range indexes (log a warning)
- [ ] Acceptance: deterministic mapping after replay; wal file truncated to zero.

---

## Tests (unit & integration)
- Core tests:
  - [ ] `test_wal_replay_single_entry` (manual wal file + open manager => ids updated + wal truncated)
  - [ ] `test_wal_replay_multiple_entries` (duplicate idx, last-write wins)
  - [ ] `test_wal_partial_record_truncates` (write partial record, replay truncates safe)
  - [ ] `test_wal_out_of_range_entries_ignored`
  - [ ] `test_wal_truncate_after_success`
- Crash simulation:
  - [ ] `test_crash_midflow_fork_kill`:
    - Spawn child to run `atomic_update(durable=true)` and parent kills child at different points to simulate crash after wal fsync / before msync / after msync -> verify replay restores mapping.
  - [ ] `test_group_commit_recovery` (if group commit implemented).
- Concurrency tests:
  - [ ] Stress test many concurrent appends & replay verification (no data loss, WAL truncation)
  - [ ] Ensure atomic_store alignment & atomic_utils usage prevent torn writes across threads/processes.
- Integration:
  - [ ] Full end-to-end SoaIdsManager recovery test (create WAL entries, don't touch ids, call open() -> mapping restored)
- Test acceptance: zero inconsistent reads for snapshot patterns; mapping exactly matches expected.

---

## Integration targets (priority)
- High priority:
  - [ ] SoaIdsManager (ids updates)
  - [ ] Pomai startup code that calls replay_wal_and_truncate
- Medium:
  - [ ] PPPQ publish/demote events (store code_nbits + in_mmap)
  - [ ] PPE publish/unpublish
  - [ ] Arena demote/promote (remote blob ids)
- Low:
  - [ ] Checkpoint / snapshots / log compaction

---

## Performance & tuning
- [ ] Measure baseline: latency of atomic_update with WAL_ALWAYS
- [ ] Implement group commit:
  - [ ] Buffer append requests; flush at interval or when buffer size threshold reached
  - [ ] Notify waiting callers when fsync done (use condition variable)
- [ ] Option for batched WAL writes for many small updates
- [ ] Expose metrics: wal_bytes_written, wal_records, fsync_count, group_commit_latency

---

## Safety & correctness notes
- [ ] Use `atomic_utils` for stores/loads into mmap'd memory (already in project)
- [ ] Use `MS_SYNC` when msync data (costly but ensures durability)
- [ ] On filesystems with weaker semantics (NFS) document behavior
- [ ] Sanity-check WAL content on replay (crc); on mismatch, truncate WAL and refuse to apply partial trailing record (log error)

---

## Logging, observability, metrics
- [ ] Emit logs for:
  - WAL open/close
  - Append (type, seq_no, payload_len)
  - Fsync success/failure
  - Replay start/end, records applied, truncated entries
- [ ] Metrics counters
  - wal_records_written, wal_bytes_written, wal_replay_applied, wal_replay_truncated, wal_fsyncs

---

## API for other components
- [ ] Provide typed helpers:
  - `append_ids_update(idx, value, durable=true)`
  - `append_pppq_event(...)`
- [ ] Provide `replay` callback signatures for different subsystems to register apply callbacks

---

## Migration/rollout plan
- [ ] Implement WalManager + tests in feature branch
- [ ] Replace SoaIdsManager's direct WAL logic with WalManager (backwards-compatible)
- [ ] Add feature flag to enable WAL_ALWAYS or GROUP_COMMIT
- [ ] Run integration tests & stress on CI
- [ ] Rollout to production with monitoring (enable WAL in stages)

---

## Misc (docs / maintenance)
- [ ] Document WAL on README: format, durability guarantees, config options
- [ ] Document recovery process and how to debug WAL issues
- [ ] Add helper CLI to inspect WAL file (list records)

---

## Acceptance criteria (summary)
- WAL format defined and implemented with checksum
- WalManager API implemented and thread-safe
- SoaIdsManager uses WalManager and follows ordering: append+fsync WAL -> atomic_store -> msync ids -> truncate WAL
- Full unit/integration tests pass (incl. recovery tests and fork+kill crash simulations)
- WAL truncated after successful replay
- No inconsistent reads observed in PPPQ/publish snapshot tests after integrating WAL

---

## Implementation estimate (rough)
- WalManager core + basic replay + tests: 2–4 days
- Integrate with SoaIdsManager + tests (including fork+kill): 2–3 days
- Group commit + perf tuning + more tests: 3–7 days
- Full rollout & docs: 1–2 days

---

## Next actions (short checklist to start)
- [ ] Create `src/memory/wal_manager.h` + `wal_manager.cc` with minimal API and tests skeleton
- [ ] Add single-entry replay test (`tests/soa/soa_wal_replay_test.cc`)
- [ ] Hook SoaIdsManager to use WalManager in a feature branch
- [ ] Run CI, add stress tests (crash simulation)
- [ ] Iterate on group-commit & optimization
