Ok. “Bigtech standard” thì câu trả lời là: **checkpoint vẫn cần**, nhưng **không được phép** là “full dump chặn hot path” như hiện tại. Production-way phải có 3 thứ: **(1) WAL đúng semantics**, **(2) snapshot/checkpoint incremental + atomic publish**, **(3) background scheduling có throttle + backpressure**.

Dưới đây là blueprint chuẩn production (đúng kiểu RocksDB/Lucene-ish, nhưng tối giản cho embedded vector DB).

---

## 0) Mục tiêu chuẩn production

PomaiDB (embedded) phải đảm bảo:

1. **Durability contract rõ**: ack Upsert nghĩa là gì (fsync hay chỉ append vào OS cache).
2. **Crash recovery bounded**: restart không replay WAL vô hạn.
3. **Không phá tail latency**: checkpoint không được làm ingest/search “đứng hình”.
4. **Atomicity**: checkpoint publish phải atomic (không có state nửa vời).
5. **Tunable**: user chỉnh được cost/latency/durability.

---

## 1) Tách rõ “Data plane” vs “Maintenance plane”

### Data plane (hot path)

* Shard worker threads: xử lý Upsert/Delete/Search.
* WAL append (nhanh).
* Update in-memory index.

### Maintenance plane (background)

* **MaintenanceScheduler**: 1 thread (hoặc threadpool size=1).
* Làm:

  * checkpoint/snapshot incremental
  * WAL recycling (truncate / switch file)
  * index persistence (nếu bật)
  * compaction segments (nếu blob store dùng segments)

> Bigtech standard: **hot path không được gọi trực tiếp “CreateCheckpoint() full”**.

---

## 2) WAL semantics chuẩn (phải chốt trước)

Bạn define 3 durability levels (giống RocksDB):

* **Durability::kNone**: ack sau khi enqueue (chỉ dùng test).
* **Durability::kWAL**: ack sau khi WAL append + flush (fdatasync optional).
* **Durability::kFsync**: ack sau khi WAL fdatasync (đắt, nhưng mạnh).

**Contract**:

* Nếu ack OK ở kWAL/kFsync, record *phải* tồn tại sau crash theo đúng level.

Thực thi:

* Mỗi shard có WAL riêng ✅
* WAL là append-only, record framing + checksum ✅

---

## 3) Checkpoint production ≠ “save everything”

Checkpoint production phải **incremental** và **bounded cost**.

### 3.1. Storage layout chuẩn

Mỗi shard có thư mục riêng:

```
db/
  shard_0/
    wal/
      000001.log
      000002.log
    segments/        # blob store data (append-only)
      seg_00010.dat
      seg_00011.dat
    manifest/        # metadata nhỏ, atomic
      MANIFEST
    checkpoints/
      cp_000123/     # optional, or just a metadata file
```

### 3.2. “Checkpoint” thực ra là **publish a consistent view**

Cái cần durable là:

* mapping `VectorId -> (segment_id, offset, len)` hoặc tương đương
* “sequence number / epoch” đã apply
* (optional) index persistence artifact

**KHÔNG** cần dump “arena pages” toàn bộ mỗi lần.

#### Atomic publish chuẩn

* Ghi metadata mới ra file temp
* fsync temp
* rename temp -> MANIFEST (atomic on POSIX)
* fsync directory (để rename durable)

---

## 4) Snapshot isolation (để checkpoint chạy song song hot path)

Bigtech-standard approach:

### 4.1. MVCC-lite bằng “epoch”

* Mỗi shard có `atomic<uint64_t> applied_seq`.
* Mọi mutation tăng seq.
* Search đọc snapshot = seq tại thời điểm bắt đầu.

### 4.2. BlobStore append-only + immutable segments

* Upsert ghi payload vào segment mới (append).
* Mapping id->location update ở memtable (có thể copy-on-write nhỏ, hoặc sharded map).

### 4.3. Checkpoint chỉ cần “freeze a view”

* Freeze “manifest view” tại seq X:

  * segment list + id->location snapshot (hoặc delta logs)
* Không động vào hot index, không dump data cũ.

**Quan trọng**: nếu bạn vẫn muốn “arena_->Ids() copy toàn bộ” thì đó là anti-production. Snapshot phải là **handle** (shared_ptr) chứ không phải copy O(n).

---

## 5) Index persistence: 2 mode chuẩn

HNSW save/load rất đắt. Bigtech làm 1 trong 2:

### Mode A (đề xuất mặc định): **Index rebuild on startup**

* Checkpoint chỉ giữ vectors/payload mapping.
* On restart: load vectors rồi rebuild HNSW.
* Ưu: code đơn giản, checkpoint rẻ.
* Nhược: restart lâu (nhưng predictable).

### Mode B: **Index persistence async**

* Background thread thỉnh thoảng snapshot index (rất ít, manual hoặc daily).
* Hot path không block.
* Khi có file index mới nhất: startup load nhanh.

**Production default thường là A**, và cho enterprise user bật B nếu họ cần fast restart.

---

## 6) Scheduling checkpoint kiểu bigtech (throttle + budgets)

Checkpoint trigger theo 3 điều kiện:

* `wal_bytes > threshold` (vd 256MB)
* `time_since_last > interval` (vd 10–30 phút)
* `ops_since_last > N` (vd 1–5 triệu)

Nhưng chạy checkpoint phải có **budget**:

### CPU/IO throttling (bắt buộc)

* Background thread `nice +5`
* IO throttling: giới hạn `MB/s` và `fsync frequency`
* “Stop if tail latency tăng”: nếu shard queue depth tăng > ngưỡng → pause maintenance.

### Backpressure safety

* Nếu WAL vượt hard limit (vd 8GB):

  * hoặc block writes (return `Status::Busy` / `TryAgain`)
  * hoặc force checkpoint (nhưng vẫn budgeted, không stop-the-world)

Bigtech standard là **predictable behavior** chứ không “random lag spike”.

---

## 7) API surface chuẩn production

Bạn cần các API/options này (tối thiểu):

### DbOptions

* `num_shards`
* `durability_level`
* `checkpoint.enable` (default true)
* `checkpoint.wal_bytes_threshold` (default 256MB)
* `checkpoint.min_interval_ms` (default 10min)
* `checkpoint.max_interval_ms` (default 60min)
* `checkpoint.max_bytes_per_sec` (IO throttle)
* `checkpoint.mode = {kManifestOnly, kWithIndex}`

### Runtime API

* `Flush(FlushOptions)` – block cho tới khi WAL durable + mapping durable
* `Checkpoint(CheckpointOptions)` – block hoặc async
* `GetStats()` – expose queue depth, wal bytes, last checkpoint time
* `BeginBulkLoad()/EndBulkLoad()` – tắt maintenance + giảm fsync, cuối bulk thì checkpoint 1 phát

Bulk load là tiêu chuẩn production (Lucene/RocksDB đều có concept tương tự).

---

## 8) “Bạn đang làm sai ở đâu” so với production

Trong code hiện tại của bạn, “checkpoint” đang là:

* full snapshot arena pages
* full index save
* truncate WAL
* chạy inline theo interval nhỏ (50k)

=> Đây là **anti-pattern** vì:

* cost checkpoint ~ O(dataset)
* frequency cao → throughput collapse
* chạy trên hot path → tail latency spike

Production checkpoint phải:

* cost ~ O(delta) hoặc O(metadata)
* frequency theo WAL size/time
* chạy background + throttle
* publish atomic + recoverable

---

## 9) Migration plan: làm đúng production mà ít rewrite nhất

Nếu muốn “most production” nhưng không phá codebase quá mạnh, làm theo thứ tự:

### Step 1 — Immediate stop-the-bleeding (1 ngày)

* Disable auto checkpoint trong ingest benchmark.
* Remove `MaybePublishSnapshot()` kiểu copy ids (thay bằng pointer/version).
* Search trả id+score, payload lazy hoặc load sau merge.

### Step 2 — Manifest-based checkpoint (2–4 ngày)

* Biến checkpoint thành “publish manifest + rotate WAL”
* BlobStore chuyển sang segments append-only (nếu chưa)
* Recovery: load manifest + replay WAL files mới hơn manifest seq.

### Step 3 — Maintenance thread + budgets (2–3 ngày)

* Thêm `MaintenanceScheduler` single thread.
* Checkpoint trigger theo wal_bytes/time + throttle.

### Step 4 — Optional index persistence (sau)

* Cho phép save/load HNSW async, không block ingest.

Cách này sẽ đưa Pomai lên đúng chuẩn “production embedded DB” mà vẫn giữ triết lý shard WAL.

---

## Chốt câu theo kiểu bigtech

**PomaiDB production vẫn cần checkpoint**, nhưng checkpoint **không được phép** là “snapshot full dataset mỗi 50k ops”.
Checkpoint production là **manifest + wal recycling + optional index persistence**, chạy background có throttle, có backpressure, và publish atomic.

Nếu bạn muốn mình “đóng vai lead engineer” và ra luôn **spec chuẩn + danh sách thay đổi vào đúng các file bạn đưa** (wal.*, shard.*, router.*, blob_store.*), mình làm ngay theo plan ở trên — ưu tiên thay đổi ít nhưng đúng chuẩn.
