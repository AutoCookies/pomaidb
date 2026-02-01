# `docs/architecture.md` — PomaiDB Base Phase (v0.1)

## 1. PomaiDB là gì ở giai đoạn base?

PomaiDB base phase là một **embedded vector database** chạy **in-process** (giống SQLite), có:

* **Durability** qua **WAL (Write-Ahead Log)**.
* **Multi-thread** theo mô hình **Shard-Actor**: mỗi shard là một “mini-engine” chạy trên **1 thread riêng**.
* **API đơn giản**: `Open / Put / Delete / Search / Flush / Close`.

Base phase cố tình **tối giản**:

* Không có ANN index (HNSW/IVF) ở giai đoạn này.
* Search hiện tại là **linear scan trong memtable** (đúng/correctness trước).
* Mục tiêu: **đóng đinh mô hình threading + durability + cấu trúc dự án**.

---

## 2. Core principles (không thỏa là fail)

### 2.1 Embedded-first

* Pomai chạy như một thư viện (`libpomai.a`).
* Ứng dụng gọi API trực tiếp, **không server**, không RPC.

### 2.2 Shard-Actor: “1 shard = 1 thread”

* Mọi thay đổi dữ liệu (`Put/Delete/Flush`) chạy trong **thread shard**.
* Không mutate data từ user thread.

**Lợi ích:**

* Tránh lock contention.
* Tính đúng đắn dễ suy luận (deterministic).

### 2.3 WAL là SSOT (Single Source of Truth)

* Mọi mutation phải **append WAL trước** rồi mới apply vào memtable.
* Memtable chỉ là **cache dẫn xuất** từ WAL.

Crash/restart:

* Memtable rebuild bằng cách **replay WAL**.

---

## 3. Project layout và trách nhiệm module

### 3.1 Public API (stable surface)

* `include/pomai/*`

File chính:

* `pomai.h`: interface `pomai::DB`
* `status.h`: `Status`, `ErrorCode`
* `options.h`: `DBOptions`, `FsyncPolicy`
* `types.h`: `VectorId`, `Slice`, `FloatSpan`

Nguyên tắc: code bên `src/` có thể đổi, nhưng API trong `include/` phải ổn định.

### 3.2 API adapter

* `src/api/db.cc`

Nhiệm vụ:

* Implement `DB::Open` và class `DbImpl`.
* `DbImpl` gọi xuống `core::Engine`.
* Lưu scratch vector kết quả search cho `SearchResult`.

### 3.3 Engine (process-level coordinator)

* `src/core/engine/*`

Nhiệm vụ:

* Validate options (path/dim/shard_count).
* Create shard runtimes, start threads.
* Routing `VectorId -> shard`.
* Fan-out `Search` tới các shard theo parallel threads và merge top-k.

### 3.4 Shard runtime (the only mutable path)

* `src/core/shard/*`

Nhiệm vụ:

* `ShardRuntime` chạy event loop.
* Nhận `Command` từ mailbox (MPSC queue).
* Xử lý `Put/Delete/Flush`:

  * `WAL append`
  * `MemTable apply`

Shard runtime là nơi duy nhất có quyền “mutate state”.

### 3.5 Storage

* `src/storage/*`

Hiện base phase có:

* `wal/`: WAL writer + replay
* `manifest/`: file `MANIFEST` dùng để initialize metadata cơ bản

### 3.6 MemTable

* `src/table/*`

* `MemTable` là map `VectorId -> float*`.

* Vector data được lưu trong `Arena` theo block để tránh malloc/free liên tục.

---

## 4. Threading model (đọc phần này là hiểu toàn hệ)

### 4.1 Các loại thread

* **User threads**: thread của application gọi `DB::Put/Search/...`
* **Shard runtime threads**: mỗi shard có 1 `std::jthread`

### 4.2 Luồng `Put/Delete/Flush`

1. User thread gọi `Engine::Put/Delete/Flush`
2. Engine route tới shard
3. Shard tạo `Command` + `std::promise`
4. Shard enqueue vào mailbox (MPSC)
5. User thread chờ `future.get()`
6. Shard runtime thread pop command và xử lý

### 4.3 Tại sao không dùng lock-free queue tự chế trong base?

Base phase ưu tiên:

* Dễ đọc
* Dễ audit correctness

`BoundedMpscQueue` dùng `mutex + condition_variable` để đảm bảo semantic rõ ràng.

---

## 5. Data flow và durability

### 5.1 Put path (WAL-first)

`Put(id, vec)` trong shard thread:

1. Validate dim
2. `wal_->AppendPut(id, vec)`
3. `mem_->Put(id, vec)`

Nếu WAL append fail => Put fail, không apply memtable.

### 5.2 Delete path (WAL-first)

`Delete(id)`:

1. `wal_->AppendDelete(id)`
2. `mem_->Delete(id)`

`MemTable::Delete` hiện là tombstone đơn giản (set pointer null). Base phase chấp nhận.

### 5.3 Flush

`Flush()`:

* Gọi `wal_->Flush()` trên từng shard.
* `FsyncPolicy` trong base:

  * `kAlways`: flush sau mỗi record (hiện chỉ `flush`, chưa làm `fdatasync`)
  * `kOnFlush`: flush khi user gọi `Flush()`
  * `kNever`: không flush cưỡng bức

Base phase dùng `std::ofstream::flush` để đơn giản và portable; production perf sẽ thay bằng `pwrite/pread + fdatasync` (Linux).

---

## 6. WAL format (v1)

WAL là chuỗi record binary:

`RecordHeader` (packed) + payload (cho PUT)

### 6.1 RecordHeader layout

```
uint64 seq
uint8  op        (1 = PUT, 2 = DEL)
uint64 id
uint32 dim       (chỉ dùng cho PUT)
uint64 checksum  (FNV1a-64 của payload)
```

### 6.2 PUT record

* Header + `dim * sizeof(float)` bytes

### 6.3 DEL record

* Header only

### 6.4 Replay rules

* Mở từng segment `wal_{shard}_{gen}.log` theo thứ tự gen tăng dần
* Đọc tuần tự record
* Với PUT:

  * đọc payload
  * verify checksum
  * `mem.Put(id, payload)`
* Với DEL:

  * `mem.Delete(id)`

Replay là idempotent trên memtable hiện tại:

* PUT overwrite id cũ
* DEL set tombstone

---

## 7. Search semantics (base phase)

### 7.1 Search là fan-out + merge

`Engine::Search(query, topk)`:

1. Spawn thread cho từng shard
2. Mỗi shard gọi `ShardRuntime::SearchLocal`
3. `SearchLocal` linear scan `MemTable::ForEach`
4. Mỗi shard trả topk local theo dot-product
5. Engine merge tất cả hits và lấy global topk

### 7.2 Similarity metric

Base phase dùng **dot product**. Không normalize vector.
Production phase sẽ hỗ trợ:

* inner product
* cosine (với pre-normalize)
* L2 (nếu cần)

---

## 8. Invariants (cam kết của base phase)

### 8.1 Single-writer per shard

* Chỉ shard runtime thread mới mutate `WAL/MemTable`.

### 8.2 WAL-first

* Không bao giờ apply memtable trước WAL.

### 8.3 Dim invariant

* `DBOptions.dim` là dim global.
* Mọi vector vào phải đúng dim.

### 8.4 Deterministic replay

* WAL replay theo thứ tự segment + seq.
* Kết quả memtable sau replay phải giống với trước crash (theo record order).

---

## 9. Hạn chế cố ý của base phase

Những thứ base phase chưa làm (sẽ làm ở phase sau):

* Async WAL I/O pipeline (hiện shard thread có thể block khi IO chậm).
* `fdatasync/fsync` thật sự.
* Snapshot/Checkpoint/Compaction.
* Tombstone compaction.
* ANN index (HNSW/IVF) + delta index.
* Budget/deadline/early termination search.
* Crash tests / fuzz tests.

---

## 10. How to run (minimal)

### 10.1 Build

```
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 10.2 Embed usage (pseudo)

* Tạo `pomai::DBOptions { path, shard_count, dim }`
* `pomai::DB::Open(opt, &db)`
* `db->Put(id, vec)`
* `db->Search(query, topk, &res)`
* `db->Flush()`
* `db->Close()`

---

# `docs/wal.md` — WAL Spec (Base v1)

## 1. Objective

WAL đảm bảo:

* Mutation được ghi trước khi apply.
* Restart có thể replay để khôi phục state.

## 2. File naming

Per shard, WAL có nhiều segment:

`wal_{shard_id}_{gen}.log`

`gen` tăng dần, engine tìm file theo thứ tự từ 0 đến hết.

## 3. Record format

Header cố định + payload tuỳ op.

* Op PUT: payload = float[dim]
* Op DEL: payload = empty

Checksum hiện tại là FNV1a-64 trên bytes payload.

## 4. Corruption handling

* Nếu checksum mismatch hoặc truncated record => `Status::Corruption`.
* Base phase dừng mở DB nếu WAL corrupt.

---

# `docs/manifest.md` — Manifest (Base v1)

## 1. Objective

Base manifest chỉ làm:

* Bootstrap db folder
* Store shard_count & dim

File: `MANIFEST`

## 2. Format (text)

```
pomai.manifest.v1
shards <N>
dim <D>
```

Base phase chưa validate manifest consistency; phase sau sẽ:

* versioned schema
* atomic install protocol
* checksums
* generation fencing

---

# `docs/performance.md` — Performance notes (Base)

## 1. What is optimized in base

* Shard actor avoids locks on mutation path.
* Arena reduces allocation churn for vector storage.

## 2. What is intentionally not optimized yet

* WAL uses `ofstream` flush (not fdatasync).
* Search is O(n) scan.
* Engine search spawns threads per request (phase sau sẽ dùng thread pool + budget).

Run tsan test
```
rm -rf build/tsan
cmake -S . -B build/tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPOMAI_BUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"

cmake --build build/tsan -j
ctest --test-dir build/tsan -V
```