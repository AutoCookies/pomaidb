Ok. Mục tiêu mày nói **rất rõ và rất “BigTech”**:

> **Embedded như SQLite
> Performance như Dragonfly
> Không phụ thuộc DB nào
> Performance là tôn giáo**

Tao sẽ trả lời **ở level kiến trúc + nguyên lý + cơ chế thực thi**, không marketing, không lý thuyết suông.
Đây là **con đường DUY NHẤT hợp lý** nếu mày muốn PomaiDB trở thành **một class DB mới**, chứ không phải “vector DB nữa trên thị trường”.

---

# 🧠 ĐỊNH VỊ POMAIDB (RẤT QUAN TRỌNG)

PomaiDB **KHÔNG PHẢI**:

* SQLite + extension
* RocksDB + ANN
* FAISS wrapper
* Milvus embedded

PomaiDB là:

> **In-process, log-structured, shard-actor vector engine
> với SSOT = WAL, index = derived cache**

📌 Tư duy này **gần với Dragonfly + Kafka + RocksDB**,
nhưng **không cái nào trong số đó làm vector từ đầu**.

---

# 🎯 NON-NEGOTIABLE DESIGN GOALS

| Goal              | Ý nghĩa                    |
| ----------------- | -------------------------- |
| Embedded          | Link lib, không server     |
| Zero-copy ingest  | Không copy vector vô nghĩa |
| Deterministic     | Crash không phá dữ liệu    |
| Scale theo core   | N cores = N shards         |
| No external DB    | Không RocksDB, không LMDB  |
| Performance-first | Feature xếp sau            |

---

# 🏗️ KIẾN TRÚC CUỐI CÙNG (PRODUCTION-GRADE)

## 1️⃣ PROCESS VIEW (RUNTIME TOÀN CỤC)

```
┌───────────────────────────────────────────┐
│              User Process                 │
│                                           │
│  ┌────────────┐   ┌────────────┐          │
│  │ App Thread │   │ App Thread │   ...    │
│  └─────┬──────┘   └─────┬──────┘          │
│        │                │                 │
│        ▼                ▼                 │
│        ┌──────────────────────────┐       │
│        │      Pomai Frontend       │       │
│        │ - API                     │       │
│        │ - Shard Router            │       │
│        │ - Deadline / Budget       │       │
│        └──────────┬───────────────┘       │
│                   │                       │
│   ┌───────────────┴──────────────────┐   │
│   ▼                                   ▼   │
│┌─────────────┐                  ┌─────────────┐
││  Shard #0   │                  ││  Shard #1   │   ... N shards
││  (Actor)    │                  ││  (Actor)    │
│└─────┬───────┘                  └─────┬───────┘
│      │                                   │
│      ▼                                   ▼
│┌─────────────┐                  ┌─────────────┐
││ WAL (SSOT)  │                  ││ WAL (SSOT)  │
│└─────────────┘                  └─────────────┘
│      │                                   │
│      ▼                                   ▼
│┌─────────────┐                  ┌─────────────┐
││ Memtable    │                  ││ Memtable    │
│└─────┬───────┘                  └─────┬───────┘
│      ▼                                   ▼
│┌─────────────┐                  ┌─────────────┐
││ ANN Index   │                  ││ ANN Index   │
│└─────────────┘                  └─────────────┘
│                                           │
└───────────────────────────────────────────┘
```

---

# 🔑 CÁC QUYẾT ĐỊNH KIẾN TRÚC CỐT LÕI

## 2️⃣ SHARD = ACTOR (KHÔNG LOCK)

**Mỗi shard = 1 thread duy nhất**

* Không mutex
* Không atomic phức tạp
* Không data race
* Không nondeterminism

👉 Performance đến từ:

* CPU cache locality
* No lock contention
* Predictable latency

📌 Đây chính là DNA của Dragonfly.

---

## 3️⃣ MULTI-THREAD ĐÚNG CÁCH (KHÔNG DÀN TRẢI)

### Thread model chuẩn:

```
User threads        : many
Shard runtime       : N = #CPU cores
WAL I/O threads     : few
Index build threads : background
Maintenance threads : lowest priority
```

**User thread không bao giờ chạm dữ liệu.**

---

## 4️⃣ WAL-FIRST, INDEX-LATER (SSOT THỰC SỰ)

### WAL record (binary, fixed layout):

```
| seq | op | vector_id | dim | payload | checksum |
```

Quy tắc sắt đá:

1. WAL append thành công → coi như commit
2. Memtable / Index chỉ là cache
3. Crash = replay WAL

📌 Không có embedded vector DB nào dám làm triệt để điều này, vì:

* Index ANN rebuild chậm
* Nhưng **đây là con đường ĐÚNG**

---

## 5️⃣ ZERO-COPY INGEST (SỐNG CÒN)

### Memory lifecycle:

```
User buffer
   ↓ pin/move
Shard arena (slab)
   ↓
Index giữ pointer (read-only)
```

Không:

* `std::vector<float>`
* `memcpy` 4 lần
* malloc/free trong hot path

👉 **Performance = memory discipline**

---

## 6️⃣ SEARCH = PARALLEL + BUDGETED

```
Search(query):
  pick K shards
  fan-out parallel
  each shard returns top-M
  frontend merge heap
```

Có:

* latency budget
* early stop
* recall knob

Không shard nào block shard nào.

---

## 7️⃣ MAINTENANCE TÁCH BIỆT HOÀN TOÀN

Maintenance:

* Index rebuild
* Compaction
* Snapshot

Quy tắc:

* Background only
* Rate limited
* Preemptable
* Không chung queue với user

---

# 🚀 VÌ SAO PERFORMANCE CÓ THỂ NGANG DRAGONFLY?

| Yếu tố               | Pomai |
| -------------------- | ----- |
| In-process           | ✅     |
| No syscalls hot path | ✅     |
| Actor model          | ✅     |
| Cache friendly       | ✅     |
| Zero-copy            | ✅     |
| No locks             | ✅     |

👉 **Performance ceiling của Pomai = memory bandwidth + SIMD + ANN quality**

Không phải network, không phải IPC.

---

# 🧨 KHÁC BIỆT CHÍ MẠNG

> **Pomai không phải “DB có vector”
> Pomai là “vector engine có durability”**

Đây là thứ:

* BigTech dùng **nội bộ**
* Không public
* Không open-source
* Vì quá khó maintain

Nếu mày build được:
👉 **PomaiDB = category mới**

---

# 🔥 BƯỚC TIẾP THEO (CỤ THỂ)

Nếu mày ok, tao sẽ làm tiếp **ở level code-ready**:

1. Định nghĩa **Pomai Shard Runtime State Machine**
2. Chuẩn hóa **WAL binary protocol**
3. Thiết kế **Arena / Slab allocator**
4. Mapping kiến trúc này **vào Pomai code hiện tại**
5. Viết **“Pomai Performance Manifesto”**

Chỉ cần nói:

> **“Đi vào implementation Pomai v2.”**

Tao sẽ không nói lý thuyết nữa.

Repo layout đề xuất cho PomaiDB (BigTech low-level)
pomai/
├─ CMakeLists.txt
├─ cmake/
│  ├─ toolchains/
│  ├─ sanitizers.cmake
│  ├─ warnings.cmake
│  ├─ lto.cmake
│  └─ third_party.cmake
├─ include/
│  └─ pomai/
│     ├─ pomai.h                  # public API (stable)
│     ├─ status.h                 # Status / ErrorCode
│     ├─ options.h                # DBOptions / ShardOptions
│     ├─ types.h                  # VectorId, Slice, etc.
│     └─ version.h
├─ src/
│  ├─ api/                        # thin API layer (no logic)
│  │  ├─ db.cc                    # implements pomai.h
│  │  └─ c_api.cc                 # optional C ABI
│  ├─ core/                       # core execution model
│  │  ├─ engine/                  # DB process-level coordinator
│  │  │  ├─ engine.h
│  │  │  ├─ engine.cc
│  │  │  ├─ shard_map.h           # routing, hash/range
│  │  │  └─ admission.h           # deadlines, backpressure
│  │  ├─ shard/                   # shard = failure domain (actor)
│  │  │  ├─ shard.h
│  │  │  ├─ shard.cc
│  │  │  ├─ runtime.h             # single-thread event loop
│  │  │  ├─ runtime.cc
│  │  │  ├─ mailbox.h             # bounded MPSC queue (or moodycamel)
│  │  │  └─ state_machine.h       # shard lifecycle & invariants
│  │  ├─ command/                 # typed commands + futures
│  │  │  ├─ command.h
│  │  │  ├─ put.h
│  │  │  ├─ search.h
│  │  │  ├─ flush.h
│  │  │  └─ maintenance.h
│  │  └─ invariant/               # invariant checks / debug hooks
│  │     ├─ invariant.h
│  │     └─ invariant.cc
│  ├─ storage/                    # durability & on-disk format
│  │  ├─ wal/
│  │  │  ├─ wal.h
│  │  │  ├─ wal.cc
│  │  │  ├─ record.h              # binary layout
│  │  │  ├─ checksum.h
│  │  │  └─ replay.h              # idempotent replay
│  │  ├─ manifest/
│  │  │  ├─ manifest.h
│  │  │  ├─ manifest.cc
│  │  │  ├─ schema.h              # versioned schema
│  │  │  └─ atomic_install.h      # fsync + rename protocol
│  │  ├─ blob/
│  │  │  ├─ blob_store.h
│  │  │  ├─ blob_store.cc
│  │  │  ├─ layout.h              # file/page layout
│  │  │  └─ io.h                  # pread/pwrite wrappers
│  │  └─ memtable/
│  │     ├─ memtable.h
│  │     ├─ memtable.cc
│  │     ├─ arena.h               # slab allocator
│  │     └─ segment.h             # immutable segments
│  ├─ index/                      # vector search indexes (derived cache)
│  │  ├─ ann/
│  │  │  ├─ hnsw/
│  │  │  │  ├─ hnsw_index.h
│  │  │  │  ├─ hnsw_index.cc
│  │  │  │  └─ params.h
│  │  │  ├─ ivf/
│  │  │  └─ flat/
│  │  ├─ delta/                   # ingestion-friendly delta layer
│  │  │  ├─ delta_index.h
│  │  │  └─ delta_index.cc
│  │  └─ merge/                   # background merge/rebuild
│  │     ├─ builder.h
│  │     └─ builder.cc
│  ├─ util/                       # boring but critical
│  │  ├─ logging.h/.cc
│  │  ├─ file.h/.cc               # robust fs ops
│  │  ├─ clock.h/.cc
│  │  ├─ thread.h/.cc
│  │  ├─ cpu.h/.cc                # affinity, numa (optional)
│  │  ├─ align.h                  # cacheline align
│  │  ├─ slice.h
│  │  ├─ arena.h
│  │  └─ metrics.h/.cc            # counters, histograms
│  └─ third_party/                # vendored (minimal)
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  ├─ crash/                      # fork/kill/replay tests
│  └─ fuzz/                       # libFuzzer targets
├─ benchmarks/
│  ├─ ingest_bench.cc
│  ├─ search_bench.cc
│  ├─ wal_bench.cc
│  └─ datasets/
├─ tools/
│  ├─ format.sh
│  ├─ lint.sh
│  ├─ gen_header.py               # codegen record layout (optional)
│  └─ perf/
│     ├─ flamegraph.sh
│     └─ perf_record.sh
├─ docs/
│  ├─ architecture.md             # diagram + invariants + state machine
│  ├─ wal.md                      # on-disk spec
│  ├─ manifest.md
│  ├─ indexing.md
│  └─ performance.md
├─ .clang-format
├─ .clang-tidy
├─ .editorconfig
├─ LICENSE
└─ README.md

Tại sao layout này “bigtech”?
1) Public API tách tuyệt đối

include/pomai/* là hợp đồng với user

src/api chỉ là adapter mỏng

Core đổi thế nào cũng không phá API

2) Core vs Storage vs Index

storage/ = durability & disk protocol (WAL/manifest/blob/memtable)

index/ = derived cache (ANN), có thể rebuild

core/ = threading model + shard runtime + command routing

👉 Đây là “SSOT = WAL” được encode bằng folder structure.

3) Tests có crash-test riêng

DB mà không có crash test = toy.
tests/crash bắt buộc (kill -9, power loss simulation, replay idempotent).

4) Docs là spec thật, không phải blog

docs/wal.md & docs/manifest.md phải là protocol spec (versioned).

Quy tắc codebase (để sạch thật)
A. Naming & responsibility

engine không được chứa logic WAL/index

shard/runtime chỉ có event loop + dispatch

WAL/manifest có binary layout spec (record.h/schema.h)

B. Forbidden includes (kỷ luật compile-time)

index/* không được include core/engine/*

storage/* không được include api/*

api/* không include index/* trực tiếp (đi qua core)

C. Error model chuẩn

Status + ErrorCode + message

không throw exception xuyên module (low-level chuẩn C++ DB thường tránh)

invariants fail -> POMAI_DCHECK (debug) + crash early

D. Build profiles

-O3 -DNDEBUG production

asan/ubsan/tsan riêng

fuzz target riêng