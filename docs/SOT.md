# PomaiDB Single Source of Truth (SOT)

> **Authority**: This document is the **FINAL AUTHORITY** on PomaiDB's behavior. If code differs from this document, the **code is wrong** (unless this document is explicitly updated via a design review).
> **Scope**: Semantics, Contracts, Invariants, and Failure Models.

---

## 1. System Identity & Core Invariants

PomaiDB is a **single-process, embedded, sharded vector database** with **snapshot isolation** and **single-writer-per-shard** architecture.

### 1.1 Core Invariants (MUST HOLD)
1.  **Single Writer Per Shard**: Each shard has exactly one thread responsible for mutations. No locks required for writes within the shard loop.
2.  **WAL is Truth**: All mutations (Put, Delete) are appended to the Write-Ahead Log (WAL) *before* being applied to in-memory state.
3.  **Snapshot Isolation**: Readers observe a consistent, immutable view of the database.
    *   A snapshot implies a specific point in the WAL sequence.
    *   Once published, a snapshot is immutable.
    *   Readers NEVER block writers. Writers NEVER block readers.
    *   **Unified View**: A snapshot includes **frozen memtables** and **disk segments**.
    *   **Active Memtable Visibility**: *[Decision: Consistent Read-Your-Writes]*
        *   Get/Exists/Search/Scan operations **MUST** observe the Active Memtable if performed by the writing context (providing strict read-your-writes) OR if explicitly requested/configured.
        *   Default public API (`DB::Search`, `DB::Get`) provides **Eventual Consistency** (bounded staleness) by default, viewing only the latest *published* snapshot, UNLESS `ReadOptions.read_own_writes` is true (which may require synchronization).
        *   *Refinement for P0.2*: To ensure unified semantics, the internal `ShardRuntime` query path merges: `Active Memtable` + `Frozen Memtables` + `Segments`.
        *   **Merge Order (Newest Wins)**: Active > Frozen (Newest to Oldest) > Segments (Newest to Oldest).
        *   **Tombstones**: A delete in a newer layer strictly hides ANY data in older layers.

## 2. API Semantics

### 2.1 DB::Open / DB::Close (Lifecycle)
*   **DB::Open(options)**:
    *   **Atomic Logic**: Creating the database directory (if new) is atomic.
    *   **Failure**: Returns `Status::Error`, NEVER `Status::OK` if partially initialized.
        *   Must clean up partial state on failure (e.g., remove lockfiles, close handles).
        *   Existing "open-in-constructor" anti-patterns must be removed.
    *   **Recovery**: Automatically plays back WALs to restore state.
*   **DB::Close()**:
    *   Gracefully shuts down shard threads.
    *   Flushes pending WAL writes (based on fsync policy).
    *   Releases all locks.
    *   Subsequent calls to DB return `Status::Closed`.

### 2.2 Write Operations (Put, PutBatch, Delete)
*   **Put(id, vector, metadata) / PutBatch**:
    *   **Atomicity**: Each Put is atomic. Batch atomicity is *not* guaranteed across shards, but *is* guaranteed within a shard for a single batch command (all or nothing in WAL write).
    *   **Durability**: Defined by `WriteOptions.fsync`:
        *   `true`: Returns only after WAL fsync.
        *   `false`: Returns after buffering (process crash may lose data).
    *   **Semantics**: Upsert. Overwrites existing ID.
    *   **Zero-Copy Ingest**: `PutBatch` must minimize vector copying.
*   **Delete(id)**:
    *   Appends Tombstone to WAL.
    *   Hides all previous versions of `id`.

### 2.3 Read Operations (Get, Exists, Search)
*   **Get(id) / Exists(id)**:
    *   **Scope**: Checks Active Memtable -> Frozen Memtables -> Segments.
    *   **Tombstones**: If most recent record is Tombstone -> NotFound / False.
*   **Search(query, k)**:
    *   **Scope**: Same merging logic as Get.
    *   **Recall**: Exact search over MemTables (brute force), Index search over Segments (or brute force if no index).
    *   **Filtering**: Filtering applies *before* top-k selection (conceptually).
    *   **Deleted Items**: MUST NOT appear in results.

### 2.4 Snapshot Iterator & Scan (`DB::Scan`)
*   **Semantics**: Iterates over a *fixed* snapshot.
*   **Deterministic Order**:
    *   Strict order: `ShardID` ASC -> `Segment/Memtable` Age (Oldest to Newest? No, usually stable ID order implies sorting).
    *   *SOT Decision*: **Shard ID ASC -> ID ASC**. (This requires merging or sorting logic, but provides user predictability).
    *   Alternatively: **Shard ID ASC -> Storage Unit ID ASC -> Vector ID ASC** (Faster, exposing internal structure).
    *   *Decision*: **Shard ID -> Vector ID**. If too expensive, clearly document relaxed ordering. For now: **Deterministic per snapshot**.
*   **Consistency**:
    *   `Next()` never jumps back.
    *   Same snapshot handle => Exact same sequence of records.
*   **Visibility**: Respects tombstones.

## 3. Error Model
*   **Multi-Shard Failures**:
    *   If one shard fails during Search (e.g., corruption, timeout):
    *   *Policy*: **Partial Results with Warning**.
    *   Return `Status::PartialFailure` containing valid results from healthy shards and a list of errors from failed shards.
*   **Fail-Fast**: For critical ops (Put/Snapshot), failure in any shard aborts the operation.

## 4. Durability & Failure Semantics
*   **Crash Recovery**:
    *   On start: Replay WALs.
    *   Corrupt WAL frame: Stop replay, truncate (if configured to salvage) or Fail (default).
    *   **Fail-Closed**: If data corruption is detected (checksum mismatch), DB refuses to open.
*   **Fsync Policy**:
    *   Controlled via `DBOptions`.
    *   Default: `WriteOptions.fsync = false` (Performance), `DBOptions.wal_fsync = true` (Safety? No, mostly async). *Clarify usage*.

## 5. Storage Format
*   **Metadata**:
    *   Schema: `{id (u64), vector (float[]), metadata (schema-less or rigid?), sparse (tokens)}`.
    *   Metadata: `timestamp` (i64), `tenant_id` (u64/str), `tags` (set).
*   **On-Disk**:
    *   Segments: Immutable, versioned (`pomai.seg.v2`).
    *   Manifest: Atomic replacement.

## 6. Hybrid Search & Scoring
*   **Score**: `S = alpha * DenseSim(v_q, v_d) + (1-alpha) * SparseSim(s_q, s_d)`
*   **Candidate Generation**:
    *   Option 1: Dense ANN -> Rerank with Sparse.
    *   Option 2: Sparse Inverted Index -> Rerank with Dense.
    *   *SOT*: System chooses based on query/data, or strictly defined (e.g. "Hybrid First"). Defaults to **Dense First** unless sparse query is highly selective (future optimization). Simple safe default: **Dense ANN**.

## 7. Metadata Filtering
*   **Predicate Support**: `==`, `!=`, `<, >, <=, >=`, `IN`, `TimestampRange`.
*   **Implementation**:
    *   Must work on MemTables (scan) and Segments (index/scan).
    *   ** correctness**: `Search(filter)` returns top-k *matching* the filter. Never return non-matching items.

---
**Versioning**
v1.0 - Initial SOT for Production Readiness.
