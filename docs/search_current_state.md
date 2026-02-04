# Search Pipeline: Current State Report

## 1. Overview
The current search implementation in PomaiDB (`ShardRuntime::Search`) is a **Brute-Force, 1-Pass Merge Scan** over immutable data snapshots. It provides strong consistency and isolation but lacks any indexing, resulting in O(N) search complexity.

## 2. Architecture & Data Sources

### Read Path (`ShardRuntime::Search`)
The search pipeline is lock-free and operates on a `ShardSnapshot`.
1.  **Acquire Snapshot**: `current_snapshot_.load()`.
2.  **Helpers**: 
    -   `SeenTracker`: Tracks visited IDs to ensure "Newest Version Wins" semantics and handle deduplication.
    -   `std::vector<SearchHit> + std::push_heap/pop_heap`: maintains top-K results.
3.  **Scan Layers** (Newest to Oldest):
    -   **Frozen MemTables** (`snap->frozen_memtables`): In-memory immutable hashmaps (`std::unordered_map`). Scanned via iteration.
    -   **Segments** (`snap->segments`): On-disk flat files (Header + Packed Entries). Scanned via `ForEach` (reads every entry).
4.  **Result**: Returns exact top-K by dot product.

### Invariants & Semantics
-   **Snapshot Isolation**: Readers see a consistent view of `frozen_memtables` + `segments` fixed at `PublishSnapshot` time.
-   **Visibility Latency**: `Search` **DOES NOT** scan the Active MemTable. Newly inserted data is only searchable after it rotates to "Frozen" (triggered by `Put` threshold or manual flush). This differs from `Get`/`Exists`, which **DO** check the Active Memtable.
-   **Tombstones**: 
    -   Represented as `nullptr` in MemTable and `kFlagTombstone` in Segments.
    -   `SeenTracker` ensures that if a newer Tombstone is seen, older versions of the same ID are ignored.

## 3. Physical Data Layout
-   **MemTable**: `std::unordered_map<VectorId, float*>` backed by a linear `Arena`.
    -   Vectors are stored contiguously in arena blocks.
    -   No index structures.
-   **Segment (V2)**:
    -   File format: `[Header] [Entry0] ... [EntryN]`.
    -   Entry: `ID (8B) | Flags (1B) | Vector (4B * Dim)`.
    -   Layout is **Flat**. No centroids, no inverted lists.

## 4. Why IVF is Bypassed
Usage of `IvfCoarse` is **currently disconnected** in the codebase:
1.  **Ingestion Disconnect**: `ShardRuntime::HandlePut` updates `mem_` (MemTable) but **never calls** `ivf_->Put()`. The index remains empty/untrained.
2.  **Deletion Partial-Connect**: `ShardRuntime::HandleDel` calls `ivf_->Delete()`, which is a no-op since the index is empty.
3.  **Read Path Bypass**: `ShardRuntime::SearchLocalInternal` performs a full scan of all layers. It does not call `ivf_->SelectCandidates()`.
4.  **Serialization Missing**: `IvfCoarse` is in-memory only. There is no code to serialize/deserialize the index to/from disk (Segments).

## 5. Summary of Gaps
| Feature | Current State | Required State (Target) |
| :--- | :--- | :--- |
| **Search Algo** | O(N) Brute Force | IVF + Candidate Gather + Rerank |
| **Index Scope** | Global (Unused) | Per-Segment |
| **Persistence** | None | Saved alongside Segments |
| **Active Data** | Invisible to Search | Should be visible (Brute forced if small) |
| **IVF Code** | Disconnected | Integrated into Segment Lifecycle |
