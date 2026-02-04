# Canonical Search Plan: IVF + Exact Rerank

## 1. Objectives
-   **High Recall**: >= 0.94 recall@1/10/100.
-   **Performance**: Sub-linear search scaling using IVF (Inverted File Index).
-   **Consistency**: Strict snapshot isolation. Readers see a fixed point-in-time view.
-   **Simplicity**: One canonical path. No index zoo.

## 2. Architecture

### 2.1. The Canonical Pipeline
The search operation follows a strictly defined "Coarse -> Gather -> Rerank" pipeline.

1.  **Coarse Stage (Routing)**:
    -   Compute distances between Query and **Centroids**.
    -   Select top `nprobe` centroids.
    -   *Input*: Query.
    -   *Output*: List of candidate Centroid IDs.

2.  **Gather Stage (Candidate Selection)**:
    -   Fetch Posting Lists for selected centroids.
    -   Accumulate Candidate Vector IDs.
    -   *Constraint*: Cap candidates at `max_candidates` per segment to prevent OOM.

3.  **Exact Rerank Stage**:
    -   Fetch full vectors for all candidates.
    -   Compute exact Dot Product (SIMD optimized).
    -   Maintains a generic Top-K Min-Heap.
    -   *Constraint*: Tie-breaking must be deterministic (Score desc, ID asc).

### 2.2. Indexing Strategy: Segment-Level IVF
Indices are immutable and scoped 1:1 with Segments.

-   **Structure**: `IVFFlat` (Centroids + Adjacency List of IDs).
-   **Persistence**: Sidecar file (e.g., `segment_X.idx`) alongside `segment_X.dat`.
-   **Lifecycle**:
    -   Built during `Freeze` (MemTable -> Segment).
    -   Built during `Compact` (Merged Segment -> New Index).
    -   Loaded via mmap/read-all during `DB::Open`.

### 2.3. Snapshot Semantics
Search adheres to strict Snapshot Isolation.
Layers (searched in order, deduped via `SeenTracker`):
1.  **Active MemTable**: Brute-force scan (small, <5000 items).
2.  **Frozen MemTables**: Brute-force scan (transient).
3.  **Segments**: IVF Search (dominant data).

**Note on Consistency**: Unlike previous versions, Search **DOES** scan the Active MemTable, ensuring `Put(X) -> Search(X)` works instantly (Read Your Writes) if X is top-k.

## 3. Configuration Knobs (Minimal Zoo)

| Knob | Default | Scope | Description |
| :--- | :--- | :--- | :--- |
| `nlist` | 64 | Build | Number of centroids (set via `IndexParams`). |
| `nprobe`| 8 | Query | Number of centroids to search (set via `IndexParams`). |
| `max_candidates` | 4096 | Query | Max candidates to rerank per segment (Hardcoded for now). |
| `rerank_k` | TopK | Query | Implicitly TopK. |

## 4. Multi-Threading Model
-   **Shard-Level**: Queries are distributed to Shards (existing).
-   **Segment-Level**: Inside a Shard, large segments are searched in parallel using a bounded `ThreadPool`.
-   **Scratch Buffers**: All temporary buffers (scores, candidates) are thread-local or pooled. No per-query allocations in hot loop.

## 5. Invariants & Fallbacks
-   **Fallback**: If `.idx` is missing/corrupt, segment falls back to Brute Force scan.
-   **Determinism**: KMeans uses fixed seed. Rerank uses stable sort.
-   **Safety**: If `max_candidates` is reached, we drop remaining candidates? No, we prioritize by coarse distance? Currently just "first nprobe lists".
