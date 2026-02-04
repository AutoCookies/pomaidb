# PomaiDB Database Contract (Constitution)

> **Authority**: This document is the final authority on PomaiDB's identity, guarantees, and boundaries.  
> **Status**: Living document, updated only through explicit design review.  
> **Branch**: `pomai-embeded`

---

## What PomaiDB IS

### Core Identity

PomaiDB is an **embedded vector database engine** with the following non-negotiable characteristics:

#### 1. Embedded Database
- **Single-process, in-memory operation** with durable on-disk state
- Linked as a library (`libpomai.a`), not a server or service
- No network layer, no RPC, no client-server protocol
- Application directly calls API functions

#### 2. Durability-First Architecture
- **Write-Ahead Log (WAL) is the Single Source of Truth (SSOT)**
- Every mutation appends to WAL before applying to in-memory structures
- MemTables and indexes are **derived caches** that can be rebuilt from WAL
- Crash recovery replays WAL to reconstruct state
- WAL prefix durability: acknowledged writes survive crashes (subject to fsync policy)

#### 3. Snapshot Isolation & Visibility
- **Bounded staleness** model: writes become visible only after Freeze (soft or explicit)
- Active MemTable is NOT visible to readers
- Only Frozen MemTables + Segments are visible in snapshot
- Readers NEVER block on writers
- **No read-your-writes** until soft freeze (5000 items/shard) or explicit `Freeze()` call

#### 4. Actor-Model Single-Writer Per Shard
- Each shard has **exactly one writer thread** (event loop)
- Writes serialized via bounded MPSC mailbox
- No lock contention on mutation path
- No shared-state mutation by readers
- Deterministic, auditable write ordering per shard

#### 5. Lock-Free Readers
- Readers load atomic snapshot pointer
- Read from immutable frozen structures only
- Zero locks on read path
- Linear-ish scaling for search throughput across CPU cores

#### 6. Storage Lifecycle: Active → Frozen → Segment
- **Active MemTable**: Mutable, single-writer, not visible to readers
- **Frozen MemTables**: Immutable, visible in snapshot, awaiting segment flush
- **Segments**: Immutable on-disk files, visible in snapshot
- State transitions are one-way and irreversible

#### 7. Crash Recovery Guarantees
- WAL replay rebuilds MemTable state
- Truncated WAL tail is tolerated (prefix durability)
- Replayed data rotated to frozen for immediate visibility on startup
- No partial visibility of uncommitted transactions

---

## What PomaiDB IS NOT

### Explicitly Rejected Identities

PomaiDB is **fundamentally different** from the following systems and will **never** converge toward them:

#### NOT a FAISS-like ANN Engine
- PomaiDB is not an Approximate Nearest Neighbor (ANN) library
- Core value is **database correctness**, not ANN recall/latency races
- ANN indexes (HNSW/IVF) are **optional accelerators**, not the primary identity
- If ANN were removed, PomaiDB would still be valuable as a data engine

#### NOT a Search-Engine-First System
- Not optimized for search recall tuning (ef, M, nprobe parameters)
- Not a SIMD/kernel-centric system
- Not a brute-force search demo
- Search is a **query modality**, not the reason PomaiDB exists

#### NOT a Distributed Database
- No replication, no consensus, no cluster coordination
- No multi-tenancy, no sharding across machines
- Shard = in-process failure domain, not network partition

#### NOT a Tunable Recall Playground
- Does not expose HNSW/IVF tuning knobs as core API surface
- Does not compete on ANN benchmark leaderboards
- Correctness and durability trump recall optimization

---

## Hard Guarantees (MUST Preserve & Strengthen)

### 1. WAL Prefix Durability
- **Guarantee**: A write is acknowledged only after WAL append succeeds.
- **Recovery**: Crash recovery replays WAL into MemTable.
- **Tolerance**: Truncated WAL tail (incomplete records) is tolerated and skipped.
- **Fsync Policy**: Durability strength controlled by `FsyncPolicy` (`kAlways`, `kOnFlush`, `kNever`).

### 2. Snapshot Isolation (Bounded Staleness)
- **Guarantee**: Active MemTable is NOT visible to readers.
- **Visibility Boundary**: Only Frozen MemTables + Segments are in snapshot.
- **Monotonicity**: Snapshot versions increase monotonically.
- **Consistency**: Reads observe a single immutable snapshot, never mixed states.

### 3. No Read-Your-Writes Before Freeze
- **Guarantee**: Writes are invisible until Active MemTable rotates to Frozen.
- **Soft Freeze Trigger**: Automatic rotation at 5000 items per shard.
- **Explicit Freeze**: User calls `Freeze()` to force visibility.
- **Correctness**: Staleness is bounded but guaranteed.

### 4. Actor-Model Writes
- **Guarantee**: Exactly one writer thread per shard.
- **Serialization**: Writes serialized via mailbox (MPSC queue).
- **Ordering**: Per-shard write order preserved, no cross-shard ordering.
- **Isolation**: No shared-state mutation by readers.

### 5. Immutable Storage States
- **Frozen MemTables**: Immutable after rotation, count fixed, no further mutation.
- **Segments**: Immutable on-disk files, content never changes after creation.
- **Snapshot**: Stable view of immutable structures, no mid-read mutations.

### 6. Crash Safety & Atomicity
- **Manifest Atomicity**: Segment list updates via atomic rename + directory fsync.
- **Freeze Atomicity**: Either old snapshot or new snapshot is visible post-crash, never partial.
- **WAL Reset**: WAL reset only after successful segment flush + manifest commit.

---

## Non-Goals (Will NOT Implement)

### 1. ANN Recall Optimization as Primary Goal
- PomaiDB does not tune HNSW `ef` or IVF `nprobe` as core features.
- ANN indexes may exist as **secondary accelerators** but are not the identity.
- Any ANN logic must include exact rerank for correctness.

### 2. SIMD/Kernel Tuning
- Low-level SIMD optimization is not the focus.
- Correctness and storage engine quality come first.
- SIMD may be used for distance computation but not as architectural focus.

### 3. Query Planners / SQL Layer
- PomaiDB is not a general-purpose SQL database.
- No query optimizer, no relational algebra, no JOINs.

### 4. Distributed Consensus / Replication
- No Raft, no Paxos, no cross-machine coordination.
- Embedded, single-process deployment only.

### 5. Multi-Tenancy / Cloud Management
- No tenant isolation, no quota enforcement, no billing.
- Designed for embedded, single-tenant use cases.

---

## Database Semantics Reference

### Operation Guarantees Matrix

| Operation       | Durability                     | Visibility                          | Isolation       | Ordering         |
|-----------------|--------------------------------|-------------------------------------|-----------------|------------------|
| Put/Delete      | WAL appended; durable if fsync | Not visible until Freeze            | Per-shard only  | Mailbox order    |
| Search/Get      | Reads snapshot state           | Snapshot state only                 | Snapshot iso    | Snapshot version |
| Freeze          | Segment + manifest fsync       | Publishes new snapshot atomically   | Shard-local     | Per-shard only   |
| Flush           | WAL fdatasync (if fsync ≠ kNever) | No visibility change             | N/A             | N/A              |
| Crash Recovery  | WAL replay                     | Replayed data visible after startup | Snapshot iso    | WAL order        |

### Visibility Timeline Example

```
T0: Put(id=1) → WAL appended, Active MemTable updated
T1: Get(id=1) → NotFound (Active not in snapshot)
T2: Active count ≥ 5000 → RotateMemTable + PublishSnapshot
T3: Get(id=1) → Found (visible in snapshot)
```

### Crash Recovery Example

```
Crash at T: Active MemTable state lost
Restart:
  1. Replay WAL into MemTable
  2. Rotate replayed MemTable to Frozen (immediately visible)
  3. PublishSnapshot
  4. Readers see recovered state
```

---

## Roadmap Constraints (DB-Centric Evolution)

### Allowed Evolution Directions

1. **Compaction & GC**: Merge segments, drop tombstones, reduce read amplification.
2. **Snapshot Iterators**: Export dataset for ML training, deterministic splits.
3. **Storage Engine Hardening**: Better fsync, better crash tests, format versioning.
4. **Metadata Indexing**: Block-level indexes in segments, better search within segments.

### Forbidden Evolution Directions

1. **FAISS-like Tuning**: Exposing ef, M, nprobe as primary API.
2. **ANN Recall Races**: Competing on ANN benchmark leaderboards.
3. **Search-Engine Features**: Full-text search, fuzzy matching, ranking functions.
4. **Distributed Coordination**: Replication, sharding across machines, consensus.

---

## Enforcement & Review Process

### Pull Request Rejection Criteria

Any PR that does the following **MUST be rejected**:

1. Tunes ANN recall as a primary feature (ef, M, nprobe knobs).
2. Optimizes SIMD kernels as architectural focus.
3. Adds FAISS-like features without DB correctness guarantees.
4. Treats WAL as "just index persistence" instead of SSOT.
5. Breaks snapshot isolation or durability guarantees.
6. Introduces linearizability or read-your-writes without explicit design review.

### Design Review Requirements

Breaking changes require:
1. Design doc/RFC in `docs/`
2. Proof that change strengthens DB identity (not search identity)
3. Migration plan for on-disk format changes
4. Updated crash tests proving correctness

---

## Success Criteria

### PomaiDB succeeds if:

1. **It makes sense without ANN**: Remove ANN indexes entirely, PomaiDB is still valuable.
2. **WAL + Snapshot are central**: Core architecture revolves around durability and visibility.
3. **It cannot be mistaken for FAISS**: Design decisions clearly diverge from search-first systems.
4. **Storage engine quality**: Crash safety, compaction, GC, format versioning are first-class.

### PomaiDB fails if:

1. ANN tuning becomes the primary value proposition.
2. Search engine features dominate the roadmap.
3. Correctness is sacrificed for performance.
4. Durability and snapshot isolation are weakened.

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) — Technical architecture
- [CONSISTENCY_MODEL.md](CONSISTENCY_MODEL.md) — Snapshot isolation semantics
- [FAILURE_SEMANTICS.md](FAILURE_SEMANTICS.md) — Crash-by-crash outcomes
- [ON_DISK_FORMAT.md](ON_DISK_FORMAT.md) — Storage format versioning
- [ROADMAP.md](ROADMAP.md) — Milestone planning

---

**Version**: 1.0  
**Last Updated**: 2026-02-04  
**Maintainers**: PomaiDB Core Team
