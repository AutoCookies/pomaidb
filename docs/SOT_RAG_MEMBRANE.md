# SOT: RAG Membrane Extension

This document is the source of truth for the VECTOR vs RAG membrane contract.

## 0.1 Membrane Model (Before / After)

### Before
**MembraneSpec fields (current):**
- `name`
- `shard_count`
- `dim`
- `metric`
- `index_params`

Membranes are vector-only. Payloads are strictly `VectorId + vector<float>[dim] + metadata`.

### After
**Proposed extension:**
```cpp
enum class MembraneKind { VECTOR, RAG };
MembraneKind kind;
```
(`MembraneKind::kVector` / `MembraneKind::kRag` in code.)

**Persistence impact (manifest / recovery):**
- Membrane manifests persist `kind` as a mandatory field.
- Load path accepts legacy manifests that omit `kind` and defaults to `VECTOR`.
- Recovery and restart read `kind` and instantiate the correct engine per membrane.

## 0.2 Payload Contracts (HARD)

### VECTOR membrane record
- `vector<float>[dim]` **(required)**
- `metadata` **(optional)**

### RAG membrane record
- `chunk_id` **(required)**
- `doc_id` **(required)**
- `token_blob` **(REQUIRED)**
- `token_offsets` **(optional)**
- `embedding_vector` **(optional)**
- `metadata` **(optional)**

**Hard invariant:**  
A RAG membrane record **WITHOUT** `token_blob` is **INVALID**, even if an embedding vector is present.

## 0.3 API Surface Changes

**Modified / added APIs (public):**
- `PutVector(...)`, `PutChunk(...)` added to the DB interface. (`include/pomai/pomai.h`)
- `SearchVector(...)`, `SearchRag(...)` added to separate query paths. (`include/pomai/pomai.h`)
- `RagChunk`, `RagQuery`, `RagSearchResult` added. (`include/pomai/rag.h`)
- `MembraneSpec::kind` added and persisted. (`include/pomai/options.h`, `src/storage/manifest/manifest.cc`)

**Runtime enforcement (entry points):**
- Engine and shard runtime reject wrong-kind vector writes. (`src/core/engine/engine.cc`, `src/core/shard/runtime.cc`)
- Membrane manager validates and routes by kind. (`src/core/membrane/manager.cc`)

**Forbidden combinations (hard errors):**
- `PutVector` on RAG membranes → `StatusCode::kInvalidArgument`.
- `PutChunk` on VECTOR membranes → `StatusCode::kInvalidArgument`.
- RAG chunk without `token_blob` → `StatusCode::kInvalidArgument`.
