# Python API

PomaiDB is exposed to Python via the **C API** and **ctypes**. The official package is **`pomaidb`** (pip-installable from the `python/` directory). You can also use the C library directly with ctypes (see examples in `examples/` and `benchmarks/`).

## Installation

1. Build the C library (from repo root):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target pomai_c
   ```
2. Install the Python package:
   ```bash
   pip install ./python
   ```
3. Set `POMAI_C_LIB` to the path of `libpomai_c.so` (Linux) or `libpomai_c.dylib` (macOS) if the package cannot find it (e.g. `export POMAI_C_LIB=$PWD/build/libpomai_c.so`).

## High-level API (pomaidb package)

| Function        | Description |
|----------------|-------------|
| `open_db(path, dim, **opts)` | Open database at `path` with vector dimension `dim`. Options: `shards`, `search_threads`, `fsync`, `metric` ("ip" or "l2"), `profile` (`edge-low-ram`, `edge-balanced`, `edge-throughput`, `user_defined`), `hnsw_m`, `hnsw_ef_construction`, `hnsw_ef_search`, `adaptive_threshold`, `gateway_rate_limit_per_sec`, `gateway_idempotency_ttl_sec`, `gateway_token_file`, `gateway_upstream_sync_enabled`, `gateway_upstream_sync_url`, `gateway_require_mtls_proxy_header`, `gateway_mtls_proxy_header`. Returns opaque db handle. |
| `resolve_effective_options(path, dim, **opts)` | Return effective runtime options as JSON after profile application (preflight/deploy validation). |
| `membrane_kind_capabilities(kind)` | Return a dict of capability flags for a membrane kind (`MEMBRANE_KIND_*` constants); matches C API `pomai_membrane_kind_capabilities`. No database open required. |
| `put_batch(db, ids, vectors)` | Insert vectors. `ids`: list of int; `vectors`: list of list of float (n × dim). |
| `meta_put(db, membrane_name, gid, value)` | Store metadata payload by global id in a `kMeta` membrane. |
| `meta_get(db, membrane_name, gid)` | Load metadata payload by global id from a `kMeta` membrane. |
| `meta_delete(db, membrane_name, gid)` | Delete metadata payload by global id from a `kMeta` membrane. |
| `create_membrane_kind(db, name, dim, shard_count, kind, ttl_sec=0, retention_max_count=0, retention_max_bytes=0)` | Create and open a membrane with optional retention policy (TTL/count/bytes). |
| `update_membrane_retention(db, membrane_name, ttl_sec=0, retention_max_count=0, retention_max_bytes=0)` | Update retention policy for an existing `kMeta`/`kKeyValue` membrane. |
| `get_membrane_retention(db, membrane_name)` | Read current retention policy for a membrane as a dict. |
| `link_objects(db, gid, vector_id, graph_vertex_id, mesh_id)` | Register multi-modal GID link for vector/graph/mesh objects. |
| `unlink_objects(db, gid)` | Remove a registered multi-modal GID link. |
| `start_edge_gateway(db, http_port=8080, ingest_port=8090, auth_token=None)` | Start embedded HTTP + ingestion listeners (Phase 3). If `auth_token` is set, gateway validates bearer token + scopes (from static token or rotating token file). |
| `stop_edge_gateway(db)` | Stop embedded edge listeners. |
| `search_zero_copy(db, query, topk=10)` | Run single query with zero-copy semantic pointers and return NumPy views/dequantized arrays. |
| `release_zero_copy_session(session_id)` | Release pinned zero-copy session returned by search results. |
| `delete(db, int)` | Delete a vector from the index by global ID. |
| `exists(db, int)` | Return a boolean indicating if a vector with the global ID exists. |
| `get(db, int)` | Return the `id`, `dim`, `vector`, `metadata`, and `is_deleted` values for a given global ID. |

Gateway endpoints/protocols:
- HTTP: `GET /health`, `GET /healthz`, `GET /metrics`, `POST /ingest/meta/<membrane>/<gid>`, `POST /ingest/vector/<membrane>/<id>`
- Versioned aliases are supported: `/v1/health`, `/v1/healthz`, `/v1/metrics`, `/v1/ingest/...`
- HTTP responses are JSON (`{"status":"ok|error|duplicate","message":"..."}`) with proper status codes (`200/400/401/404/429`).
- HTTP error schema includes machine-readable `code` (`auth_scope_denied`, `write_failed`, `invalid_path`, ...).
- HTTP idempotency: send `X-Idempotency-Key: <key>` to make retries safe; repeated keys return `status=duplicate`.
- Idempotency keys are persisted under DB path (`gateway/idempotency.log`) so duplicate suppression survives gateway restart.
- Gateway writes operational audit lines to `gateway/audit.log` (`timestamp|event|detail`) for incident/debug workflows.
- Idempotency log is periodically compacted to keep disk usage bounded.
- Audit log also rotates (`audit.log` -> `audit.log.1`) to bound local disk growth.
- `/metrics` exports gateway durability/ops counters including log compactions, log errors, and audit rotations.
- Optional async upstream sync: configure `gateway_upstream_sync_enabled` + `gateway_upstream_sync_url` for edge->regional forwarding with checkpointed sequence progress.
- Edge profile note: profile presets tune operational limits (memtable/fsync/gateway) but do **not** overwrite user-provided index params.
- Ingest TCP (MQTT/WS-style line protocol):
  - no-auth: `MQTT|<membrane>|<id>|f1,f2,f3`
  - auth: `MQTT|<token>|<membrane>|<id>|f1,f2,f3`
  - optional idempotency + durability: `...|<idem_key>|D`
  - durable ack: `|D` => server flushes and replies only after flush
  - replies: `ACK|accepted|seq=<n>`, `ACK|durable_ack|seq=<n>`, `ACK|duplicate`, or `ERR|...`
  - Retry semantics: duplicate idempotency key => `duplicate`; accepted write => `accepted`; durable flush success => `durable_ack`.
| `freeze(db)`   | Flush memtable to segment and build index. Must be called before new data is visible to search. |
| `search_batch(db, queries, topk=10)` | Batch search. `queries`: list of list of float (n_queries × dim). Returns list of `(ids, scores)` per query. |
| `close(db)`    | Close the database. |
| **RAG** | |
| `create_rag_membrane(db, name, dim, shard_count=1)` | Create and open a RAG membrane for chunk storage and hybrid search. |
| `put_chunk(db, membrane_name, chunk_id, doc_id, token_ids, vector=None)` | Insert a chunk: token IDs (required) and optional embedding vector. |
| `search_rag(db, membrane_name, token_ids=None, vector=None, topk=10, ...)` | RAG search by token overlap and/or vector. Returns list of `(chunk_id, doc_id, score, token_matches)`. |
| **Typed Membranes (Other)** | |
| `ts_put(db, membrane_name, series_id, ts, value)` | Put a timeseries data point. |
| `kv_put(db, membrane_name, key, value)` | Store a KV pair in KEYVALUE membrane. |
| `kv_get(db, membrane_name, key)` | Get the string value for a given key. |
| `kv_delete(db, membrane_name, key)` | Delete a KV pair from a KEYVALUE membrane. |
| `sketch_add(db, membrane_name, key, increment)` | Add value to a SKETCH membrane counter. |
| `blob_put(db, membrane_name, blob_id, data)` | Store a binary blob (`bytes`) in a BLOB membrane. |
| **Agent Memory** | |
| `agent_memory_open(path, dim, metric="l2", max_messages_per_agent, max_device_bytes)` | Open or create an AgentMemory backend at the given path. |
| `agent_memory_close(mem)` | Close an AgentMemory backend. |
| `agent_memory_append(mem, agent_id, session_id, kind, logical_ts, text, embedding=None)` | Append a single agent memory record. |
| `agent_memory_append_batch(mem, records)` | Append multiple agent memory records as a list of dicts. |
| `agent_memory_get_recent(mem, agent_id, session_id=None, limit=10)` | Fetch recent agent memory records. |
| `agent_memory_search(mem, agent_id, session_id=None, kind=None, min_ts=0, max_ts=0, embedding=None, topk=10)` | Semantic search over AgentMemory. |
| `agent_memory_prune_old(mem, agent_id, keep_last_n, min_ts_to_keep)` | Prune old records for an agent. |
| `agent_memory_prune_device(mem, target_total_bytes)` | Prune global device wide records. |
| `agent_memory_freeze_if_needed(mem)` | Flush memory indexes to disk if pending. |

Exceptions: `pomaidb.PomaiDBError` on any failing call.

### Example

```python
import pomaidb

db = pomaidb.open_db("/tmp/vec_db", dim=128, shards=1, metric="ip")
pomaidb.put_batch(db, ids=[1, 2], vectors=[[0.1] * 128, [0.2] * 128])
pomaidb.freeze(db)
for ids, scores in pomaidb.search_batch(db, [[0.15] * 128], topk=5):
    print(ids, scores)
pomaidb.close(db)
```

## C API (ctypes)

The shared library exposes:

- `pomai_options_init(opts)` — initialize options struct
- `pomai_open(opts, &db)` — open DB; returns status (null = ok)
- `pomai_put_batch(db, upserts, n)` — batch insert
- `pomai_meta_put(db, membrane_name, gid, value)` / `pomai_meta_get(...)` / `pomai_meta_delete(...)` — metadata membrane CRUD by GID
- `pomai_create_membrane_kind_with_retention(db, name, dim, shard_count, kind, ttl_sec, retention_max_count, retention_max_bytes)` — create typed membrane with lifecycle retention limits
- `pomai_update_membrane_retention(db, membrane_name, ttl_sec, retention_max_count, retention_max_bytes)` — update retention policy without recreating membrane
- `pomai_get_membrane_retention_json(db, membrane_name, &out_json, &out_len)` — read current retention policy (JSON)
- `pomai_link_objects(db, gid, vector_id, graph_vertex_id, mesh_id)` / `pomai_unlink_objects(...)` — multi-modal linker APIs
- `pomai_edge_gateway_start(db, http_port, ingest_port)` / `pomai_edge_gateway_start_secure(db, http_port, ingest_port, auth_token)` / `pomai_edge_gateway_stop(db)` — edge connectivity lifecycle
- `pomai_freeze(db)` — flush and build index
- `pomai_search_batch(db, queries, n, &out)` — batch search; `out` is array of `PomaiSearchResults`
- `pomai_search_batch_free(out, n)` — free search results
- `pomai_close(db)` — close DB
- **RAG:** `pomai_create_rag_membrane(db, name, dim, shard_count)`, `pomai_put_chunk(db, membrane_name, chunk)`, `pomai_search_rag(db, membrane_name, query, opts, result)`, `pomai_rag_search_result_free(result)`
- `pomai_status_message(status)` / `pomai_status_free(status)` — error message

Struct layouts and constants are in `include/pomai/c_types.h`. The `pomaidb` package registers these for you. RAG quick start: `examples/rag_quickstart.py`.

## Versioning

The Python package follows the same version as the project (see [VERSIONING.md](VERSIONING.md)). Compatibility is maintained with the C ABI within a MAJOR version.
