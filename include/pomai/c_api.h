#ifndef POMAI_C_API_H
#define POMAI_C_API_H

#include "pomai/c_types.h"
#include "pomai/c_status.h"
#include "pomai/c_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options
POMAI_API void pomai_options_init(pomai_options_t* opts);
POMAI_API void pomai_scan_options_init(pomai_scan_options_t* opts);
POMAI_API void pomai_membrane_scan_options_init(pomai_membrane_scan_options_t* opts);
/** Zero-initialize pomai_membrane_capabilities_t and set struct_size (forward-compatible). */
POMAI_API void pomai_membrane_capabilities_init(pomai_membrane_capabilities_t* caps);
/** Fill capability flags for a membrane kind (0 … POMAI_MEMBRANE_KIND_META). */
POMAI_API pomai_status_t* pomai_membrane_kind_capabilities(uint8_t kind, pomai_membrane_capabilities_t* out_caps);
// Resolve effective runtime options after applying edge profile.
POMAI_API pomai_status_t* pomai_options_resolve_json(const pomai_options_t* opts, char** out_json, size_t* out_len);

// Database Management
POMAI_API pomai_status_t* pomai_open(const pomai_options_t* opts, pomai_db_t** out_db);
POMAI_API pomai_status_t* pomai_close(pomai_db_t* db);
POMAI_API pomai_status_t* pomai_freeze(pomai_db_t* db);

// Mutation
POMAI_API pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item);
POMAI_API pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n);
POMAI_API pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id);

// Point Query
POMAI_API pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists);
POMAI_API pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record);
POMAI_API void pomai_record_free(pomai_record_t* record);

// Search
POMAI_API pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out);
POMAI_API void pomai_search_results_free(pomai_search_results_t* results);

// Typed membranes (TimeSeries/KeyValue/Sketch/Blob)
POMAI_API pomai_status_t* pomai_create_membrane_kind(pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count, uint32_t kind);
POMAI_API pomai_status_t* pomai_create_membrane_kind_with_retention(
    pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count, uint32_t kind,
    uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes);
POMAI_API pomai_status_t* pomai_ts_put(pomai_db_t* db, const char* membrane_name, uint64_t series_id, uint64_t ts, double value);
POMAI_API pomai_status_t* pomai_kv_put(pomai_db_t* db, const char* membrane_name, const char* key, const char* value);
POMAI_API pomai_status_t* pomai_kv_get(pomai_db_t* db, const char* membrane_name, const char* key, char** out_value, size_t* out_len);
POMAI_API pomai_status_t* pomai_kv_delete(pomai_db_t* db, const char* membrane_name, const char* key);
POMAI_API pomai_status_t* pomai_meta_put(pomai_db_t* db, const char* membrane_name, const char* gid, const char* value);
POMAI_API pomai_status_t* pomai_meta_get(pomai_db_t* db, const char* membrane_name, const char* gid, char** out_value, size_t* out_len);
POMAI_API pomai_status_t* pomai_meta_delete(pomai_db_t* db, const char* membrane_name, const char* gid);
POMAI_API pomai_status_t* pomai_link_objects(pomai_db_t* db, const char* gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id);
POMAI_API pomai_status_t* pomai_unlink_objects(pomai_db_t* db, const char* gid);
POMAI_API pomai_status_t* pomai_edge_gateway_start(pomai_db_t* db, uint16_t http_port, uint16_t ingest_port);
POMAI_API pomai_status_t* pomai_edge_gateway_start_secure(pomai_db_t* db, uint16_t http_port, uint16_t ingest_port, const char* auth_token);
POMAI_API pomai_status_t* pomai_edge_gateway_stop(pomai_db_t* db);
POMAI_API pomai_status_t* pomai_list_membranes_json(pomai_db_t* db, char** out_json, size_t* out_len);
POMAI_API pomai_status_t* pomai_compact_membrane(pomai_db_t* db, const char* membrane_name);
POMAI_API pomai_status_t* pomai_update_membrane_retention(
    pomai_db_t* db, const char* membrane_name,
    uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes);
POMAI_API pomai_status_t* pomai_get_membrane_retention_json(
    pomai_db_t* db, const char* membrane_name, char** out_json, size_t* out_len);
POMAI_API pomai_status_t* pomai_sketch_add(pomai_db_t* db, const char* membrane_name, const char* key, uint64_t increment);
POMAI_API pomai_status_t* pomai_blob_put(pomai_db_t* db, const char* membrane_name, uint64_t blob_id, const uint8_t* data, size_t len);

// RAG membrane and search
POMAI_API pomai_status_t* pomai_create_rag_membrane(pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count);
POMAI_API pomai_status_t* pomai_put_chunk(pomai_db_t* db, const char* membrane_name, const pomai_rag_chunk_t* chunk);
POMAI_API pomai_status_t* pomai_search_rag(pomai_db_t* db, const char* membrane_name, const pomai_rag_query_t* query,
                                           const pomai_rag_search_options_t* opts, pomai_rag_search_result_t* out_result);
POMAI_API void pomai_rag_search_result_free(pomai_rag_search_result_t* result);

// RAG pipeline (ingest document + retrieve context; uses built-in mock embedder when no external provider)
POMAI_API pomai_status_t* pomai_rag_pipeline_create(pomai_db_t* db, const char* membrane_name, uint32_t embedding_dim,
    const pomai_rag_chunk_options_t* chunk_options, pomai_rag_pipeline_t** out_pipeline);
POMAI_API pomai_status_t* pomai_rag_ingest_document(pomai_rag_pipeline_t* pipeline, uint64_t doc_id,
    const char* text_buf, size_t text_len);
POMAI_API pomai_status_t* pomai_rag_retrieve_context(pomai_rag_pipeline_t* pipeline, const char* query_buf, size_t query_len,
    uint32_t top_k, char** out_buf, size_t* out_len);
/** Same but copy into caller-provided buffer (no alloc; use when cross-DLL free is problematic). */
POMAI_API pomai_status_t* pomai_rag_retrieve_context_buf(pomai_rag_pipeline_t* pipeline, const char* query_buf, size_t query_len,
    uint32_t top_k, char* out_buf, size_t max_len, size_t* out_len);
POMAI_API void pomai_rag_pipeline_free(pomai_rag_pipeline_t* pipeline);

// Snapshot & Scan
POMAI_API pomai_status_t* pomai_get_snapshot(pomai_db_t* db, pomai_snapshot_t** out_snap);
POMAI_API void pomai_snapshot_free(pomai_snapshot_t* snap);

POMAI_API pomai_status_t* pomai_scan(
    pomai_db_t* db,
    const pomai_scan_options_t* opts,
    const pomai_snapshot_t* snap,
    pomai_iter_t** out_iter);

POMAI_API bool pomai_iter_valid(const pomai_iter_t* iter);
POMAI_API void pomai_iter_next(pomai_iter_t* iter);
POMAI_API pomai_status_t* pomai_iter_status(const pomai_iter_t* iter);
POMAI_API pomai_status_t* pomai_iter_get_record(const pomai_iter_t* iter, pomai_record_view_t* out_view);
POMAI_API void pomai_iter_free(pomai_iter_t* iter);

/** Unified membrane full-scan (any MembraneKind; see membrane_iterator.h). */
POMAI_API pomai_status_t* pomai_membrane_scan(pomai_db_t* db, const char* membrane_name,
                                            const pomai_membrane_scan_options_t* opts,
                                            pomai_membrane_iter_t** out_iter);
POMAI_API bool pomai_membrane_iter_valid(const pomai_membrane_iter_t* iter);
POMAI_API void pomai_membrane_iter_next(pomai_membrane_iter_t* iter);
POMAI_API pomai_status_t* pomai_membrane_iter_status(const pomai_membrane_iter_t* iter);
POMAI_API bool pomai_membrane_iter_truncated(const pomai_membrane_iter_t* iter);
/** Fills out_view with pointers into iterator-owned buffers (valid until next get_record/next/free). */
POMAI_API pomai_status_t* pomai_membrane_iter_get_record(pomai_membrane_iter_t* iter,
                                                       pomai_membrane_record_view_t* out_view);
POMAI_API void pomai_membrane_iter_free(pomai_membrane_iter_t* iter);

// Utils
/** Release a zero-copy semantic pointer session (see POMAI_QUERY_FLAG_ZERO_COPY). */
POMAI_API void pomai_release_pointer(uint64_t session_id);
POMAI_API void pomai_free(void* p);

// AgentMemory: high-level context/memory backend for local agents

POMAI_API pomai_status_t* pomai_agent_memory_open(
    const pomai_agent_memory_options_t* opts,
    pomai_agent_memory_t** out_mem);

POMAI_API pomai_status_t* pomai_agent_memory_close(
    pomai_agent_memory_t* mem);

POMAI_API pomai_status_t* pomai_agent_memory_append(
    pomai_agent_memory_t* mem,
    const pomai_agent_memory_record_t* record,
    uint64_t* out_id);

POMAI_API pomai_status_t* pomai_agent_memory_append_batch(
    pomai_agent_memory_t* mem,
    const pomai_agent_memory_record_t* records,
    size_t n,
    uint64_t* out_ids);

POMAI_API pomai_status_t* pomai_agent_memory_get_recent(
    pomai_agent_memory_t* mem,
    const char* agent_id,
    const char* session_id,
    size_t limit,
    pomai_agent_memory_result_set_t** out);

POMAI_API void pomai_agent_memory_result_set_free(
    pomai_agent_memory_result_set_t* result);

POMAI_API pomai_status_t* pomai_agent_memory_search(
    pomai_agent_memory_t* mem,
    const pomai_agent_memory_query_t* query,
    pomai_agent_memory_search_result_t** out);

POMAI_API void pomai_agent_memory_search_result_free(
    pomai_agent_memory_search_result_t* result);

POMAI_API pomai_status_t* pomai_agent_memory_prune_old(
    pomai_agent_memory_t* mem,
    const char* agent_id,
    size_t keep_last_n,
    int64_t min_ts_to_keep);

POMAI_API pomai_status_t* pomai_agent_memory_prune_device(
    pomai_agent_memory_t* mem,
    uint64_t target_total_bytes);

POMAI_API pomai_status_t* pomai_agent_memory_freeze_if_needed(
    pomai_agent_memory_t* mem);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_API_H
