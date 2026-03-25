#ifndef POMAI_C_TYPES_H
#define POMAI_C_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Cross-platform symbol visibility for the C ABI.
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(POMAI_C_BUILD_DLL)
#    define POMAI_API __declspec(dllexport)
#  elif defined(POMAI_C_USE_DLL)
#    define POMAI_API __declspec(dllimport)
#  else
#    define POMAI_API
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define POMAI_API __attribute__((visibility("default")))
#else
#  define POMAI_API
#endif

// Opaque handles
typedef struct pomai_db_t pomai_db_t;
typedef struct pomai_rag_pipeline_t pomai_rag_pipeline_t;
typedef struct pomai_snapshot_t pomai_snapshot_t;
typedef struct pomai_iter_t pomai_iter_t;
typedef struct pomai_membrane_iter_t pomai_membrane_iter_t;
typedef struct pomai_txn_t pomai_txn_t;
typedef struct pomai_status_t pomai_status_t;
typedef struct pomai_agent_memory_t pomai_agent_memory_t;

#define POMAI_QUERY_FLAG_ZERO_COPY 1

typedef enum {
    POMAI_FSYNC_POLICY_NEVER = 0,
    POMAI_FSYNC_POLICY_ALWAYS = 1,
} pomai_fsync_policy_t;

typedef struct {
    uint32_t struct_size;
    const char* path;
    uint32_t shards;
    uint32_t dim;
    uint32_t search_threads;
    pomai_fsync_policy_t fsync_policy;
    uint64_t memory_budget_bytes;
    uint32_t deadline_ms;
    
    // Indexing
    uint8_t index_type; // 0 = IVF, 1 = HNSW
    uint32_t hnsw_m;
    uint32_t hnsw_ef_construction;
    uint32_t hnsw_ef_search;
    uint32_t adaptive_threshold;
    uint8_t metric; // 0 = L2, 1 = IP
    uint8_t edge_profile; // 0=user_defined, 1=low_ram, 2=balanced, 3=throughput
    uint32_t gateway_rate_limit_per_sec;
    uint32_t gateway_idempotency_ttl_sec;
    const char* gateway_token_file;
    const char* gateway_upstream_sync_url;
    bool gateway_upstream_sync_enabled;
    bool gateway_require_mtls_proxy_header;
    const char* gateway_mtls_proxy_header;
} pomai_options_t;

typedef struct {
    uint32_t struct_size;
    uint64_t id;
    const float* vector;
    uint32_t dim;
    const uint8_t* metadata;
    uint32_t metadata_len;
} pomai_upsert_t;

typedef struct {
    uint32_t struct_size;
    uint64_t id;
    uint32_t dim;
    const float* vector;
    const uint8_t* metadata;
    uint32_t metadata_len;
    bool is_deleted;
} pomai_record_t;

// Current-row record view for iterators.
// Pointers are valid only until pomai_iter_next() or pomai_iter_free().
typedef struct {
    uint32_t struct_size;
    uint64_t id;
    uint32_t dim;
    const float* vector;
    const uint8_t* metadata;
    uint32_t metadata_len;
    bool is_deleted;
} pomai_record_view_t;

typedef struct {
    uint32_t struct_size;
    const float* vector;
    uint32_t dim;
    uint32_t topk;
    const char* filter_expression;
    const char* partition_device_id;
    const char* partition_location_id;
    uint64_t as_of_ts;
    uint64_t as_of_lsn;
    uint32_t aggregate_op;     // 0:none 1:sum 2:avg 3:min 4:max 5:count 6:topk
    const char* aggregate_field; // score|timestamp|lsn
    uint32_t aggregate_topk;
    uint32_t mesh_detail_preference; // 0:auto-latency 1:high-detail
    float alpha;
    uint32_t deadline_ms;
    uint32_t flags;
} pomai_query_t;

typedef struct {
    uint32_t struct_size;
    const void* raw_data_ptr;
    uint32_t dim;
    float quant_min;
    float quant_inv_scale;
    uint64_t session_id;
} pomai_semantic_pointer_t;

typedef struct {
    uint32_t struct_size;
    size_t count;
    uint64_t* ids;
    float* scores;
    uint32_t* shard_ids;
    uint32_t total_shards_count;
    uint32_t pruned_shards_count;
    double aggregate_value;
    uint32_t aggregate_op;
    uint32_t mesh_lod_level; // reserved for mesh result paths
    pomai_semantic_pointer_t* zero_copy_pointers;
} pomai_search_results_t;

POMAI_API pomai_status_t* pomai_search_batch(
    pomai_db_t* db, const pomai_query_t* queries, size_t num_queries,
    pomai_search_results_t** out_results);

POMAI_API void pomai_search_batch_free(pomai_search_results_t* results, size_t num_queries);

// RAG pipeline chunk options (for pomai_rag_pipeline_create)
typedef struct {
    uint32_t struct_size;
    size_t max_chunk_bytes;
    size_t max_doc_bytes;
    size_t max_chunks_per_batch;
    size_t overlap_bytes;
} pomai_rag_chunk_options_t;

// RAG: chunk and query types
typedef struct {
    uint32_t struct_size;
    uint64_t chunk_id;
    uint64_t doc_id;
    const uint32_t* token_ids;
    size_t token_count;
    const float* vector;
    uint32_t dim;  // 0 if no vector
    const char* chunk_text;   // optional; stored for RetrieveContext (NULL or empty = not stored)
    size_t chunk_text_len;
} pomai_rag_chunk_t;

typedef struct {
    uint32_t struct_size;
    const uint32_t* token_ids;
    size_t token_count;
    const float* vector;
    uint32_t dim;   // 0 if no vector
    uint32_t topk;
} pomai_rag_query_t;

typedef struct {
    uint32_t struct_size;
    uint32_t candidate_budget;
    uint32_t token_budget;           // 0 = no limit
    bool enable_vector_rerank;
} pomai_rag_search_options_t;

typedef struct {
    uint64_t chunk_id;
    uint64_t doc_id;
    float score;
    uint32_t token_matches;
    char* chunk_text;   // optional; from stored chunk text (caller must pomai_free)
    size_t chunk_text_len;
} pomai_rag_hit_t;

typedef struct {
    size_t hit_count;
    pomai_rag_hit_t* hits;
} pomai_rag_search_result_t;

typedef struct {
    uint32_t struct_size;
    uint64_t start_id;
    bool has_start_id;
    uint32_t deadline_ms;
} pomai_scan_options_t;

/** Options for pomai_membrane_scan (unified export across membrane kinds). */
typedef struct {
    uint32_t struct_size;
    uint64_t max_records;
    uint64_t max_materialized_keys;
    uint32_t deadline_ms;
    size_t max_field_bytes;
} pomai_membrane_scan_options_t;

/**
 * View of current membrane scan row. Pointers are owned by the iterator until
 * pomai_membrane_iter_next() or pomai_membrane_iter_free().
 */
typedef struct {
    uint32_t struct_size;
    uint8_t membrane_kind;
    uint64_t id;
    const char* key;
    size_t key_len;
    const char* value;
    size_t value_len;
    const float* vector;
    uint32_t vector_dim;
} pomai_membrane_record_view_t;

/** Values align with pomai::MembraneKind (include/pomai/options.h). */
#define POMAI_MEMBRANE_KIND_VECTOR 0
#define POMAI_MEMBRANE_KIND_RAG 1
#define POMAI_MEMBRANE_KIND_GRAPH 2
#define POMAI_MEMBRANE_KIND_TEXT 3
#define POMAI_MEMBRANE_KIND_TIMESERIES 4
#define POMAI_MEMBRANE_KIND_KEYVALUE 5
#define POMAI_MEMBRANE_KIND_SKETCH 6
#define POMAI_MEMBRANE_KIND_BLOB 7
#define POMAI_MEMBRANE_KIND_SPATIAL 8
#define POMAI_MEMBRANE_KIND_MESH 9
#define POMAI_MEMBRANE_KIND_SPARSE 10
#define POMAI_MEMBRANE_KIND_BITSET 11
#define POMAI_MEMBRANE_KIND_META 12

#define POMAI_MEMBRANE_STABILITY_STABLE 0
#define POMAI_MEMBRANE_STABILITY_EXPERIMENTAL 1

typedef struct {
    uint32_t struct_size;
    uint8_t kind;
    uint8_t stability; /**< POMAI_MEMBRANE_STABILITY_* */
    uint8_t reserved0;
    uint8_t reserved1;
    bool read_path;
    bool write_path;
    bool unified_scan;
    bool snapshot_isolated_scan;
} pomai_membrane_capabilities_t;

// AgentMemory C API types

typedef struct {
    uint32_t struct_size;
    const char* path;
    uint32_t dim;
    uint8_t metric; // 0 = L2, 1 = InnerProduct, 2 = Cosine
    uint32_t max_messages_per_agent;
    uint64_t max_device_bytes;
} pomai_agent_memory_options_t;

typedef struct {
    uint32_t struct_size;
    const char* agent_id;
    const char* session_id;
    const char* kind;        // "message" | "summary" | "knowledge"
    int64_t logical_ts;
    const char* text;
    const float* embedding;
    uint32_t dim;
} pomai_agent_memory_record_t;

typedef struct {
    uint32_t struct_size;
    const char* agent_id;
    const char* session_id;  // NULL or empty for "any session"
    const char* kind;        // NULL or empty for "any kind"
    int64_t min_ts;
    int64_t max_ts;
    const float* embedding;
    uint32_t dim;
    uint32_t topk;
} pomai_agent_memory_query_t;

typedef struct {
    uint32_t struct_size;
    size_t count;
    pomai_agent_memory_record_t* records;
} pomai_agent_memory_result_set_t;

typedef struct {
    uint32_t struct_size;
    size_t count;
    pomai_agent_memory_record_t* records;
    float* scores;
} pomai_agent_memory_search_result_t;

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_TYPES_H
