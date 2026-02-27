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
typedef struct pomai_snapshot_t pomai_snapshot_t;
typedef struct pomai_iter_t pomai_iter_t;
typedef struct pomai_txn_t pomai_txn_t;
typedef struct pomai_status_t pomai_status_t;

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
    pomai_semantic_pointer_t* zero_copy_pointers;
} pomai_search_results_t;

POMAI_API pomai_status_t* pomai_search_batch(
    pomai_db_t* db, const pomai_query_t* queries, size_t num_queries,
    pomai_search_results_t** out_results);

POMAI_API void pomai_search_batch_free(pomai_search_results_t* results, size_t num_queries);

typedef struct {
    uint32_t struct_size;
    uint64_t start_id;
    bool has_start_id;
    uint32_t deadline_ms;
} pomai_scan_options_t;

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_TYPES_H
