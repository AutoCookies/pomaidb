#ifndef POMAI_C_TYPES_H
#define POMAI_C_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Opaque handles
typedef struct pomai_db_t pomai_db_t;
typedef struct pomai_snapshot_t pomai_snapshot_t;
typedef struct pomai_iter_t pomai_iter_t;

// Enums matching C++ Options
typedef enum {
    POMAI_FSYNC_NEVER = 0,
    POMAI_FSYNC_ALWAYS = 1,
} pomai_fsync_policy_t;

typedef enum {
    POMAI_METRIC_L2 = 0,
    POMAI_METRIC_IP = 1,
    POMAI_METRIC_COSINE = 2,
} pomai_metric_type_t;

// Structs
typedef struct {
    uint32_t nlist;
    uint32_t nprobe;
} pomai_index_params_t;

typedef struct {
    const char* path;
    uint32_t shard_count;
    uint32_t dim;
    pomai_fsync_policy_t fsync;
    pomai_index_params_t index_params;
    uint64_t memory_budget_bytes;
} pomai_options_t;

typedef struct {
    uint64_t id;
    const float* vector;     // Pointer to data owned by caller
    uint32_t dim;
    const uint8_t* metadata; // Optional metadata blob
    uint32_t metadata_len;
} pomai_upsert_t;

typedef struct {
    uint64_t id;
    uint32_t dim;
    const float* vector;        // Owned by Pomai
    const uint8_t* metadata;    // Owned by Pomai
    uint32_t metadata_len;
    bool is_deleted;
} pomai_record_t;

typedef struct {
    const float* vector;
    uint32_t dim;
    uint32_t topk;
    const char* filter_expr; // Optional filter string
} pomai_query_t;

typedef struct {
    uint64_t id;
    float score;
} pomai_search_hit_t;

typedef struct {
    uint32_t shard_id;
    const char* message;
} pomai_shard_error_t;

typedef struct {
    size_t count;
    pomai_search_hit_t* hits;     // Array of hits
    
    size_t error_count;
    pomai_shard_error_t* errors; // Array of errors (if any)
} pomai_search_results_t;

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_TYPES_H
