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
} pomai_query_t;

typedef struct {
    uint32_t struct_size;
    size_t count;
    uint64_t* ids;
    float* scores;
    uint32_t* shard_ids;
} pomai_search_results_t;

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
