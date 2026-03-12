// palloc_page_pool.h
// Page-based buffer pool / cache built on top of the palloc_compat interface.
// Exposes a C-friendly API used by PomaiDB core components to fetch/pin/unpin
// fixed-size pages under a hard memory budget, with optional swap-to-disk.
//
// This header is intentionally minimal and C-compatible so it can be used from
// both C and C++ translation units.

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for a page pool.
typedef struct palloc_page_pool palloc_page_pool;

// Device profile for adaptive page sizing. Callers may choose an explicit
// page size instead by setting page_size > 0 in the options.
typedef enum palloc_device_profile_e {
  PALLOC_DEVICE_PROFILE_SMALL = 0,   // e.g. <= 64 MiB RAM
  PALLOC_DEVICE_PROFILE_MEDIUM = 1,  // e.g. ~64–128 MiB RAM
  PALLOC_DEVICE_PROFILE_LARGE = 2    // e.g. > 128 MiB RAM / desktop
} palloc_device_profile_t;

typedef struct palloc_page_pool_options_s {
  // Desired page size in bytes. If zero, an adaptive default is chosen
  // based on device_profile.
  size_t page_size;

  // Hard cap on total resident bytes for pages in memory. If zero, a
  // reasonable default derived from device_profile is used.
  size_t capacity_bytes;

  // Optional path to a dedicated swap file used for evicted dirty pages.
  // May be NULL to disable swapping (eviction will fail once memory is full).
  const char* swap_file_path;

  // Device profile hint for automatic tuning of page size and capacity.
  palloc_device_profile_t device_profile;
} palloc_page_pool_options;

// Statistics for observability and tuning.
typedef struct palloc_page_pool_stats_s {
  size_t page_size;
  size_t capacity_bytes;
  size_t resident_pages;
  size_t dirty_pages;
  size_t bytes_in_swap;
  size_t evictions;
} palloc_page_pool_stats;

// Create / destroy a page pool. Returns NULL on allocation or initialization
// failure.
palloc_page_pool* palloc_page_pool_create(const palloc_page_pool_options* opts);
void palloc_page_pool_destroy(palloc_page_pool* pool);

// Fetch (pin) a page by logical page_id. Returns a pointer to the page-sized
// buffer on success, or NULL on failure (e.g., capacity exceeded and no
// evictable pages, or I/O error reading from swap).
//
// - for_write: non-zero if the caller intends to modify the page.
// - is_new (optional): set to non-zero if the page was created freshly and
//   did not previously exist on disk or in memory.
void* palloc_fetch_page(palloc_page_pool* pool, uint64_t page_id,
                        int for_write, int* is_new);

// Unpin a page previously fetched. If mark_dirty_if_modified is non-zero,
// the page is marked dirty and will be written to the swap file before
// eviction. It is safe to call Unpin multiple times as long as it matches
// the number of successful FetchPage calls that returned the page to the
// caller.
void palloc_unpin_page(palloc_page_pool* pool, uint64_t page_id,
                       int mark_dirty_if_modified);

// Flush a single page or all pages. Returns 0 on success, non-zero on error.
int palloc_flush_page(palloc_page_pool* pool, uint64_t page_id);
int palloc_flush_all(palloc_page_pool* pool);

// Retrieve current statistics for observability and tuning.
void palloc_page_pool_get_stats(palloc_page_pool* pool,
                                palloc_page_pool_stats* out_stats);

// Obtain a process-global default page pool suitable for general-purpose
// usage inside PomaiDB components that do not require custom tuning.
// The swap_file_path_hint (may be NULL) is only used the first time this
// function creates the pool; subsequent calls ignore it and return the
// already-initialized instance.
palloc_page_pool* palloc_get_default_page_pool(const char* swap_file_path_hint);

#ifdef __cplusplus
}  // extern "C"
#endif

