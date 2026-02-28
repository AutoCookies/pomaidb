/* ----------------------------------------------------------------------------
Copyright (c) 2018-2026, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_H
#define PALLOC_H

#define PALLOC_MALLOC_VERSION 227  // major + 2 digits minor

// ------------------------------------------------------
// Compiler specific attributes
// ------------------------------------------------------

#ifdef __cplusplus
  #if (__cplusplus >= 201103L) || (_MSC_VER > 1900)  // C++11
    #define palloc_attr_noexcept   noexcept
  #else
    #define palloc_attr_noexcept   throw()
  #endif
#else
  #define palloc_attr_noexcept
#endif

#if defined(__cplusplus) && (__cplusplus >= 201703)
  #define palloc_decl_nodiscard    [[nodiscard]]
#elif (defined(__GNUC__) && (__GNUC__ >= 4)) || defined(__clang__)  // includes clang, icc, and clang-cl
  #define palloc_decl_nodiscard    __attribute__((warn_unused_result))
#elif defined(_HAS_NODISCARD)
  #define palloc_decl_nodiscard    _NODISCARD
#elif (_MSC_VER >= 1700)
  #define palloc_decl_nodiscard    _Check_return_
#else
  #define palloc_decl_nodiscard
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
  #if !defined(PALLOC_SHARED_LIB)
    #define palloc_decl_export
  #elif defined(PALLOC_SHARED_LIB_EXPORT)
    #define palloc_decl_export              __declspec(dllexport)
  #else
    #define palloc_decl_export              __declspec(dllimport)
  #endif
  #if defined(__MINGW32__)
    #define palloc_decl_restrict
    #define palloc_attr_malloc              __attribute__((malloc))
  #else
    #if (_MSC_VER >= 1900) && !defined(__EDG__)
      #define palloc_decl_restrict          __declspec(allocator) __declspec(restrict)
    #else
      #define palloc_decl_restrict          __declspec(restrict)
    #endif
    #define palloc_attr_malloc
  #endif
  #define palloc_cdecl                      __cdecl
  #define palloc_attr_alloc_size(s)
  #define palloc_attr_alloc_size2(s1,s2)
  #define palloc_attr_alloc_align(p)
#elif defined(__GNUC__)                 // includes clang and icc
  #if defined(PALLOC_SHARED_LIB) && defined(PALLOC_SHARED_LIB_EXPORT)
    #define palloc_decl_export              __attribute__((visibility("default")))
  #else
    #define palloc_decl_export
  #endif
  #define palloc_cdecl                      // leads to warnings... __attribute__((cdecl))
  #define palloc_decl_restrict
  #define palloc_attr_malloc                __attribute__((malloc))
  #if (defined(__clang_major__) && (__clang_major__ < 4)) || (__GNUC__ < 5)
    #define palloc_attr_alloc_size(s)
    #define palloc_attr_alloc_size2(s1,s2)
    #define palloc_attr_alloc_align(p)
  #elif defined(__INTEL_COMPILER)
    #define palloc_attr_alloc_size(s)       __attribute__((alloc_size(s)))
    #define palloc_attr_alloc_size2(s1,s2)  __attribute__((alloc_size(s1,s2)))
    #define palloc_attr_alloc_align(p)
  #else
    #define palloc_attr_alloc_size(s)       __attribute__((alloc_size(s)))
    #define palloc_attr_alloc_size2(s1,s2)  __attribute__((alloc_size(s1,s2)))
    #define palloc_attr_alloc_align(p)      __attribute__((alloc_align(p)))
  #endif
#else
  #define palloc_cdecl
  #define palloc_decl_export
  #define palloc_decl_restrict
  #define palloc_attr_malloc
  #define palloc_attr_alloc_size(s)
  #define palloc_attr_alloc_size2(s1,s2)
  #define palloc_attr_alloc_align(p)
#endif

// ------------------------------------------------------
// Includes
// ------------------------------------------------------

#include <stddef.h>     // size_t
#include <stdbool.h>    // bool
#include <stdint.h>     // INTPTR_MAX

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------------------------------
// Standard malloc interface
// ------------------------------------------------------

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_malloc(size_t size)  palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_calloc(size_t count, size_t size)  palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(1,2);
palloc_decl_nodiscard palloc_decl_export void* palloc_realloc(void* p, size_t newsize)      palloc_attr_noexcept palloc_attr_alloc_size(2);
palloc_decl_export void* palloc_expand(void* p, size_t newsize)                         palloc_attr_noexcept palloc_attr_alloc_size(2);

palloc_decl_export void palloc_free(void* p) palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict char* palloc_strdup(const char* s) palloc_attr_noexcept palloc_attr_malloc;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict char* palloc_strndup(const char* s, size_t n) palloc_attr_noexcept palloc_attr_malloc;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict char* palloc_realpath(const char* fname, char* resolved_name) palloc_attr_noexcept palloc_attr_malloc;

// ------------------------------------------------------
// Extended functionality
// ------------------------------------------------------
#define PALLOC_SMALL_WSIZE_MAX  (128)
#define PALLOC_SMALL_SIZE_MAX   (PALLOC_SMALL_WSIZE_MAX*sizeof(void*))

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_malloc_small(size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_zalloc_small(size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_zalloc(size_t size)       palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_mallocn(size_t count, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(1,2);
palloc_decl_nodiscard palloc_decl_export void* palloc_reallocn(void* p, size_t count, size_t size)        palloc_attr_noexcept palloc_attr_alloc_size2(2,3);
palloc_decl_nodiscard palloc_decl_export void* palloc_reallocf(void* p, size_t newsize)                   palloc_attr_noexcept palloc_attr_alloc_size(2);

palloc_decl_nodiscard palloc_decl_export size_t palloc_usable_size(const void* p) palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export size_t palloc_good_size(size_t size)     palloc_attr_noexcept;


// ------------------------------------------------------
// Internals
// ------------------------------------------------------

typedef void (palloc_cdecl palloc_deferred_free_fun)(bool force, unsigned long long heartbeat, void* arg);
palloc_decl_export void palloc_register_deferred_free(palloc_deferred_free_fun* deferred_free, void* arg) palloc_attr_noexcept;

typedef void (palloc_cdecl palloc_output_fun)(const char* msg, void* arg);
palloc_decl_export void palloc_register_output(palloc_output_fun* out, void* arg) palloc_attr_noexcept;

typedef void (palloc_cdecl palloc_error_fun)(int err, void* arg);
palloc_decl_export void palloc_register_error(palloc_error_fun* fun, void* arg);

palloc_decl_export void palloc_collect(bool force)    palloc_attr_noexcept;
palloc_decl_export int  palloc_version(void)          palloc_attr_noexcept;
palloc_decl_export void palloc_stats_reset(void)      palloc_attr_noexcept;
palloc_decl_export void palloc_stats_merge(void)      palloc_attr_noexcept;
palloc_decl_export void palloc_stats_print(void* out) palloc_attr_noexcept;  // backward compatibility: `out` is ignored and should be NULL
palloc_decl_export void palloc_stats_print_out(palloc_output_fun* out, void* arg) palloc_attr_noexcept;
palloc_decl_export void palloc_thread_stats_print_out(palloc_output_fun* out, void* arg) palloc_attr_noexcept;
palloc_decl_export void palloc_options_print(void)    palloc_attr_noexcept;

palloc_decl_export void palloc_process_info(size_t* elapsed_msecs, size_t* user_msecs, size_t* system_msecs,
                                    size_t* current_rss, size_t* peak_rss,
                                    size_t* current_commit, size_t* peak_commit, size_t* page_faults) palloc_attr_noexcept;


// Generally do not use the following as these are usually called automatically
palloc_decl_export void palloc_process_init(void)     palloc_attr_noexcept;
palloc_decl_export void palloc_cdecl palloc_process_done(void) palloc_attr_noexcept;
palloc_decl_export void palloc_thread_init(void)      palloc_attr_noexcept;
palloc_decl_export void palloc_thread_done(void)      palloc_attr_noexcept;


// -------------------------------------------------------------------------------------
// Aligned allocation
// Note that `alignment` always follows `size` for consistency with unaligned
// allocation, but unfortunately this differs from `posix_memalign` and `aligned_alloc`.
// -------------------------------------------------------------------------------------

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_malloc_aligned(size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1) palloc_attr_alloc_align(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_malloc_aligned_at(size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_zalloc_aligned(size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1) palloc_attr_alloc_align(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_zalloc_aligned_at(size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_calloc_aligned(size_t count, size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(1,2) palloc_attr_alloc_align(3);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_calloc_aligned_at(size_t count, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(1,2);
palloc_decl_nodiscard palloc_decl_export void* palloc_realloc_aligned(void* p, size_t newsize, size_t alignment) palloc_attr_noexcept palloc_attr_alloc_size(2) palloc_attr_alloc_align(3);
palloc_decl_nodiscard palloc_decl_export void* palloc_realloc_aligned_at(void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_alloc_size(2);


// -----------------------------------------------------------------
// Return allocated block size (if the return value is not NULL)
// -----------------------------------------------------------------

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_umalloc(size_t size, size_t* block_size)  palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_ucalloc(size_t count, size_t size, size_t* block_size)  palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(1,2);
palloc_decl_nodiscard palloc_decl_export void* palloc_urealloc(void* p, size_t newsize, size_t* block_size_pre, size_t* block_size_post) palloc_attr_noexcept palloc_attr_alloc_size(2);
palloc_decl_export void palloc_ufree(void* p, size_t* block_size) palloc_attr_noexcept;

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_umalloc_aligned(size_t size, size_t alignment, size_t* block_size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1) palloc_attr_alloc_align(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_uzalloc_aligned(size_t size, size_t alignment, size_t* block_size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1) palloc_attr_alloc_align(2);

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_umalloc_small(size_t size, size_t* block_size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_uzalloc_small(size_t size, size_t* block_size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);


// -------------------------------------------------------------------------------------
// Heaps: first-class, but can only allocate from the same thread that created it.
// -------------------------------------------------------------------------------------

struct palloc_heap_s;
typedef struct palloc_heap_s palloc_heap_t;

palloc_decl_nodiscard palloc_decl_export palloc_heap_t* palloc_heap_new(void);
palloc_decl_export void       palloc_heap_delete(palloc_heap_t* heap);
palloc_decl_export void       palloc_heap_destroy(palloc_heap_t* heap);
palloc_decl_export palloc_heap_t* palloc_heap_set_default(palloc_heap_t* heap);
palloc_decl_export palloc_heap_t* palloc_heap_get_default(void);
palloc_decl_export palloc_heap_t* palloc_heap_get_backing(void);
palloc_decl_export void       palloc_heap_collect(palloc_heap_t* heap, bool force) palloc_attr_noexcept;

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_malloc(palloc_heap_t* heap, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_zalloc(palloc_heap_t* heap, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_calloc(palloc_heap_t* heap, size_t count, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(2, 3);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_mallocn(palloc_heap_t* heap, size_t count, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(2, 3);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_malloc_small(palloc_heap_t* heap, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2);

palloc_decl_nodiscard palloc_decl_export void* palloc_heap_realloc(palloc_heap_t* heap, void* p, size_t newsize)              palloc_attr_noexcept palloc_attr_alloc_size(3);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_reallocn(palloc_heap_t* heap, void* p, size_t count, size_t size)  palloc_attr_noexcept palloc_attr_alloc_size2(3,4);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_reallocf(palloc_heap_t* heap, void* p, size_t newsize)             palloc_attr_noexcept palloc_attr_alloc_size(3);

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict char* palloc_heap_strdup(palloc_heap_t* heap, const char* s)            palloc_attr_noexcept palloc_attr_malloc;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict char* palloc_heap_strndup(palloc_heap_t* heap, const char* s, size_t n) palloc_attr_noexcept palloc_attr_malloc;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict char* palloc_heap_realpath(palloc_heap_t* heap, const char* fname, char* resolved_name) palloc_attr_noexcept palloc_attr_malloc;

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_malloc_aligned(palloc_heap_t* heap, size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2) palloc_attr_alloc_align(3);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_malloc_aligned_at(palloc_heap_t* heap, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_zalloc_aligned(palloc_heap_t* heap, size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2) palloc_attr_alloc_align(3);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_zalloc_aligned_at(palloc_heap_t* heap, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_calloc_aligned(palloc_heap_t* heap, size_t count, size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(2, 3) palloc_attr_alloc_align(4);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_calloc_aligned_at(palloc_heap_t* heap, size_t count, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size2(2, 3);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_realloc_aligned(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment) palloc_attr_noexcept palloc_attr_alloc_size(3) palloc_attr_alloc_align(4);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_realloc_aligned_at(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_alloc_size(3);


// --------------------------------------------------------------------------------
// Zero initialized re-allocation.
// Only valid on memory that was originally allocated with zero initialization too.
// e.g. `palloc_calloc`, `palloc_zalloc`, `palloc_zalloc_aligned` etc.
// see <https://github.com/microsoft/palloc/issues/63#issuecomment-508272992>
// --------------------------------------------------------------------------------

palloc_decl_nodiscard palloc_decl_export void* palloc_rezalloc(void* p, size_t newsize)                palloc_attr_noexcept palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export void* palloc_recalloc(void* p, size_t newcount, size_t size)  palloc_attr_noexcept palloc_attr_alloc_size2(2,3);

palloc_decl_nodiscard palloc_decl_export void* palloc_rezalloc_aligned(void* p, size_t newsize, size_t alignment) palloc_attr_noexcept palloc_attr_alloc_size(2) palloc_attr_alloc_align(3);
palloc_decl_nodiscard palloc_decl_export void* palloc_rezalloc_aligned_at(void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export void* palloc_recalloc_aligned(void* p, size_t newcount, size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_alloc_size2(2,3) palloc_attr_alloc_align(4);
palloc_decl_nodiscard palloc_decl_export void* palloc_recalloc_aligned_at(void* p, size_t newcount, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_alloc_size2(2,3);

palloc_decl_nodiscard palloc_decl_export void* palloc_heap_rezalloc(palloc_heap_t* heap, void* p, size_t newsize)                palloc_attr_noexcept palloc_attr_alloc_size(3);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_recalloc(palloc_heap_t* heap, void* p, size_t newcount, size_t size)  palloc_attr_noexcept palloc_attr_alloc_size2(3,4);

palloc_decl_nodiscard palloc_decl_export void* palloc_heap_rezalloc_aligned(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment) palloc_attr_noexcept palloc_attr_alloc_size(3) palloc_attr_alloc_align(4);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_rezalloc_aligned_at(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_alloc_size(3);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_recalloc_aligned(palloc_heap_t* heap, void* p, size_t newcount, size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_alloc_size2(3,4) palloc_attr_alloc_align(5);
palloc_decl_nodiscard palloc_decl_export void* palloc_heap_recalloc_aligned_at(palloc_heap_t* heap, void* p, size_t newcount, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept palloc_attr_alloc_size2(3,4);


// ------------------------------------------------------
// Analysis
// ------------------------------------------------------

palloc_decl_export bool palloc_heap_contains_block(palloc_heap_t* heap, const void* p);
palloc_decl_export bool palloc_heap_check_owned(palloc_heap_t* heap, const void* p);
palloc_decl_export bool palloc_check_owned(const void* p);

// An area of heap space contains blocks of a single size.
typedef struct palloc_heap_area_s {
  void*  blocks;      // start of the area containing heap blocks
  size_t reserved;    // bytes reserved for this area (virtual)
  size_t committed;   // current available bytes for this area
  size_t used;        // number of allocated blocks
  size_t block_size;  // size in bytes of each block
  size_t full_block_size; // size in bytes of a full block including padding and metadata.
  int    heap_tag;    // heap tag associated with this area
} palloc_heap_area_t;

typedef bool (palloc_cdecl palloc_block_visit_fun)(const palloc_heap_t* heap, const palloc_heap_area_t* area, void* block, size_t block_size, void* arg);

palloc_decl_export bool palloc_heap_visit_blocks(const palloc_heap_t* heap, bool visit_blocks, palloc_block_visit_fun* visitor, void* arg);

// Experimental
palloc_decl_nodiscard palloc_decl_export bool palloc_is_in_heap_region(const void* p) palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export bool palloc_is_redirected(void) palloc_attr_noexcept;

palloc_decl_export int   palloc_reserve_huge_os_pages_interleave(size_t pages, size_t numa_nodes, size_t timeout_msecs) palloc_attr_noexcept;
palloc_decl_export int   palloc_reserve_huge_os_pages_at(size_t pages, int numa_node, size_t timeout_msecs) palloc_attr_noexcept;

palloc_decl_export int   palloc_reserve_os_memory(size_t size, bool commit, bool allow_large) palloc_attr_noexcept;
palloc_decl_export bool  palloc_manage_os_memory(void* start, size_t size, bool is_committed, bool is_large, bool is_zero, int numa_node) palloc_attr_noexcept;

palloc_decl_export void  palloc_debug_show_arenas(void) palloc_attr_noexcept;
palloc_decl_export void  palloc_arenas_print(void) palloc_attr_noexcept;

// Experimental: heaps associated with specific memory arena's
typedef int palloc_arena_id_t;
palloc_decl_export void* palloc_arena_area(palloc_arena_id_t arena_id, size_t* size);
palloc_decl_export int   palloc_reserve_huge_os_pages_at_ex(size_t pages, int numa_node, size_t timeout_msecs, bool exclusive, palloc_arena_id_t* arena_id) palloc_attr_noexcept;
palloc_decl_export int   palloc_reserve_os_memory_ex(size_t size, bool commit, bool allow_large, bool exclusive, palloc_arena_id_t* arena_id) palloc_attr_noexcept;
palloc_decl_export bool  palloc_manage_os_memory_ex(void* start, size_t size, bool is_committed, bool is_large, bool is_zero, int numa_node, bool exclusive, palloc_arena_id_t* arena_id) palloc_attr_noexcept;

#if PALLOC_MALLOC_VERSION >= 182
// Create a heap that only allocates in the specified arena
palloc_decl_nodiscard palloc_decl_export palloc_heap_t* palloc_heap_new_in_arena(palloc_arena_id_t arena_id);
#endif


// Experimental: allow sub-processes whose memory areas stay separated (and no reclamation between them)
// Used for example for separate interpreters in one process.
typedef void* palloc_subproc_id_t;
palloc_decl_export palloc_subproc_id_t palloc_subproc_main(void);
palloc_decl_export palloc_subproc_id_t palloc_subproc_new(void);
palloc_decl_export void palloc_subproc_delete(palloc_subproc_id_t subproc);
palloc_decl_export void palloc_subproc_add_current_thread(palloc_subproc_id_t subproc); // this should be called right after a thread is created (and no allocation has taken place yet)

// Experimental: visit abandoned heap areas (that are not owned by a specific heap)
palloc_decl_export bool palloc_abandoned_visit_blocks(palloc_subproc_id_t subproc_id, int heap_tag, bool visit_blocks, palloc_block_visit_fun* visitor, void* arg);

// Experimental: objects followed by a guard page.
// A sample rate of 0 disables guarded objects, while 1 uses a guard page for every object.
// A seed of 0 uses a random start point. Only objects within the size bound are eligable for guard pages.
palloc_decl_export void palloc_heap_guarded_set_sample_rate(palloc_heap_t* heap, size_t sample_rate, size_t seed);
palloc_decl_export void palloc_heap_guarded_set_size_bound(palloc_heap_t* heap, size_t min, size_t max);

// Experimental: communicate that the thread is part of a threadpool
palloc_decl_export void palloc_thread_set_in_threadpool(void) palloc_attr_noexcept;

// Experimental: create a new heap with a specified heap tag. Set `allow_destroy` to false to allow the thread
// to reclaim abandoned memory (with a compatible heap_tag and arena_id) but in that case `palloc_heap_destroy` will
// fall back to `palloc_heap_delete`.
palloc_decl_nodiscard palloc_decl_export palloc_heap_t* palloc_heap_new_ex(int heap_tag, bool allow_destroy, palloc_arena_id_t arena_id);

// deprecated
palloc_decl_export int palloc_reserve_huge_os_pages(size_t pages, double max_secs, size_t* pages_reserved) palloc_attr_noexcept;
palloc_decl_export void palloc_collect_reduce(size_t target_thread_owned) palloc_attr_noexcept;



// ------------------------------------------------------
// Convenience
// ------------------------------------------------------

#define palloc_malloc_tp(tp)                ((tp*)palloc_malloc(sizeof(tp)))
#define palloc_zalloc_tp(tp)                ((tp*)palloc_zalloc(sizeof(tp)))
#define palloc_calloc_tp(tp,n)              ((tp*)palloc_calloc(n,sizeof(tp)))
#define palloc_mallocn_tp(tp,n)             ((tp*)palloc_mallocn(n,sizeof(tp)))
#define palloc_reallocn_tp(p,tp,n)          ((tp*)palloc_reallocn(p,n,sizeof(tp)))
#define palloc_recalloc_tp(p,tp,n)          ((tp*)palloc_recalloc(p,n,sizeof(tp)))

#define palloc_heap_malloc_tp(hp,tp)        ((tp*)palloc_heap_malloc(hp,sizeof(tp)))
#define palloc_heap_zalloc_tp(hp,tp)        ((tp*)palloc_heap_zalloc(hp,sizeof(tp)))
#define palloc_heap_calloc_tp(hp,tp,n)      ((tp*)palloc_heap_calloc(hp,n,sizeof(tp)))
#define palloc_heap_mallocn_tp(hp,tp,n)     ((tp*)palloc_heap_mallocn(hp,n,sizeof(tp)))
#define palloc_heap_reallocn_tp(hp,p,tp,n)  ((tp*)palloc_heap_reallocn(hp,p,n,sizeof(tp)))
#define palloc_heap_recalloc_tp(hp,p,tp,n)  ((tp*)palloc_heap_recalloc(hp,p,n,sizeof(tp)))


// ------------------------------------------------------
// Options
// ------------------------------------------------------

typedef enum palloc_option_e {
  // stable options
  palloc_option_show_errors,                // print error messages
  palloc_option_show_stats,                 // print statistics on termination
  palloc_option_verbose,                    // print verbose messages
  // advanced options
  palloc_option_eager_commit,               // eager commit segments? (after `eager_commit_delay` segments) (=1)
  palloc_option_arena_eager_commit,         // eager commit arenas? Use 2 to enable just on overcommit systems (=2)
  palloc_option_purge_decommits,            // should a memory purge decommit? (=1). Set to 0 to use memory reset on a purge (instead of decommit)
  palloc_option_allow_large_os_pages,       // allow use of large (2 or 4 MiB) OS pages, implies eager commit.
  palloc_option_reserve_huge_os_pages,      // reserve N huge OS pages (1GiB pages) at startup
  palloc_option_reserve_huge_os_pages_at,   // reserve huge OS pages at a specific NUMA node
  palloc_option_reserve_os_memory,          // reserve specified amount of OS memory in an arena at startup (internally, this value is in KiB; use `palloc_option_get_size`)
  palloc_option_deprecated_segment_cache,
  palloc_option_deprecated_page_reset,
  palloc_option_abandoned_page_purge,       // immediately purge delayed purges on thread termination
  palloc_option_deprecated_segment_reset,
  palloc_option_eager_commit_delay,         // the first N segments per thread are not eagerly committed (but per page in the segment on demand)
  palloc_option_purge_delay,                // memory purging is delayed by N milli seconds; use 0 for immediate purging or -1 for no purging at all. (=10)
  palloc_option_use_numa_nodes,             // 0 = use all available numa nodes, otherwise use at most N nodes.
  palloc_option_disallow_os_alloc,          // 1 = do not use OS memory for allocation (but only programmatically reserved arenas)
  palloc_option_os_tag,                     // tag used for OS logging (macOS only for now) (=100)
  palloc_option_max_errors,                 // issue at most N error messages
  palloc_option_max_warnings,               // issue at most N warning messages
  palloc_option_max_segment_reclaim,        // max. percentage of the abandoned segments can be reclaimed per try (=10%)
  palloc_option_destroy_on_exit,            // if set, release all memory on exit; sometimes used for dynamic unloading but can be unsafe
  palloc_option_arena_reserve,              // initial memory size for arena reservation (= 1 GiB on 64-bit) (internally, this value is in KiB; use `palloc_option_get_size`)
  palloc_option_arena_purge_mult,           // multiplier for `purge_delay` for the purging delay for arenas (=10)
  palloc_option_purge_extend_delay,
  palloc_option_abandoned_reclaim_on_free,  // allow to reclaim an abandoned segment on a free (=1)
  palloc_option_disallow_arena_alloc,       // 1 = do not use arena's for allocation (except if using specific arena id's)
  palloc_option_retry_on_oom,               // retry on out-of-memory for N milli seconds (=400), set to 0 to disable retries. (only on windows)
  palloc_option_visit_abandoned,            // allow visiting heap blocks from abandoned threads (=0)
  palloc_option_guarded_min,                // only used when building with PALLOC_GUARDED: minimal rounded object size for guarded objects (=0)
  palloc_option_guarded_max,                // only used when building with PALLOC_GUARDED: maximal rounded object size for guarded objects (=0)
  palloc_option_guarded_precise,            // disregard minimal alignment requirement to always place guarded blocks exactly in front of a guard page (=0)
  palloc_option_guarded_sample_rate,        // 1 out of N allocations in the min/max range will be guarded (=1000)
  palloc_option_guarded_sample_seed,        // can be set to allow for a (more) deterministic re-execution when a guard page is triggered (=0)
  palloc_option_target_segments_per_thread, // experimental (=0)
  palloc_option_generic_collect,            // collect heaps every N (=10000) generic allocation calls
  palloc_option_allow_thp,                  // allow transparent huge pages? (=1) (on Android =0 by default). Set to 0 to disable THP for the process.
  _palloc_option_last,
  // legacy option names
  palloc_option_large_os_pages = palloc_option_allow_large_os_pages,
  palloc_option_eager_region_commit = palloc_option_arena_eager_commit,
  palloc_option_reset_decommits = palloc_option_purge_decommits,
  palloc_option_reset_delay = palloc_option_purge_delay,
  palloc_option_abandoned_page_reset = palloc_option_abandoned_page_purge,
  palloc_option_limit_os_alloc = palloc_option_disallow_os_alloc
} palloc_option_t;


palloc_decl_nodiscard palloc_decl_export bool palloc_option_is_enabled(palloc_option_t option);
palloc_decl_export void palloc_option_enable(palloc_option_t option);
palloc_decl_export void palloc_option_disable(palloc_option_t option);
palloc_decl_export void palloc_option_set_enabled(palloc_option_t option, bool enable);
palloc_decl_export void palloc_option_set_enabled_default(palloc_option_t option, bool enable);

palloc_decl_nodiscard palloc_decl_export long   palloc_option_get(palloc_option_t option);
palloc_decl_nodiscard palloc_decl_export long   palloc_option_get_clamp(palloc_option_t option, long min, long max);
palloc_decl_nodiscard palloc_decl_export size_t palloc_option_get_size(palloc_option_t option);
palloc_decl_export void palloc_option_set(palloc_option_t option, long value);
palloc_decl_export void palloc_option_set_default(palloc_option_t option, long value);


// -------------------------------------------------------------------------------------------------------
// "mi" prefixed implementations of various posix, Unix, Windows, and C++ allocation functions.
// (This can be convenient when providing overrides of these functions as done in `palloc-override.h`.)
// note: we use `palloc_cfree` as "checked free" and it checks if the pointer is in our heap before free-ing.
// -------------------------------------------------------------------------------------------------------

palloc_decl_export void  palloc_cfree(void* p) palloc_attr_noexcept;
palloc_decl_export void* palloc__expand(void* p, size_t newsize) palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export size_t palloc_malloc_size(const void* p)        palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export size_t palloc_malloc_good_size(size_t size)     palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export size_t palloc_malloc_usable_size(const void *p) palloc_attr_noexcept;

palloc_decl_export int palloc_posix_memalign(void** p, size_t alignment, size_t size)   palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_memalign(size_t alignment, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2) palloc_attr_alloc_align(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_valloc(size_t size)  palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_pvalloc(size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_aligned_alloc(size_t alignment, size_t size) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(2) palloc_attr_alloc_align(1);

palloc_decl_nodiscard palloc_decl_export void* palloc_reallocarray(void* p, size_t count, size_t size) palloc_attr_noexcept palloc_attr_alloc_size2(2,3);
palloc_decl_nodiscard palloc_decl_export int   palloc_reallocarr(void* p, size_t count, size_t size) palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export void* palloc_aligned_recalloc(void* p, size_t newcount, size_t size, size_t alignment) palloc_attr_noexcept;
palloc_decl_nodiscard palloc_decl_export void* palloc_aligned_offset_recalloc(void* p, size_t newcount, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept;

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict unsigned short* palloc_wcsdup(const unsigned short* s) palloc_attr_noexcept palloc_attr_malloc;
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict unsigned char*  palloc_mbsdup(const unsigned char* s)  palloc_attr_noexcept palloc_attr_malloc;
palloc_decl_export int palloc_dupenv_s(char** buf, size_t* size, const char* name)                      palloc_attr_noexcept;
palloc_decl_export int palloc_wdupenv_s(unsigned short** buf, size_t* size, const unsigned short* name) palloc_attr_noexcept;

palloc_decl_export void palloc_free_size(void* p, size_t size)                           palloc_attr_noexcept;
palloc_decl_export void palloc_free_size_aligned(void* p, size_t size, size_t alignment) palloc_attr_noexcept;
palloc_decl_export void palloc_free_aligned(void* p, size_t alignment)                   palloc_attr_noexcept;

// The `palloc_new` wrappers implement C++ semantics on out-of-memory instead of directly returning `NULL`.
// (and call `std::get_new_handler` and potentially raise a `std::bad_alloc` exception).
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_new(size_t size)                   palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_new_aligned(size_t size, size_t alignment) palloc_attr_malloc palloc_attr_alloc_size(1) palloc_attr_alloc_align(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_new_nothrow(size_t size)           palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_new_aligned_nothrow(size_t size, size_t alignment) palloc_attr_noexcept palloc_attr_malloc palloc_attr_alloc_size(1) palloc_attr_alloc_align(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_new_n(size_t count, size_t size)   palloc_attr_malloc palloc_attr_alloc_size2(1, 2);
palloc_decl_nodiscard palloc_decl_export void* palloc_new_realloc(void* p, size_t newsize)                palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export void* palloc_new_reallocn(void* p, size_t newcount, size_t size) palloc_attr_alloc_size2(2, 3);

palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_alloc_new(palloc_heap_t* heap, size_t size)                palloc_attr_malloc palloc_attr_alloc_size(2);
palloc_decl_nodiscard palloc_decl_export palloc_decl_restrict void* palloc_heap_alloc_new_n(palloc_heap_t* heap, size_t count, size_t size) palloc_attr_malloc palloc_attr_alloc_size2(2, 3);

#ifdef __cplusplus
}
#endif

// ---------------------------------------------------------------------------------------------
// Implement the C++ std::allocator interface for use in STL containers.
// (note: see `palloc-new-delete.h` for overriding the new/delete operators globally)
// ---------------------------------------------------------------------------------------------
#ifdef __cplusplus

#include <cstddef>     // std::size_t
#include <cstdint>     // PTRDIFF_MAX
#if (__cplusplus >= 201103L) || (_MSC_VER > 1900)  // C++11
#include <type_traits> // std::true_type
#include <utility>     // std::forward
#endif

template<class T> struct _palloc_stl_allocator_common {
  typedef T                 value_type;
  typedef std::size_t       size_type;
  typedef std::ptrdiff_t    difference_type;
  typedef value_type&       reference;
  typedef value_type const& const_reference;
  typedef value_type*       pointer;
  typedef value_type const* const_pointer;

  #if ((__cplusplus >= 201103L) || (_MSC_VER > 1900))  // C++11
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap            = std::true_type;
  template <class U, class ...Args> void construct(U* p, Args&& ...args) { ::new(p) U(std::forward<Args>(args)...); }
  template <class U> void destroy(U* p) palloc_attr_noexcept { p->~U(); }
  #else
  void construct(pointer p, value_type const& val) { ::new(p) value_type(val); }
  void destroy(pointer p) { p->~value_type(); }
  #endif

  size_type     max_size() const palloc_attr_noexcept { return (PTRDIFF_MAX/sizeof(value_type)); }
  pointer       address(reference x) const        { return &x; }
  const_pointer address(const_reference x) const  { return &x; }
};

template<class T> struct palloc_stl_allocator : public _palloc_stl_allocator_common<T> {
  using typename _palloc_stl_allocator_common<T>::size_type;
  using typename _palloc_stl_allocator_common<T>::value_type;
  using typename _palloc_stl_allocator_common<T>::pointer;
  template <class U> struct rebind { typedef palloc_stl_allocator<U> other; };

  palloc_stl_allocator()                                             palloc_attr_noexcept = default;
  palloc_stl_allocator(const palloc_stl_allocator&)                      palloc_attr_noexcept = default;
  template<class U> palloc_stl_allocator(const palloc_stl_allocator<U>&) palloc_attr_noexcept { }
  palloc_stl_allocator  select_on_container_copy_construction() const { return *this; }
  void              deallocate(T* p, size_type) { palloc_free(p); }

  #if (__cplusplus >= 201703L)  // C++17
  palloc_decl_nodiscard T* allocate(size_type count) { return static_cast<T*>(palloc_new_n(count, sizeof(T))); }
  palloc_decl_nodiscard T* allocate(size_type count, const void*) { return allocate(count); }
  #else
  palloc_decl_nodiscard pointer allocate(size_type count, const void* = 0) { return static_cast<pointer>(palloc_new_n(count, sizeof(value_type))); }
  #endif

  #if ((__cplusplus >= 201103L) || (_MSC_VER > 1900))  // C++11
  using is_always_equal = std::true_type;
  #endif
};

template<class T1,class T2> bool operator==(const palloc_stl_allocator<T1>& , const palloc_stl_allocator<T2>& ) palloc_attr_noexcept { return true; }
template<class T1,class T2> bool operator!=(const palloc_stl_allocator<T1>& , const palloc_stl_allocator<T2>& ) palloc_attr_noexcept { return false; }


#if (__cplusplus >= 201103L) || (_MSC_VER >= 1900)  // C++11
#define PALLOC_HAS_HEAP_STL_ALLOCATOR 1

#include <memory>      // std::shared_ptr

// Common base class for STL allocators in a specific heap
template<class T, bool _palloc_destroy> struct _palloc_heap_stl_allocator_common : public _palloc_stl_allocator_common<T> {
  using typename _palloc_stl_allocator_common<T>::size_type;
  using typename _palloc_stl_allocator_common<T>::value_type;
  using typename _palloc_stl_allocator_common<T>::pointer;

  _palloc_heap_stl_allocator_common(palloc_heap_t* hp) : heap(hp, [](palloc_heap_t*) {}) {}    /* will not delete nor destroy the passed in heap */

  #if (__cplusplus >= 201703L)  // C++17
  palloc_decl_nodiscard T* allocate(size_type count) { return static_cast<T*>(palloc_heap_alloc_new_n(this->heap.get(), count, sizeof(T))); }
  palloc_decl_nodiscard T* allocate(size_type count, const void*) { return allocate(count); }
  #else
  palloc_decl_nodiscard pointer allocate(size_type count, const void* = 0) { return static_cast<pointer>(palloc_heap_alloc_new_n(this->heap.get(), count, sizeof(value_type))); }
  #endif

  #if ((__cplusplus >= 201103L) || (_MSC_VER > 1900))  // C++11
  using is_always_equal = std::false_type;
  #endif

  void collect(bool force) { palloc_heap_collect(this->heap.get(), force); }
  template<class U> bool is_equal(const _palloc_heap_stl_allocator_common<U, _palloc_destroy>& x) const { return (this->heap == x.heap); }

protected:
  std::shared_ptr<palloc_heap_t> heap;
  template<class U, bool D> friend struct _palloc_heap_stl_allocator_common;

  _palloc_heap_stl_allocator_common() {
    palloc_heap_t* hp = palloc_heap_new();
    this->heap.reset(hp, (_palloc_destroy ? &heap_destroy : &heap_delete));  /* calls heap_delete/destroy when the refcount drops to zero */
  }
  _palloc_heap_stl_allocator_common(const _palloc_heap_stl_allocator_common& x) palloc_attr_noexcept : heap(x.heap) { }
  template<class U> _palloc_heap_stl_allocator_common(const _palloc_heap_stl_allocator_common<U, _palloc_destroy>& x) palloc_attr_noexcept : heap(x.heap) { }

private:
  static void heap_delete(palloc_heap_t* hp)  { if (hp != NULL) { palloc_heap_delete(hp); } }
  static void heap_destroy(palloc_heap_t* hp) { if (hp != NULL) { palloc_heap_destroy(hp); } }
};

// STL allocator allocation in a specific heap
template<class T> struct palloc_heap_stl_allocator : public _palloc_heap_stl_allocator_common<T, false> {
  using typename _palloc_heap_stl_allocator_common<T, false>::size_type;
  palloc_heap_stl_allocator() : _palloc_heap_stl_allocator_common<T, false>() { } // creates fresh heap that is deleted when the destructor is called
  palloc_heap_stl_allocator(palloc_heap_t* hp) : _palloc_heap_stl_allocator_common<T, false>(hp) { }  // no delete nor destroy on the passed in heap
  template<class U> palloc_heap_stl_allocator(const palloc_heap_stl_allocator<U>& x) palloc_attr_noexcept : _palloc_heap_stl_allocator_common<T, false>(x) { }

  palloc_heap_stl_allocator select_on_container_copy_construction() const { return *this; }
  void deallocate(T* p, size_type) { palloc_free(p); }
  template<class U> struct rebind { typedef palloc_heap_stl_allocator<U> other; };
};

template<class T1, class T2> bool operator==(const palloc_heap_stl_allocator<T1>& x, const palloc_heap_stl_allocator<T2>& y) palloc_attr_noexcept { return (x.is_equal(y)); }
template<class T1, class T2> bool operator!=(const palloc_heap_stl_allocator<T1>& x, const palloc_heap_stl_allocator<T2>& y) palloc_attr_noexcept { return (!x.is_equal(y)); }


// STL allocator allocation in a specific heap, where `free` does nothing and
// the heap is destroyed in one go on destruction -- use with care!
template<class T> struct palloc_heap_destroy_stl_allocator : public _palloc_heap_stl_allocator_common<T, true> {
  using typename _palloc_heap_stl_allocator_common<T, true>::size_type;
  palloc_heap_destroy_stl_allocator() : _palloc_heap_stl_allocator_common<T, true>() { } // creates fresh heap that is destroyed when the destructor is called
  palloc_heap_destroy_stl_allocator(palloc_heap_t* hp) : _palloc_heap_stl_allocator_common<T, true>(hp) { }  // no delete nor destroy on the passed in heap
  template<class U> palloc_heap_destroy_stl_allocator(const palloc_heap_destroy_stl_allocator<U>& x) palloc_attr_noexcept : _palloc_heap_stl_allocator_common<T, true>(x) { }

  palloc_heap_destroy_stl_allocator select_on_container_copy_construction() const { return *this; }
  void deallocate(T*, size_type) { /* do nothing as we destroy the heap on destruct. */ }
  template<class U> struct rebind { typedef palloc_heap_destroy_stl_allocator<U> other; };
};

template<class T1, class T2> bool operator==(const palloc_heap_destroy_stl_allocator<T1>& x, const palloc_heap_destroy_stl_allocator<T2>& y) palloc_attr_noexcept { return (x.is_equal(y)); }
template<class T1, class T2> bool operator!=(const palloc_heap_destroy_stl_allocator<T1>& x, const palloc_heap_destroy_stl_allocator<T2>& y) palloc_attr_noexcept { return (!x.is_equal(y)); }

#endif // C++11

#endif // __cplusplus

#endif
