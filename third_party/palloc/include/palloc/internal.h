/* ----------------------------------------------------------------------------
Copyright (c) 2018-2023, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_INTERNAL_H
#define PALLOC_INTERNAL_H

// --------------------------------------------------------------------------
// This file contains the internal API's of palloc and various utility
// functions and macros.
// --------------------------------------------------------------------------

#include "types.h"
#include "track.h"


// --------------------------------------------------------------------------
// Compiler defines
// --------------------------------------------------------------------------

#if (PALLOC_DEBUG>0)
#define palloc_trace_message(...)  _palloc_trace_message(__VA_ARGS__)
#else
#define palloc_trace_message(...)
#endif

#define palloc_decl_cache_align     palloc_decl_align(64)

#if defined(_MSC_VER)
#pragma warning(disable:4127)   // suppress constant conditional warning (due to PALLOC_SECURE paths)
#pragma warning(disable:26812)  // unscoped enum warning
#define palloc_decl_noinline        __declspec(noinline)
#define palloc_decl_thread          __declspec(thread)
#define palloc_decl_align(a)        __declspec(align(a))
#define palloc_decl_noreturn        __declspec(noreturn)
#define palloc_decl_weak
#define palloc_decl_hidden
#define palloc_decl_cold
#elif (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__clang__) // includes clang and icc
#define palloc_decl_noinline        __attribute__((noinline))
#define palloc_decl_thread          __thread
#define palloc_decl_align(a)        __attribute__((aligned(a)))
#define palloc_decl_noreturn        __attribute__((noreturn))
#define palloc_decl_weak            __attribute__((weak))
#define palloc_decl_hidden          __attribute__((visibility("hidden")))
#if (__GNUC__ >= 4) || defined(__clang__)
#define palloc_decl_cold            __attribute__((cold))
#else
#define palloc_decl_cold
#endif
#elif __cplusplus >= 201103L    // c++11
#define palloc_decl_noinline
#define palloc_decl_thread          thread_local
#define palloc_decl_align(a)        alignas(a)
#define palloc_decl_noreturn        [[noreturn]]
#define palloc_decl_weak
#define palloc_decl_hidden
#define palloc_decl_cold
#else
#define palloc_decl_noinline
#define palloc_decl_thread          __thread        // hope for the best :-)
#define palloc_decl_align(a)
#define palloc_decl_noreturn
#define palloc_decl_weak
#define palloc_decl_hidden
#define palloc_decl_cold
#endif

#if defined(__GNUC__) || defined(__clang__)
#define palloc_unlikely(x)     (__builtin_expect(!!(x),false))
#define palloc_likely(x)       (__builtin_expect(!!(x),true))
#elif (defined(__cplusplus) && (__cplusplus >= 202002L)) || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
#define palloc_unlikely(x)     (x) [[unlikely]]
#define palloc_likely(x)       (x) [[likely]]
#else
#define palloc_unlikely(x)     (x)
#define palloc_likely(x)       (x)
#endif

#ifndef __has_builtin
#define __has_builtin(x)    0
#endif

#if defined(__cplusplus)
#define palloc_decl_externc     extern "C"
#else
#define palloc_decl_externc
#endif

#if defined(__EMSCRIPTEN__) && !defined(__wasi__)
#define __wasi__
#endif


// --------------------------------------------------------------------------
// Internal functions
// --------------------------------------------------------------------------

// "libc.c"
#include    <stdarg.h>
int         _palloc_vsnprintf(char* buf, size_t bufsize, const char* fmt, va_list args);
int         _palloc_snprintf(char* buf, size_t buflen, const char* fmt, ...);
char        _palloc_toupper(char c);
int         _palloc_strnicmp(const char* s, const char* t, size_t n);
void        _palloc_strlcpy(char* dest, const char* src, size_t dest_size);
void        _palloc_strlcat(char* dest, const char* src, size_t dest_size);
size_t      _palloc_strlen(const char* s);
size_t      _palloc_strnlen(const char* s, size_t max_len);
bool        _palloc_getenv(const char* name, char* result, size_t result_size);

// "options.c"
void        _palloc_fputs(palloc_output_fun* out, void* arg, const char* prefix, const char* message);
void        _palloc_fprintf(palloc_output_fun* out, void* arg, const char* fmt, ...);
void        _palloc_message(const char* fmt, ...);
void        _palloc_warning_message(const char* fmt, ...);
void        _palloc_verbose_message(const char* fmt, ...);
void        _palloc_trace_message(const char* fmt, ...);
void        _palloc_options_init(void);
long        _palloc_option_get_fast(palloc_option_t option);
void        _palloc_error_message(int err, const char* fmt, ...);

// random.c
void        _palloc_random_init(palloc_random_ctx_t* ctx);
void        _palloc_random_init_weak(palloc_random_ctx_t* ctx);
void        _palloc_random_reinit_if_weak(palloc_random_ctx_t * ctx);
void        _palloc_random_split(palloc_random_ctx_t* ctx, palloc_random_ctx_t* new_ctx);
uintptr_t   _palloc_random_next(palloc_random_ctx_t* ctx);
uintptr_t   _palloc_heap_random_next(palloc_heap_t* heap);
uintptr_t   _palloc_os_random_weak(uintptr_t extra_seed);
static inline uintptr_t _palloc_random_shuffle(uintptr_t x);

// init.c
extern palloc_decl_hidden palloc_decl_cache_align palloc_stats_t       _palloc_stats_main;
extern palloc_decl_hidden palloc_decl_cache_align const palloc_page_t  _palloc_page_empty;
void        _palloc_auto_process_init(void);
void palloc_cdecl _palloc_auto_process_done(void) palloc_attr_noexcept;
bool        _palloc_is_redirected(void);
bool        _palloc_allocator_init(const char** message);
void        _palloc_allocator_done(void);
bool        _palloc_is_main_thread(void);
size_t      _palloc_current_thread_count(void);
bool        _palloc_preloading(void);           // true while the C runtime is not initialized yet
void        _palloc_thread_done(palloc_heap_t* heap);
void        _palloc_thread_data_collect(void);
void        _palloc_tld_init(palloc_tld_t* tld, palloc_heap_t* bheap);
palloc_threadid_t _palloc_thread_id(void) palloc_attr_noexcept;
palloc_heap_t*    _palloc_heap_main_get(void);     // statically allocated main backing heap
palloc_subproc_t* _palloc_subproc_from_id(palloc_subproc_id_t subproc_id);
void        _palloc_heap_guarded_init(palloc_heap_t* heap);

// os.c
void        _palloc_os_init(void);                                            // called from process init
void*       _palloc_os_alloc(size_t size, palloc_memid_t* memid);
void*       _palloc_os_zalloc(size_t size, palloc_memid_t* memid);
void        _palloc_os_free(void* p, size_t size, palloc_memid_t memid);
void        _palloc_os_free_ex(void* p, size_t size, bool still_committed, palloc_memid_t memid);

size_t      _palloc_os_page_size(void);
size_t      _palloc_os_good_alloc_size(size_t size);
bool        _palloc_os_has_overcommit(void);
bool        _palloc_os_has_virtual_reserve(void);

bool        _palloc_os_reset(void* addr, size_t size);
bool        _palloc_os_decommit(void* addr, size_t size);
bool        _palloc_os_unprotect(void* addr, size_t size);
bool        _palloc_os_purge(void* p, size_t size);
bool        _palloc_os_purge_ex(void* p, size_t size, bool allow_reset, size_t stat_size);
void        _palloc_os_reuse(void* p, size_t size);
palloc_decl_nodiscard bool _palloc_os_commit(void* p, size_t size, bool* is_zero);
palloc_decl_nodiscard bool _palloc_os_commit_ex(void* addr, size_t size, bool* is_zero, size_t stat_size);
bool        _palloc_os_protect(void* addr, size_t size);

void*       _palloc_os_alloc_aligned(size_t size, size_t alignment, bool commit, bool allow_large, palloc_memid_t* memid);
void*       _palloc_os_alloc_aligned_at_offset(size_t size, size_t alignment, size_t align_offset, bool commit, bool allow_large, palloc_memid_t* memid);

void*       _palloc_os_get_aligned_hint(size_t try_alignment, size_t size);
bool        _palloc_os_canuse_large_page(size_t size, size_t alignment);
size_t      _palloc_os_large_page_size(void);
void*       _palloc_os_alloc_huge_os_pages(size_t pages, int numa_node, palloc_msecs_t max_secs, size_t* pages_reserved, size_t* psize, palloc_memid_t* memid);

int         _palloc_os_numa_node_count(void);
int         _palloc_os_numa_node(void);

// arena.c
palloc_arena_id_t _palloc_arena_id_none(void);
void        _palloc_arena_free(void* p, size_t size, size_t still_committed_size, palloc_memid_t memid);
void*       _palloc_arena_alloc(size_t size, bool commit, bool allow_large, palloc_arena_id_t req_arena_id, palloc_memid_t* memid);
void*       _palloc_arena_alloc_aligned(size_t size, size_t alignment, size_t align_offset, bool commit, bool allow_large, palloc_arena_id_t req_arena_id, palloc_memid_t* memid);
bool        _palloc_arena_memid_is_suitable(palloc_memid_t memid, palloc_arena_id_t request_arena_id);
bool        _palloc_arena_contains(const void* p);
void        _palloc_arenas_collect(bool force_purge);
void        _palloc_arena_unsafe_destroy_all(void);

bool        _palloc_arena_segment_clear_abandoned(palloc_segment_t* segment);
void        _palloc_arena_segment_mark_abandoned(palloc_segment_t* segment);

void*       _palloc_arena_meta_zalloc(size_t size, palloc_memid_t* memid);
void        _palloc_arena_meta_free(void* p, palloc_memid_t memid, size_t size);

typedef struct palloc_arena_field_cursor_s { // abstract struct
  size_t         os_list_count;           // max entries to visit in the OS abandoned list
  size_t         start;                   // start arena idx (may need to be wrapped)
  size_t         end;                     // end arena idx (exclusive, may need to be wrapped)
  size_t         bitmap_idx;              // current bit idx for an arena
  palloc_subproc_t*  subproc;                 // only visit blocks in this sub-process
  bool           visit_all;               // ensure all abandoned blocks are seen (blocking)
  bool           hold_visit_lock;         // if the subproc->abandoned_os_visit_lock is held
} palloc_arena_field_cursor_t;
void          _palloc_arena_field_cursor_init(palloc_heap_t* heap, palloc_subproc_t* subproc, bool visit_all, palloc_arena_field_cursor_t* current);
palloc_segment_t* _palloc_arena_segment_clear_abandoned_next(palloc_arena_field_cursor_t* previous);
void          _palloc_arena_field_cursor_done(palloc_arena_field_cursor_t* current);

// "segment-map.c"
void        _palloc_segment_map_allocated_at(const palloc_segment_t* segment);
void        _palloc_segment_map_freed_at(const palloc_segment_t* segment);
void        _palloc_segment_map_unsafe_destroy(void);

// "segment.c"
palloc_page_t* _palloc_segment_page_alloc(palloc_heap_t* heap, size_t block_size, size_t page_alignment, palloc_segments_tld_t* tld);
void       _palloc_segment_page_free(palloc_page_t* page, bool force, palloc_segments_tld_t* tld);
void       _palloc_segment_page_abandon(palloc_page_t* page, palloc_segments_tld_t* tld);
bool       _palloc_segment_try_reclaim_abandoned( palloc_heap_t* heap, bool try_all, palloc_segments_tld_t* tld);
void       _palloc_segment_collect(palloc_segment_t* segment, bool force);

#if PALLOC_HUGE_PAGE_ABANDON
void        _palloc_segment_huge_page_free(palloc_segment_t* segment, palloc_page_t* page, palloc_block_t* block);
#else
void        _palloc_segment_huge_page_reset(palloc_segment_t* segment, palloc_page_t* page, palloc_block_t* block);
#endif

uint8_t*   _palloc_segment_page_start(const palloc_segment_t* segment, const palloc_page_t* page, size_t* page_size); // page start for any page
void       _palloc_abandoned_reclaim_all(palloc_heap_t* heap, palloc_segments_tld_t* tld);
void       _palloc_abandoned_collect(palloc_heap_t* heap, bool force, palloc_segments_tld_t* tld);
bool       _palloc_segment_attempt_reclaim(palloc_heap_t* heap, palloc_segment_t* segment);
bool       _palloc_segment_visit_blocks(palloc_segment_t* segment, int heap_tag, bool visit_blocks, palloc_block_visit_fun* visitor, void* arg);

// "page.c"
void*       _palloc_malloc_generic(palloc_heap_t* heap, size_t size, bool zero, size_t huge_alignment, size_t* usable)  palloc_attr_noexcept palloc_attr_malloc;

void        _palloc_page_retire(palloc_page_t* page) palloc_attr_noexcept;                  // free the page if there are no other pages with many free blocks
void        _palloc_page_unfull(palloc_page_t* page);
void        _palloc_page_free(palloc_page_t* page, palloc_page_queue_t* pq, bool force);   // free the page
void        _palloc_page_abandon(palloc_page_t* page, palloc_page_queue_t* pq);            // abandon the page, to be picked up by another thread...
void        _palloc_page_force_abandon(palloc_page_t* page);

void        _palloc_heap_delayed_free_all(palloc_heap_t* heap);
bool        _palloc_heap_delayed_free_partial(palloc_heap_t* heap);
void        _palloc_heap_collect_retired(palloc_heap_t* heap, bool force);

void        _palloc_page_use_delayed_free(palloc_page_t* page, palloc_delayed_t delay, bool override_never);
bool        _palloc_page_try_use_delayed_free(palloc_page_t* page, palloc_delayed_t delay, bool override_never);
size_t      _palloc_page_queue_append(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_queue_t* append);
void        _palloc_deferred_free(palloc_heap_t* heap, bool force);

void        _palloc_page_free_collect(palloc_page_t* page,bool force);
void        _palloc_page_reclaim(palloc_heap_t* heap, palloc_page_t* page);   // callback from segments

size_t      _palloc_page_stats_bin(const palloc_page_t* page); // for stats
size_t      _palloc_bin_size(size_t bin);                  // for stats
size_t      _palloc_bin(size_t size);                      // for stats

// "heap.c"
void        _palloc_heap_init(palloc_heap_t* heap, palloc_tld_t* tld, palloc_arena_id_t arena_id, bool noreclaim, uint8_t tag);
void        _palloc_heap_destroy_pages(palloc_heap_t* heap);
void        _palloc_heap_collect_abandon(palloc_heap_t* heap);
void        _palloc_heap_set_default_direct(palloc_heap_t* heap);
bool        _palloc_heap_memid_is_suitable(palloc_heap_t* heap, palloc_memid_t memid);
void        _palloc_heap_unsafe_destroy_all(palloc_heap_t* heap);
palloc_heap_t*  _palloc_heap_by_tag(palloc_heap_t* heap, uint8_t tag);
void        _palloc_heap_area_init(palloc_heap_area_t* area, palloc_page_t* page);
bool        _palloc_heap_area_visit_blocks(const palloc_heap_area_t* area, palloc_page_t* page, palloc_block_visit_fun* visitor, void* arg);

// "stats.c"
void        _palloc_stats_done(palloc_stats_t* stats);
void        _palloc_stats_merge_thread(palloc_tld_t* tld);
palloc_msecs_t  _palloc_clock_now(void);
palloc_msecs_t  _palloc_clock_end(palloc_msecs_t start);
palloc_msecs_t  _palloc_clock_start(void);

// "alloc.c"
void*       _palloc_page_malloc_zero(palloc_heap_t* heap, palloc_page_t* page, size_t size, bool zero, size_t* usable) palloc_attr_noexcept;  // called from `_palloc_malloc_generic`
void*       _palloc_page_malloc(palloc_heap_t* heap, palloc_page_t* page, size_t size) palloc_attr_noexcept;                  // called from `_palloc_heap_malloc_aligned`
void*       _palloc_page_malloc_zeroed(palloc_heap_t* heap, palloc_page_t* page, size_t size) palloc_attr_noexcept;           // called from `_palloc_heap_malloc_aligned`
void*       _palloc_heap_malloc_zero(palloc_heap_t* heap, size_t size, bool zero) palloc_attr_noexcept;
void*       _palloc_heap_malloc_zero_ex(palloc_heap_t* heap, size_t size, bool zero, size_t huge_alignment, size_t* usable) palloc_attr_noexcept;     // called from `_palloc_heap_malloc_aligned`
void*       _palloc_heap_realloc_zero(palloc_heap_t* heap, void* p, size_t newsize, bool zero, size_t* usable_pre, size_t* usable_post) palloc_attr_noexcept;
palloc_block_t* _palloc_page_ptr_unalign(const palloc_page_t* page, const void* p);
bool        _palloc_free_delayed_block(palloc_block_t* block);
void        _palloc_free_generic(palloc_segment_t* segment, palloc_page_t* page, bool is_local, void* p) palloc_attr_noexcept;  // for runtime integration
void        _palloc_padding_shrink(const palloc_page_t* page, const palloc_block_t* block, const size_t min_size);

#if PALLOC_DEBUG>1
bool        _palloc_page_is_valid(palloc_page_t* page);
#endif


/* -----------------------------------------------------------
  Error codes passed to `_palloc_fatal_error`
  All are recoverable but EFAULT is a serious error and aborts by default in secure mode.
  For portability define undefined error codes using common Unix codes:
  <https://www-numi.fnal.gov/offline_software/srt_public_context/WebDocs/Errors/unix_system_errors.html>
----------------------------------------------------------- */
#include <errno.h>
#ifndef EAGAIN         // double free
#define EAGAIN (11)
#endif
#ifndef ENOMEM         // out of memory
#define ENOMEM (12)
#endif
#ifndef EFAULT         // corrupted free-list or meta-data
#define EFAULT (14)
#endif
#ifndef EINVAL         // trying to free an invalid pointer
#define EINVAL (22)
#endif
#ifndef EOVERFLOW      // count*size overflow
#define EOVERFLOW (75)
#endif


// ------------------------------------------------------
// Assertions
// ------------------------------------------------------

#if (PALLOC_DEBUG)
// use our own assertion to print without memory allocation
palloc_decl_noreturn palloc_decl_cold void _palloc_assert_fail(const char* assertion, const char* fname, unsigned int line, const char* func) palloc_attr_noexcept;
#define palloc_assert(expr)     ((expr) ? (void)0 : _palloc_assert_fail(#expr,__FILE__,__LINE__,__func__))
#else
#define palloc_assert(x)
#endif

#if (PALLOC_DEBUG>1)
#define palloc_assert_internal    palloc_assert
#else
#define palloc_assert_internal(x)
#endif

#if (PALLOC_DEBUG>2)
#define palloc_assert_expensive   palloc_assert
#else
#define palloc_assert_expensive(x)
#endif



/* -----------------------------------------------------------
  Inlined definitions
----------------------------------------------------------- */
#define PALLOC_UNUSED(x)     (void)(x)
#if (PALLOC_DEBUG>0)
#define PALLOC_UNUSED_RELEASE(x)
#else
#define PALLOC_UNUSED_RELEASE(x)  PALLOC_UNUSED(x)
#endif

#define PALLOC_INIT4(x)   x(),x(),x(),x()
#define PALLOC_INIT8(x)   PALLOC_INIT4(x),PALLOC_INIT4(x)
#define PALLOC_INIT16(x)  PALLOC_INIT8(x),PALLOC_INIT8(x)
#define PALLOC_INIT32(x)  PALLOC_INIT16(x),PALLOC_INIT16(x)
#define PALLOC_INIT64(x)  PALLOC_INIT32(x),PALLOC_INIT32(x)
#define PALLOC_INIT128(x) PALLOC_INIT64(x),PALLOC_INIT64(x)
#define PALLOC_INIT256(x) PALLOC_INIT128(x),PALLOC_INIT128(x)
#define PALLOC_INIT74(x)  PALLOC_INIT64(x),PALLOC_INIT8(x),x(),x()

#include <string.h>
// initialize a local variable to zero; use memset as compilers optimize constant sized memset's
#define _palloc_memzero_var(x)  memset(&x,0,sizeof(x))

// Is `x` a power of two? (0 is considered a power of two)
static inline bool _palloc_is_power_of_two(uintptr_t x) {
  return ((x & (x - 1)) == 0);
}

// Is a pointer aligned?
static inline bool _palloc_is_aligned(void* p, size_t alignment) {
  palloc_assert_internal(alignment != 0);
  return (((uintptr_t)p % alignment) == 0);
}

// Align upwards
static inline uintptr_t _palloc_align_up(uintptr_t sz, size_t alignment) {
  palloc_assert_internal(alignment != 0);
  uintptr_t mask = alignment - 1;
  if ((alignment & mask) == 0) {  // power of two?
    return ((sz + mask) & ~mask);
  }
  else {
    return (((sz + mask)/alignment)*alignment);
  }
}

// Align downwards
static inline uintptr_t _palloc_align_down(uintptr_t sz, size_t alignment) {
  palloc_assert_internal(alignment != 0);
  uintptr_t mask = alignment - 1;
  if ((alignment & mask) == 0) { // power of two?
    return (sz & ~mask);
  }
  else {
    return ((sz / alignment) * alignment);
  }
}

// Align a pointer upwards
static inline void* palloc_align_up_ptr(void* p, size_t alignment) {
  return (void*)_palloc_align_up((uintptr_t)p, alignment);
}

// Align a pointer downwards
static inline void* palloc_align_down_ptr(void* p, size_t alignment) {
  return (void*)_palloc_align_down((uintptr_t)p, alignment);
}


// Divide upwards: `s <= _palloc_divide_up(s,d)*d < s+d`.
static inline uintptr_t _palloc_divide_up(uintptr_t size, size_t divider) {
  palloc_assert_internal(divider != 0);
  return (divider == 0 ? size : ((size + divider - 1) / divider));
}


// clamp an integer
static inline size_t _palloc_clamp(size_t sz, size_t min, size_t max) {
  if (sz < min) return min;
  else if (sz > max) return max;
  else return sz;
}

// Is memory zero initialized?
static inline bool palloc_mem_is_zero(const void* p, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (((uint8_t*)p)[i] != 0) return false;
  }
  return true;
}


// Align a byte size to a size in _machine words_,
// i.e. byte size == `wsize*sizeof(void*)`.
static inline size_t _palloc_wsize_from_size(size_t size) {
  palloc_assert_internal(size <= SIZE_MAX - sizeof(uintptr_t));
  return (size + sizeof(uintptr_t) - 1) / sizeof(uintptr_t);
}

// Overflow detecting multiply
#if __has_builtin(__builtin_umul_overflow) || (defined(__GNUC__) && (__GNUC__ >= 5))
#include <limits.h>      // UINT_MAX, ULONG_MAX
#if defined(_CLOCK_T)    // for Illumos
#undef _CLOCK_T
#endif
static inline bool palloc_mul_overflow(size_t count, size_t size, size_t* total) {
  #if (SIZE_MAX == ULONG_MAX)
    return __builtin_umull_overflow(count, size, (unsigned long *)total);
  #elif (SIZE_MAX == UINT_MAX)
    return __builtin_umul_overflow(count, size, (unsigned int *)total);
  #else
    return __builtin_umulll_overflow(count, size, (unsigned long long *)total);
  #endif
}
#else /* __builtin_umul_overflow is unavailable */
static inline bool palloc_mul_overflow(size_t count, size_t size, size_t* total) {
  #define PALLOC_MUL_COULD_OVERFLOW ((size_t)1 << (4*sizeof(size_t)))  // sqrt(SIZE_MAX)
  *total = count * size;
  // note: gcc/clang optimize this to directly check the overflow flag
  return ((size >= PALLOC_MUL_COULD_OVERFLOW || count >= PALLOC_MUL_COULD_OVERFLOW) && size > 0 && (SIZE_MAX / size) < count);
}
#endif

// Safe multiply `count*size` into `total`; return `true` on overflow.
static inline bool palloc_count_size_overflow(size_t count, size_t size, size_t* total) {
  if (count==1) {  // quick check for the case where count is one (common for C++ allocators)
    *total = size;
    return false;
  }
  else if palloc_unlikely(palloc_mul_overflow(count, size, total)) {
    #if PALLOC_DEBUG > 0
    _palloc_error_message(EOVERFLOW, "allocation request is too large (%zu * %zu bytes)\n", count, size);
    #endif
    *total = SIZE_MAX;
    return true;
  }
  else return false;
}


/*----------------------------------------------------------------------------------------
  Heap functions
------------------------------------------------------------------------------------------- */

extern palloc_decl_hidden const palloc_heap_t _palloc_heap_empty;  // read-only empty heap, initial value of the thread local default heap

static inline bool palloc_heap_is_backing(const palloc_heap_t* heap) {
  return (heap->tld->heap_backing == heap);
}

static inline bool palloc_heap_is_initialized(palloc_heap_t* heap) {
  palloc_assert_internal(heap != NULL);
  return (heap != NULL && heap != &_palloc_heap_empty);
}

static inline uintptr_t _palloc_ptr_cookie(const void* p) {
  extern palloc_decl_hidden palloc_heap_t _palloc_heap_main;
  palloc_assert_internal(_palloc_heap_main.cookie != 0);
  return ((uintptr_t)p ^ _palloc_heap_main.cookie);
}

/* -----------------------------------------------------------
  Pages
----------------------------------------------------------- */

static inline palloc_page_t* _palloc_heap_get_free_small_page(palloc_heap_t* heap, size_t size) {
  palloc_assert_internal(size <= (PALLOC_SMALL_SIZE_MAX + PALLOC_PADDING_SIZE));
  const size_t idx = _palloc_wsize_from_size(size);
  palloc_assert_internal(idx < PALLOC_PAGES_DIRECT);
  return heap->pages_free_direct[idx];
}

// Segment that contains the pointer
// Large aligned blocks may be aligned at N*PALLOC_SEGMENT_SIZE (inside a huge segment > PALLOC_SEGMENT_SIZE),
// and we need align "down" to the segment info which is `PALLOC_SEGMENT_SIZE` bytes before it;
// therefore we align one byte before `p`.
// We check for NULL afterwards on 64-bit systems to improve codegen for `palloc_free`.
static inline palloc_segment_t* _palloc_ptr_segment(const void* p) {
  palloc_segment_t* const segment = (palloc_segment_t*)(((uintptr_t)p - 1) & ~PALLOC_SEGMENT_MASK);
  #if PALLOC_INTPTR_SIZE <= 4
  return (p==NULL ? NULL : segment);
  #else
  return ((intptr_t)segment <= 0 ? NULL : segment);
  #endif
}

static inline palloc_page_t* palloc_slice_to_page(palloc_slice_t* s) {
  palloc_assert_internal(s->slice_offset== 0 && s->slice_count > 0);
  return (palloc_page_t*)(s);
}

static inline palloc_slice_t* palloc_page_to_slice(palloc_page_t* p) {
  palloc_assert_internal(p->slice_offset== 0 && p->slice_count > 0);
  return (palloc_slice_t*)(p);
}

// Segment belonging to a page
static inline palloc_segment_t* _palloc_page_segment(const palloc_page_t* page) {
  palloc_assert_internal(page!=NULL);
  palloc_segment_t* segment = _palloc_ptr_segment(page);
  palloc_assert_internal(segment == NULL || ((palloc_slice_t*)page >= segment->slices && (palloc_slice_t*)page < segment->slices + segment->slice_entries));
  return segment;
}

static inline palloc_slice_t* palloc_slice_first(const palloc_slice_t* slice) {
  palloc_slice_t* start = (palloc_slice_t*)((uint8_t*)slice - slice->slice_offset);
  palloc_assert_internal(start >= _palloc_ptr_segment(slice)->slices);
  palloc_assert_internal(start->slice_offset == 0);
  palloc_assert_internal(start + start->slice_count > slice);
  return start;
}

// Get the page containing the pointer (performance critical as it is called in palloc_free)
static inline palloc_page_t* _palloc_segment_page_of(const palloc_segment_t* segment, const void* p) {
  palloc_assert_internal(p > (void*)segment);
  ptrdiff_t diff = (uint8_t*)p - (uint8_t*)segment;
  palloc_assert_internal(diff > 0 && diff <= (ptrdiff_t)PALLOC_SEGMENT_SIZE);
  size_t idx = (size_t)diff >> PALLOC_SEGMENT_SLICE_SHIFT;
  palloc_assert_internal(idx <= segment->slice_entries);
  palloc_slice_t* slice0 = (palloc_slice_t*)&segment->slices[idx];
  palloc_slice_t* slice = palloc_slice_first(slice0);  // adjust to the block that holds the page data
  palloc_assert_internal(slice->slice_offset == 0);
  palloc_assert_internal(slice >= segment->slices && slice < segment->slices + segment->slice_entries);
  return palloc_slice_to_page(slice);
}

// Quick page start for initialized pages
static inline uint8_t* palloc_page_start(const palloc_page_t* page) {
  palloc_assert_internal(page->page_start != NULL);
  palloc_assert_expensive(_palloc_segment_page_start(_palloc_page_segment(page),page,NULL) == page->page_start);
  return page->page_start;
}

// Get the page containing the pointer
static inline palloc_page_t* _palloc_ptr_page(void* p) {
  palloc_assert_internal(p!=NULL);
  return _palloc_segment_page_of(_palloc_ptr_segment(p), p);
}

// Get the block size of a page (special case for huge objects)
static inline size_t palloc_page_block_size(const palloc_page_t* page) {
  palloc_assert_internal(page->block_size > 0);
  return page->block_size;
}

static inline bool palloc_page_is_huge(const palloc_page_t* page) {
  palloc_assert_internal((page->is_huge && _palloc_page_segment(page)->kind == PALLOC_SEGMENT_HUGE) ||
                     (!page->is_huge && _palloc_page_segment(page)->kind != PALLOC_SEGMENT_HUGE));
  return page->is_huge;
}

// Get the usable block size of a page without fixed padding.
// This may still include internal padding due to alignment and rounding up size classes.
static inline size_t palloc_page_usable_block_size(const palloc_page_t* page) {
  return palloc_page_block_size(page) - PALLOC_PADDING_SIZE;
}

// size of a segment
static inline size_t palloc_segment_size(palloc_segment_t* segment) {
  return segment->segment_slices * PALLOC_SEGMENT_SLICE_SIZE;
}

static inline uint8_t* palloc_segment_end(palloc_segment_t* segment) {
  return (uint8_t*)segment + palloc_segment_size(segment);
}

// Thread free access
static inline palloc_block_t* palloc_page_thread_free(const palloc_page_t* page) {
  return (palloc_block_t*)(palloc_atomic_load_relaxed(&((palloc_page_t*)page)->xthread_free) & ~3);
}

static inline palloc_delayed_t palloc_page_thread_free_flag(const palloc_page_t* page) {
  return (palloc_delayed_t)(palloc_atomic_load_relaxed(&((palloc_page_t*)page)->xthread_free) & 3);
}

// Heap access
static inline palloc_heap_t* palloc_page_heap(const palloc_page_t* page) {
  return (palloc_heap_t*)(palloc_atomic_load_relaxed(&((palloc_page_t*)page)->xheap));
}

static inline void palloc_page_set_heap(palloc_page_t* page, palloc_heap_t* heap) {
  palloc_assert_internal(palloc_page_thread_free_flag(page) != PALLOC_DELAYED_FREEING);
  palloc_atomic_store_release(&page->xheap,(uintptr_t)heap);
  if (heap != NULL) { page->heap_tag = heap->tag; }
}

// Thread free flag helpers
static inline palloc_block_t* palloc_tf_block(palloc_thread_free_t tf) {
  return (palloc_block_t*)(tf & ~0x03);
}
static inline palloc_delayed_t palloc_tf_delayed(palloc_thread_free_t tf) {
  return (palloc_delayed_t)(tf & 0x03);
}
static inline palloc_thread_free_t palloc_tf_make(palloc_block_t* block, palloc_delayed_t delayed) {
  return (palloc_thread_free_t)((uintptr_t)block | (uintptr_t)delayed);
}
static inline palloc_thread_free_t palloc_tf_set_delayed(palloc_thread_free_t tf, palloc_delayed_t delayed) {
  return palloc_tf_make(palloc_tf_block(tf),delayed);
}
static inline palloc_thread_free_t palloc_tf_set_block(palloc_thread_free_t tf, palloc_block_t* block) {
  return palloc_tf_make(block, palloc_tf_delayed(tf));
}

// are all blocks in a page freed?
// note: needs up-to-date used count, (as the `xthread_free` list may not be empty). see `_palloc_page_collect_free`.
static inline bool palloc_page_all_free(const palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  return (page->used == 0);
}

// are there any available blocks?
static inline bool palloc_page_has_any_available(const palloc_page_t* page) {
  palloc_assert_internal(page != NULL && page->reserved > 0);
  return (page->used < page->reserved || (palloc_page_thread_free(page) != NULL));
}

// are there immediately available blocks, i.e. blocks available on the free list.
static inline bool palloc_page_immediate_available(const palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  return (page->free != NULL);
}

// is more than 7/8th of a page in use?
static inline bool palloc_page_is_mostly_used(const palloc_page_t* page) {
  if (page==NULL) return true;
  uint16_t frac = page->reserved / 8U;
  return (page->reserved - page->used <= frac);
}

static inline palloc_page_queue_t* palloc_page_queue(const palloc_heap_t* heap, size_t size) {
  return &((palloc_heap_t*)heap)->pages[_palloc_bin(size)];
}



//-----------------------------------------------------------
// Page flags
//-----------------------------------------------------------
static inline bool palloc_page_is_in_full(const palloc_page_t* page) {
  return page->flags.x.in_full;
}

static inline void palloc_page_set_in_full(palloc_page_t* page, bool in_full) {
  page->flags.x.in_full = in_full;
}

static inline bool palloc_page_has_aligned(const palloc_page_t* page) {
  return page->flags.x.has_aligned;
}

static inline void palloc_page_set_has_aligned(palloc_page_t* page, bool has_aligned) {
  page->flags.x.has_aligned = has_aligned;
}

/* -------------------------------------------------------------------
  Guarded objects
------------------------------------------------------------------- */
#if PALLOC_GUARDED
static inline bool palloc_block_ptr_is_guarded(const palloc_block_t* block, const void* p) {
  const ptrdiff_t offset = (uint8_t*)p - (uint8_t*)block;
  return (offset >= (ptrdiff_t)(sizeof(palloc_block_t)) && block->next == PALLOC_BLOCK_TAG_GUARDED);
}

static inline bool palloc_heap_malloc_use_guarded(palloc_heap_t* heap, size_t size) {
  // this code is written to result in fast assembly as it is on the hot path for allocation
  const size_t count = heap->guarded_sample_count - 1;  // if the rate was 0, this will underflow and count for a long time..
  if palloc_likely(count != 0) {
    // no sample
    heap->guarded_sample_count = count;
    return false;
  }
  else if (size >= heap->guarded_size_min && size <= heap->guarded_size_max) {
    // use guarded allocation
    heap->guarded_sample_count = heap->guarded_sample_rate;  // reset
    return (heap->guarded_sample_rate != 0);
  }
  else {
    // failed size criteria, rewind count (but don't write to an empty heap)
    if (heap->guarded_sample_rate != 0) { heap->guarded_sample_count = 1; }
    return false;
  }
}

palloc_decl_restrict void* _palloc_heap_malloc_guarded(palloc_heap_t* heap, size_t size, bool zero) palloc_attr_noexcept;

#endif


/* -------------------------------------------------------------------
Encoding/Decoding the free list next pointers

This is to protect against buffer overflow exploits where the
free list is mutated. Many hardened allocators xor the next pointer `p`
with a secret key `k1`, as `p^k1`. This prevents overwriting with known
values but might be still too weak: if the attacker can guess
the pointer `p` this  can reveal `k1` (since `p^k1^p == k1`).
Moreover, if multiple blocks can be read as well, the attacker can
xor both as `(p1^k1) ^ (p2^k1) == p1^p2` which may reveal a lot
about the pointers (and subsequently `k1`).

Instead palloc uses an extra key `k2` and encodes as `((p^k2)<<<k1)+k1`.
Since these operations are not associative, the above approaches do not
work so well any more even if the `p` can be guesstimated. For example,
for the read case we can subtract two entries to discard the `+k1` term,
but that leads to `((p1^k2)<<<k1) - ((p2^k2)<<<k1)` at best.
We include the left-rotation since xor and addition are otherwise linear
in the lowest bit. Finally, both keys are unique per page which reduces
the re-use of keys by a large factor.

We also pass a separate `null` value to be used as `NULL` or otherwise
`(k2<<<k1)+k1` would appear (too) often as a sentinel value.
------------------------------------------------------------------- */

static inline bool palloc_is_in_same_segment(const void* p, const void* q) {
  return (_palloc_ptr_segment(p) == _palloc_ptr_segment(q));
}

static inline bool palloc_is_in_same_page(const void* p, const void* q) {
  palloc_segment_t* segment = _palloc_ptr_segment(p);
  if (_palloc_ptr_segment(q) != segment) return false;
  // assume q may be invalid // return (_palloc_segment_page_of(segment, p) == _palloc_segment_page_of(segment, q));
  palloc_page_t* page = _palloc_segment_page_of(segment, p);
  size_t psize;
  uint8_t* start = _palloc_segment_page_start(segment, page, &psize);
  return (start <= (uint8_t*)q && (uint8_t*)q < start + psize);
}

static inline uintptr_t palloc_rotl(uintptr_t x, uintptr_t shift) {
  shift %= PALLOC_INTPTR_BITS;
  return (shift==0 ? x : ((x << shift) | (x >> (PALLOC_INTPTR_BITS - shift))));
}
static inline uintptr_t palloc_rotr(uintptr_t x, uintptr_t shift) {
  shift %= PALLOC_INTPTR_BITS;
  return (shift==0 ? x : ((x >> shift) | (x << (PALLOC_INTPTR_BITS - shift))));
}

static inline void* palloc_ptr_decode(const void* null, const palloc_encoded_t x, const uintptr_t* keys) {
  void* p = (void*)(palloc_rotr(x - keys[0], keys[0]) ^ keys[1]);
  return (p==null ? NULL : p);
}

static inline palloc_encoded_t palloc_ptr_encode(const void* null, const void* p, const uintptr_t* keys) {
  uintptr_t x = (uintptr_t)(p==NULL ? null : p);
  return palloc_rotl(x ^ keys[1], keys[0]) + keys[0];
}

static inline uint32_t palloc_ptr_encode_canary(const void* null, const void* p, const uintptr_t* keys) {
  const uint32_t x = (uint32_t)(palloc_ptr_encode(null,p,keys));
  // make the lowest byte 0 to prevent spurious read overflows which could be a security issue (issue #951)
  #ifdef PALLOC_BIG_ENDIAN
  return (x & 0x00FFFFFF);
  #else
  return (x & 0xFFFFFF00);
  #endif
}

static inline palloc_block_t* palloc_block_nextx( const void* null, const palloc_block_t* block, const uintptr_t* keys ) {
  palloc_track_mem_defined(block,sizeof(palloc_block_t));
  palloc_block_t* next;
  #ifdef PALLOC_ENCODE_FREELIST
  next = (palloc_block_t*)palloc_ptr_decode(null, block->next, keys);
  #else
  PALLOC_UNUSED(keys); PALLOC_UNUSED(null);
  next = (palloc_block_t*)block->next;
  #endif
  palloc_track_mem_noaccess(block,sizeof(palloc_block_t));
  return next;
}

static inline void palloc_block_set_nextx(const void* null, palloc_block_t* block, const palloc_block_t* next, const uintptr_t* keys) {
  palloc_track_mem_undefined(block,sizeof(palloc_block_t));
  #ifdef PALLOC_ENCODE_FREELIST
  block->next = palloc_ptr_encode(null, next, keys);
  #else
  PALLOC_UNUSED(keys); PALLOC_UNUSED(null);
  block->next = (palloc_encoded_t)next;
  #endif
  palloc_track_mem_noaccess(block,sizeof(palloc_block_t));
}

static inline palloc_block_t* palloc_block_next(const palloc_page_t* page, const palloc_block_t* block) {
  #ifdef PALLOC_ENCODE_FREELIST
  palloc_block_t* next = palloc_block_nextx(page,block,page->keys);
  // check for free list corruption: is `next` at least in the same page?
  // TODO: check if `next` is `page->block_size` aligned?
  if palloc_unlikely(next!=NULL && !palloc_is_in_same_page(block, next)) {
    _palloc_error_message(EFAULT, "corrupted free list entry of size %zub at %p: value 0x%zx\n", palloc_page_block_size(page), block, (uintptr_t)next);
    next = NULL;
  }
  return next;
  #else
  PALLOC_UNUSED(page);
  return palloc_block_nextx(page,block,NULL);
  #endif
}

static inline void palloc_block_set_next(const palloc_page_t* page, palloc_block_t* block, const palloc_block_t* next) {
  #ifdef PALLOC_ENCODE_FREELIST
  palloc_block_set_nextx(page,block,next, page->keys);
  #else
  PALLOC_UNUSED(page);
  palloc_block_set_nextx(page,block,next,NULL);
  #endif
}


// -------------------------------------------------------------------
// commit mask
// -------------------------------------------------------------------

static inline void palloc_commit_mask_create_empty(palloc_commit_mask_t* cm) {
  for (size_t i = 0; i < PALLOC_COMMIT_MASK_FIELD_COUNT; i++) {
    cm->mask[i] = 0;
  }
}

static inline void palloc_commit_mask_create_full(palloc_commit_mask_t* cm) {
  for (size_t i = 0; i < PALLOC_COMMIT_MASK_FIELD_COUNT; i++) {
    cm->mask[i] = ~((size_t)0);
  }
}

static inline bool palloc_commit_mask_is_empty(const palloc_commit_mask_t* cm) {
  for (size_t i = 0; i < PALLOC_COMMIT_MASK_FIELD_COUNT; i++) {
    if (cm->mask[i] != 0) return false;
  }
  return true;
}

static inline bool palloc_commit_mask_is_full(const palloc_commit_mask_t* cm) {
  for (size_t i = 0; i < PALLOC_COMMIT_MASK_FIELD_COUNT; i++) {
    if (cm->mask[i] != ~((size_t)0)) return false;
  }
  return true;
}

// defined in `segment.c`:
size_t _palloc_commit_mask_committed_size(const palloc_commit_mask_t* cm, size_t total);
size_t _palloc_commit_mask_next_run(const palloc_commit_mask_t* cm, size_t* idx);

#define palloc_commit_mask_foreach(cm,idx,count) \
  idx = 0; \
  while ((count = _palloc_commit_mask_next_run(cm,&idx)) > 0) {

#define palloc_commit_mask_foreach_end() \
    idx += count; \
  }



/* -----------------------------------------------------------
  memory id's
----------------------------------------------------------- */

static inline palloc_memid_t _palloc_memid_create(palloc_memkind_t memkind) {
  palloc_memid_t memid;
  _palloc_memzero_var(memid);
  memid.memkind = memkind;
  return memid;
}

static inline palloc_memid_t _palloc_memid_none(void) {
  return _palloc_memid_create(PALLOC_MEM_NONE);
}

static inline palloc_memid_t _palloc_memid_create_os(void* base, size_t size, bool committed, bool is_zero, bool is_large) {
  palloc_memid_t memid = _palloc_memid_create(PALLOC_MEM_OS);
  memid.mem.os.base = base;
  memid.mem.os.size = size;
  memid.initially_committed = committed;
  memid.initially_zero = is_zero;
  memid.is_pinned = is_large;
  return memid;
}


// -------------------------------------------------------------------
// Fast "random" shuffle
// -------------------------------------------------------------------

static inline uintptr_t _palloc_random_shuffle(uintptr_t x) {
  if (x==0) { x = 17; }   // ensure we don't get stuck in generating zeros
#if (PALLOC_INTPTR_SIZE>=8)
  // by Sebastiano Vigna, see: <http://xoshiro.di.unimi.it/splitmix64.c>
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9UL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebUL;
  x ^= x >> 31;
#elif (PALLOC_INTPTR_SIZE==4)
  // by Chris Wellons, see: <https://nullprogram.com/blog/2018/07/31/>
  x ^= x >> 16;
  x *= 0x7feb352dUL;
  x ^= x >> 15;
  x *= 0x846ca68bUL;
  x ^= x >> 16;
#endif
  return x;
}



// -----------------------------------------------------------------------
// Count bits: trailing or leading zeros (with PALLOC_INTPTR_BITS on all zero)
// -----------------------------------------------------------------------

#if defined(__GNUC__)

#include <limits.h>       // LONG_MAX
#define PALLOC_HAVE_FAST_BITSCAN
static inline size_t palloc_clz(size_t x) {
  if (x==0) return PALLOC_SIZE_BITS;
  #if (SIZE_MAX == ULONG_MAX)
    return __builtin_clzl(x);
  #else
    return __builtin_clzll(x);
  #endif
}
static inline size_t palloc_ctz(size_t x) {
  if (x==0) return PALLOC_SIZE_BITS;
  #if (SIZE_MAX == ULONG_MAX)
    return __builtin_ctzl(x);
  #else
    return __builtin_ctzll(x);
  #endif
}

#elif defined(_MSC_VER)

#include <limits.h>       // LONG_MAX
#include <intrin.h>       // BitScanReverse64
#define PALLOC_HAVE_FAST_BITSCAN
static inline size_t palloc_clz(size_t x) {
  if (x==0) return PALLOC_SIZE_BITS;
  unsigned long idx;
  #if (SIZE_MAX == ULONG_MAX)
    _BitScanReverse(&idx, x);
  #else
    _BitScanReverse64(&idx, x);
  #endif
  return ((PALLOC_SIZE_BITS - 1) - (size_t)idx);
}
static inline size_t palloc_ctz(size_t x) {
  if (x==0) return PALLOC_SIZE_BITS;
  unsigned long idx;
  #if (SIZE_MAX == ULONG_MAX)
    _BitScanForward(&idx, x);
  #else
    _BitScanForward64(&idx, x);
  #endif
  return (size_t)idx;
}

#else

static inline size_t palloc_ctz_generic32(uint32_t x) {
  // de Bruijn multiplication, see <http://supertech.csail.mit.edu/papers/debruijn.pdf>
  static const uint8_t debruijn[32] = {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
  };
  if (x==0) return 32;
  return debruijn[(uint32_t)((x & -(int32_t)x) * (uint32_t)(0x077CB531U)) >> 27];
}

static inline size_t palloc_clz_generic32(uint32_t x) {
  // de Bruijn multiplication, see <http://supertech.csail.mit.edu/papers/debruijn.pdf>
  static const uint8_t debruijn[32] = {
    31, 22, 30, 21, 18, 10, 29, 2, 20, 17, 15, 13, 9, 6, 28, 1,
    23, 19, 11, 3, 16, 14, 7, 24, 12, 4, 8, 25, 5, 26, 27, 0
  };
  if (x==0) return 32;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return debruijn[(uint32_t)(x * (uint32_t)(0x07C4ACDDU)) >> 27];
}

static inline size_t palloc_ctz(size_t x) {
  if (x==0) return PALLOC_SIZE_BITS;
  #if (PALLOC_SIZE_BITS <= 32)
    return palloc_ctz_generic32((uint32_t)x);
  #else
    const uint32_t lo = (uint32_t)x;
    if (lo != 0) {
      return palloc_ctz_generic32(lo);
    }
    else {
      return (32 + palloc_ctz_generic32((uint32_t)(x>>32)));
    }
  #endif
}

static inline size_t palloc_clz(size_t x) {
  if (x==0) return PALLOC_SIZE_BITS;
  #if (PALLOC_SIZE_BITS <= 32)
    return palloc_clz_generic32((uint32_t)x);
  #else
    const uint32_t hi = (uint32_t)(x>>32);
    if (hi != 0) {
      return palloc_clz_generic32(hi);
    }
    else {
      return 32 + palloc_clz_generic32((uint32_t)x);
    }
  #endif
}

#endif

// "bit scan reverse": Return index of the highest bit (or PALLOC_SIZE_BITS if `x` is zero)
static inline size_t palloc_bsr(size_t x) {
  return (x==0 ? PALLOC_SIZE_BITS : PALLOC_SIZE_BITS - 1 - palloc_clz(x));
}

size_t _palloc_popcount_generic(size_t x);

static inline size_t palloc_popcount(size_t x) {
  if (x<=1) return x;
  if (x==SIZE_MAX) return PALLOC_SIZE_BITS;
  #if defined(__GNUC__)
    #if (SIZE_MAX == ULONG_MAX)
      return __builtin_popcountl(x);
    #else
      return __builtin_popcountll(x);
    #endif
  #else
    return _palloc_popcount_generic(x);
  #endif
}

// ---------------------------------------------------------------------------------
// Provide our own `_palloc_memcpy` for potential performance optimizations.
//
// For now, only on Windows with msvc/clang-cl we optimize to `rep movsb` if
// we happen to run on x86/x64 cpu's that have "fast short rep movsb" (FSRM) support
// (AMD Zen3+ (~2020) or Intel Ice Lake+ (~2017). See also issue #201 and pr #253.
// ---------------------------------------------------------------------------------

#if !PALLOC_TRACK_ENABLED && defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
extern palloc_decl_hidden bool _palloc_cpu_has_fsrm;
extern palloc_decl_hidden bool _palloc_cpu_has_erms;
static inline void _palloc_memcpy(void* dst, const void* src, size_t n) {
  if (_palloc_cpu_has_fsrm && n <= 127) { // || (_palloc_cpu_has_erms && n > 128)) {
    __movsb((unsigned char*)dst, (const unsigned char*)src, n);
  }
  else {
    memcpy(dst, src, n);
  }
}
static inline void _palloc_memzero(void* dst, size_t n) {
  if (_palloc_cpu_has_fsrm && n <= 127) { // || (_palloc_cpu_has_erms && n > 128)) {
    __stosb((unsigned char*)dst, 0, n);
  }
  else {
    memset(dst, 0, n);
  }
}
#else
static inline void _palloc_memcpy(void* dst, const void* src, size_t n) {
  memcpy(dst, src, n);
}
static inline void _palloc_memzero(void* dst, size_t n) {
  memset(dst, 0, n);
}
#endif

// -------------------------------------------------------------------------------
// The `_palloc_memcpy_aligned` can be used if the pointers are machine-word aligned
// This is used for example in `palloc_realloc`.
// -------------------------------------------------------------------------------

#if (defined(__GNUC__) && (__GNUC__ >= 4)) || defined(__clang__)
// On GCC/CLang we provide a hint that the pointers are word aligned.
static inline void _palloc_memcpy_aligned(void* dst, const void* src, size_t n) {
  palloc_assert_internal(((uintptr_t)dst % PALLOC_INTPTR_SIZE == 0) && ((uintptr_t)src % PALLOC_INTPTR_SIZE == 0));
  void* adst = __builtin_assume_aligned(dst, PALLOC_INTPTR_SIZE);
  const void* asrc = __builtin_assume_aligned(src, PALLOC_INTPTR_SIZE);
  _palloc_memcpy(adst, asrc, n);
}

static inline void _palloc_memzero_aligned(void* dst, size_t n) {
  palloc_assert_internal((uintptr_t)dst % PALLOC_INTPTR_SIZE == 0);
  void* adst = __builtin_assume_aligned(dst, PALLOC_INTPTR_SIZE);
  _palloc_memzero(adst, n);
}
#else
// Default fallback on `_palloc_memcpy`
static inline void _palloc_memcpy_aligned(void* dst, const void* src, size_t n) {
  palloc_assert_internal(((uintptr_t)dst % PALLOC_INTPTR_SIZE == 0) && ((uintptr_t)src % PALLOC_INTPTR_SIZE == 0));
  _palloc_memcpy(dst, src, n);
}

static inline void _palloc_memzero_aligned(void* dst, size_t n) {
  palloc_assert_internal((uintptr_t)dst % PALLOC_INTPTR_SIZE == 0);
  _palloc_memzero(dst, n);
}
#endif


#endif
