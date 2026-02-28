/* ----------------------------------------------------------------------------
Copyright (c) 2018-2022, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/prim.h"

#include <string.h>  // memcpy, memset
#include <stdlib.h>  // atexit


// Empty page used to initialize the small free pages array
const palloc_page_t _palloc_page_empty = {
  0,
  false, false, false, false,
  0,       // capacity
  0,       // reserved capacity
  { 0 },   // flags
  false,   // is_zero
  0,       // retire_expire
  NULL,    // free
  NULL,    // local_free
  0,       // used
  0,       // block size shift
  0,       // heap tag
  0,       // block_size
  NULL,    // page_start
  #if (PALLOC_PADDING || PALLOC_ENCODE_FREELIST)
  { 0, 0 },
  #endif
  PALLOC_ATOMIC_VAR_INIT(0), // xthread_free
  PALLOC_ATOMIC_VAR_INIT(0), // xheap
  NULL, NULL
  , { 0 }  // padding
};

#define PALLOC_PAGE_EMPTY() ((palloc_page_t*)&_palloc_page_empty)

#if (PALLOC_SMALL_WSIZE_MAX==128)
#if (PALLOC_PADDING>0) && (PALLOC_INTPTR_SIZE >= 8)
#define PALLOC_SMALL_PAGES_EMPTY  { PALLOC_INIT128(PALLOC_PAGE_EMPTY), PALLOC_PAGE_EMPTY(), PALLOC_PAGE_EMPTY() }
#elif (PALLOC_PADDING>0)
#define PALLOC_SMALL_PAGES_EMPTY  { PALLOC_INIT128(PALLOC_PAGE_EMPTY), PALLOC_PAGE_EMPTY(), PALLOC_PAGE_EMPTY(), PALLOC_PAGE_EMPTY() }
#else
#define PALLOC_SMALL_PAGES_EMPTY  { PALLOC_INIT128(PALLOC_PAGE_EMPTY), PALLOC_PAGE_EMPTY() }
#endif
#else
#error "define right initialization sizes corresponding to PALLOC_SMALL_WSIZE_MAX"
#endif

// Empty page queues for every bin
#define QNULL(sz)  { NULL, NULL, (sz)*sizeof(uintptr_t) }
#define PALLOC_PAGE_QUEUES_EMPTY \
  { QNULL(1), \
    QNULL(     1), QNULL(     2), QNULL(     3), QNULL(     4), QNULL(     5), QNULL(     6), QNULL(     7), QNULL(     8), /* 8 */ \
    QNULL(    10), QNULL(    12), QNULL(    14), QNULL(    16), QNULL(    20), QNULL(    24), QNULL(    28), QNULL(    32), /* 16 */ \
    QNULL(    40), QNULL(    48), QNULL(    56), QNULL(    64), QNULL(    80), QNULL(    96), QNULL(   112), QNULL(   128), /* 24 */ \
    QNULL(   160), QNULL(   192), QNULL(   224), QNULL(   256), QNULL(   320), QNULL(   384), QNULL(   448), QNULL(   512), /* 32 */ \
    QNULL(   640), QNULL(   768), QNULL(   896), QNULL(  1024), QNULL(  1280), QNULL(  1536), QNULL(  1792), QNULL(  2048), /* 40 */ \
    QNULL(  2560), QNULL(  3072), QNULL(  3584), QNULL(  4096), QNULL(  5120), QNULL(  6144), QNULL(  7168), QNULL(  8192), /* 48 */ \
    QNULL( 10240), QNULL( 12288), QNULL( 14336), QNULL( 16384), QNULL( 20480), QNULL( 24576), QNULL( 28672), QNULL( 32768), /* 56 */ \
    QNULL( 40960), QNULL( 49152), QNULL( 57344), QNULL( 65536), QNULL( 81920), QNULL( 98304), QNULL(114688), QNULL(131072), /* 64 */ \
    QNULL(163840), QNULL(196608), QNULL(229376), QNULL(262144), QNULL(327680), QNULL(393216), QNULL(458752), QNULL(524288), /* 72 */ \
    QNULL(PALLOC_MEDIUM_OBJ_WSIZE_MAX + 1  /* 655360, Huge queue */), \
    QNULL(PALLOC_MEDIUM_OBJ_WSIZE_MAX + 2) /* Full queue */ }

#define PALLOC_STAT_COUNT_NULL()  {0,0,0}

// Empty statistics
#define PALLOC_STATS_NULL  \
  PALLOC_STAT_COUNT_NULL(), PALLOC_STAT_COUNT_NULL(), PALLOC_STAT_COUNT_NULL(), \
  { 0 }, { 0 }, \
  PALLOC_STAT_COUNT_NULL(), PALLOC_STAT_COUNT_NULL(), PALLOC_STAT_COUNT_NULL(), \
  PALLOC_STAT_COUNT_NULL(), PALLOC_STAT_COUNT_NULL(), PALLOC_STAT_COUNT_NULL(), \
  { 0 }, { 0 }, { 0 }, { 0 }, \
  { 0 }, { 0 }, { 0 }, { 0 }, \
  \
  { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, \
  PALLOC_INIT4(PALLOC_STAT_COUNT_NULL), \
  { 0 }, { 0 }, { 0 }, { 0 },  \
  \
  { PALLOC_INIT4(PALLOC_STAT_COUNT_NULL) }, \
  { { 0 }, { 0 }, { 0 }, { 0 } }, \
  \
  { PALLOC_INIT74(PALLOC_STAT_COUNT_NULL) }, \
  { PALLOC_INIT74(PALLOC_STAT_COUNT_NULL) }


// Empty slice span queues for every bin
#define SQNULL(sz)  { NULL, NULL, sz }
#define PALLOC_SEGMENT_SPAN_QUEUES_EMPTY \
  { SQNULL(1), \
    SQNULL(     1), SQNULL(     2), SQNULL(     3), SQNULL(     4), SQNULL(     5), SQNULL(     6), SQNULL(     7), SQNULL(    10), /*  8 */ \
    SQNULL(    12), SQNULL(    14), SQNULL(    16), SQNULL(    20), SQNULL(    24), SQNULL(    28), SQNULL(    32), SQNULL(    40), /* 16 */ \
    SQNULL(    48), SQNULL(    56), SQNULL(    64), SQNULL(    80), SQNULL(    96), SQNULL(   112), SQNULL(   128), SQNULL(   160), /* 24 */ \
    SQNULL(   192), SQNULL(   224), SQNULL(   256), SQNULL(   320), SQNULL(   384), SQNULL(   448), SQNULL(   512), SQNULL(   640), /* 32 */ \
    SQNULL(   768), SQNULL(   896), SQNULL(  1024) /* 35 */ }


// --------------------------------------------------------
// Statically allocate an empty heap as the initial
// thread local value for the default heap,
// and statically allocate the backing heap for the main
// thread so it can function without doing any allocation
// itself (as accessing a thread local for the first time
// may lead to allocation itself on some platforms)
// --------------------------------------------------------

palloc_decl_cache_align const palloc_heap_t _palloc_heap_empty = {
  NULL,
  PALLOC_ATOMIC_VAR_INIT(NULL),
  0,                // tid
  0,                // cookie
  0,                // arena id
  { 0, 0 },         // keys
  { {0}, {0}, 0, true }, // random
  0,                // page count
  PALLOC_BIN_FULL, 0,   // page retired min/max
  0, 0,             // generic count
  NULL,             // next
  false,            // can reclaim
  0,                // tag
  #if PALLOC_GUARDED
  0, 0, 0, 1,       // count is 1 so we never write to it (see `internal.h:palloc_heap_malloc_use_guarded`)
  #endif
  PALLOC_SMALL_PAGES_EMPTY,
  PALLOC_PAGE_QUEUES_EMPTY
};

static palloc_decl_cache_align palloc_subproc_t palloc_subproc_default;

#define tld_empty_stats  ((palloc_stats_t*)((uint8_t*)&tld_empty + offsetof(palloc_tld_t,stats)))

palloc_decl_cache_align static const palloc_tld_t tld_empty = {
  0,
  false,
  NULL, NULL,
  { PALLOC_SEGMENT_SPAN_QUEUES_EMPTY, 0, 0, 0, 0, 0, &palloc_subproc_default, tld_empty_stats }, // segments
  { sizeof(palloc_stats_t), PALLOC_STAT_VERSION, PALLOC_STATS_NULL }       // stats
};

palloc_threadid_t _palloc_thread_id(void) palloc_attr_noexcept {
  return _palloc_prim_thread_id();
}

// the thread-local default heap for allocation
palloc_decl_thread palloc_heap_t* _palloc_heap_default = (palloc_heap_t*)&_palloc_heap_empty;

extern palloc_decl_hidden palloc_heap_t _palloc_heap_main;

static palloc_decl_cache_align palloc_tld_t tld_main = {
  0, false,
  &_palloc_heap_main, & _palloc_heap_main,
  { PALLOC_SEGMENT_SPAN_QUEUES_EMPTY, 0, 0, 0, 0, 0, &palloc_subproc_default, &tld_main.stats }, // segments
  { sizeof(palloc_stats_t), PALLOC_STAT_VERSION, PALLOC_STATS_NULL }       // stats
};

palloc_decl_cache_align palloc_heap_t _palloc_heap_main = {
  &tld_main,
  PALLOC_ATOMIC_VAR_INIT(NULL),
  0,                // thread id
  0,                // initial cookie
  0,                // arena id
  { 0, 0 },         // the key of the main heap can be fixed (unlike page keys that need to be secure!)
  { {0x846ca68b}, {0}, 0, true },  // random
  0,                // page count
  PALLOC_BIN_FULL, 0,   // page retired min/max
  0, 0,             // generic count
  NULL,             // next heap
  false,            // can reclaim
  0,                // tag
  #if PALLOC_GUARDED
  0, 0, 0, 0,
  #endif
  PALLOC_SMALL_PAGES_EMPTY,
  PALLOC_PAGE_QUEUES_EMPTY
};

bool _palloc_process_is_initialized = false;  // set to `true` in `palloc_process_init`.

palloc_stats_t _palloc_stats_main = { sizeof(palloc_stats_t), PALLOC_STAT_VERSION, PALLOC_STATS_NULL };

#if PALLOC_GUARDED
palloc_decl_export void palloc_heap_guarded_set_sample_rate(palloc_heap_t* heap, size_t sample_rate, size_t seed) {
  heap->guarded_sample_rate  = sample_rate;
  heap->guarded_sample_count = sample_rate;  // count down samples
  if (heap->guarded_sample_rate > 1) {
    if (seed == 0) {
      seed = _palloc_heap_random_next(heap);
    }
    heap->guarded_sample_count = (seed % heap->guarded_sample_rate) + 1;  // start at random count between 1 and `sample_rate`
  }
}

palloc_decl_export void palloc_heap_guarded_set_size_bound(palloc_heap_t* heap, size_t min, size_t max) {
  heap->guarded_size_min = min;
  heap->guarded_size_max = (min > max ? min : max);
}

void _palloc_heap_guarded_init(palloc_heap_t* heap) {
  palloc_heap_guarded_set_sample_rate(heap,
    (size_t)palloc_option_get_clamp(palloc_option_guarded_sample_rate, 0, LONG_MAX),
    (size_t)palloc_option_get(palloc_option_guarded_sample_seed));
  palloc_heap_guarded_set_size_bound(heap,
    (size_t)palloc_option_get_clamp(palloc_option_guarded_min, 0, LONG_MAX),
    (size_t)palloc_option_get_clamp(palloc_option_guarded_max, 0, LONG_MAX) );
}
#else
palloc_decl_export void palloc_heap_guarded_set_sample_rate(palloc_heap_t* heap, size_t sample_rate, size_t seed) {
  PALLOC_UNUSED(heap); PALLOC_UNUSED(sample_rate); PALLOC_UNUSED(seed);
}

palloc_decl_export void palloc_heap_guarded_set_size_bound(palloc_heap_t* heap, size_t min, size_t max) {
  PALLOC_UNUSED(heap); PALLOC_UNUSED(min); PALLOC_UNUSED(max);
}
void _palloc_heap_guarded_init(palloc_heap_t* heap) {
  PALLOC_UNUSED(heap);
}
#endif


static void palloc_heap_main_init(void) {
  if (_palloc_heap_main.cookie == 0) {
    _palloc_heap_main.thread_id = _palloc_thread_id();
    _palloc_heap_main.cookie = 1;
    #if defined(_WIN32) && !defined(PALLOC_SHARED_LIB)
      _palloc_random_init_weak(&_palloc_heap_main.random);    // prevent allocation failure during bcrypt dll initialization with static linking
    #else
      _palloc_random_init(&_palloc_heap_main.random);
    #endif
    _palloc_heap_main.cookie  = _palloc_heap_random_next(&_palloc_heap_main);
    _palloc_heap_main.keys[0] = _palloc_heap_random_next(&_palloc_heap_main);
    _palloc_heap_main.keys[1] = _palloc_heap_random_next(&_palloc_heap_main);
    palloc_lock_init(&palloc_subproc_default.abandoned_os_lock);
    palloc_lock_init(&palloc_subproc_default.abandoned_os_visit_lock);
    _palloc_heap_guarded_init(&_palloc_heap_main);
  }
}

palloc_heap_t* _palloc_heap_main_get(void) {
  palloc_heap_main_init();
  return &_palloc_heap_main;
}

/* -----------------------------------------------------------
  Sub process
----------------------------------------------------------- */

palloc_subproc_id_t palloc_subproc_main(void) {
  return NULL;
}

palloc_subproc_id_t palloc_subproc_new(void) {
  palloc_memid_t memid = _palloc_memid_none();
  palloc_subproc_t* subproc = (palloc_subproc_t*)_palloc_arena_meta_zalloc(sizeof(palloc_subproc_t), &memid);
  if (subproc == NULL) return NULL;
  subproc->memid = memid;
  subproc->abandoned_os_list = NULL;
  palloc_lock_init(&subproc->abandoned_os_lock);
  palloc_lock_init(&subproc->abandoned_os_visit_lock);
  return subproc;
}

palloc_subproc_t* _palloc_subproc_from_id(palloc_subproc_id_t subproc_id) {
  return (subproc_id == NULL ? &palloc_subproc_default : (palloc_subproc_t*)subproc_id);
}

void palloc_subproc_delete(palloc_subproc_id_t subproc_id) {
  if (subproc_id == NULL) return;
  palloc_subproc_t* subproc = _palloc_subproc_from_id(subproc_id);
  // check if there are no abandoned segments still..
  bool safe_to_delete = false;
  palloc_lock(&subproc->abandoned_os_lock) {
    if (subproc->abandoned_os_list == NULL) {
      safe_to_delete = true;
    }
  }
  if (!safe_to_delete) return;
  // safe to release
  // todo: should we refcount subprocesses?
  palloc_lock_done(&subproc->abandoned_os_lock);
  palloc_lock_done(&subproc->abandoned_os_visit_lock);
  _palloc_arena_meta_free(subproc, subproc->memid, sizeof(palloc_subproc_t));
}

void palloc_subproc_add_current_thread(palloc_subproc_id_t subproc_id) {
  palloc_heap_t* heap = palloc_heap_get_default();
  if (heap == NULL) return;
  palloc_assert(heap->tld->segments.subproc == &palloc_subproc_default);
  if (heap->tld->segments.subproc != &palloc_subproc_default) return;
  heap->tld->segments.subproc = _palloc_subproc_from_id(subproc_id);
}



/* -----------------------------------------------------------
  Initialization and freeing of the thread local heaps
----------------------------------------------------------- */

// note: in x64 in release build `sizeof(palloc_thread_data_t)` is under 4KiB (= OS page size).
typedef struct palloc_thread_data_s {
  palloc_heap_t  heap;   // must come first due to cast in `_palloc_heap_done`
  palloc_tld_t   tld;
  palloc_memid_t memid;  // must come last due to zero'ing
} palloc_thread_data_t;


// Thread meta-data is allocated directly from the OS. For
// some programs that do not use thread pools and allocate and
// destroy many OS threads, this may causes too much overhead
// per thread so we maintain a small cache of recently freed metadata.

#define TD_CACHE_SIZE (32)
static _Atomic(palloc_thread_data_t*) td_cache[TD_CACHE_SIZE];

static palloc_thread_data_t* palloc_thread_data_zalloc(void) {
  // try to find thread metadata in the cache
  palloc_thread_data_t* td = NULL;
  for (int i = 0; i < TD_CACHE_SIZE; i++) {
    td = palloc_atomic_load_ptr_relaxed(palloc_thread_data_t, &td_cache[i]);
    if (td != NULL) {
      // found cached allocation, try use it
      td = palloc_atomic_exchange_ptr_acq_rel(palloc_thread_data_t, &td_cache[i], NULL);
      if (td != NULL) {
        _palloc_memzero(td, offsetof(palloc_thread_data_t,memid));
        return td;
      }
    }
  }

  // if that fails, allocate as meta data
  palloc_memid_t memid;
  td = (palloc_thread_data_t*)_palloc_os_zalloc(sizeof(palloc_thread_data_t), &memid);
  if (td == NULL) {
    // if this fails, try once more. (issue #257)
    td = (palloc_thread_data_t*)_palloc_os_zalloc(sizeof(palloc_thread_data_t), &memid);
    if (td == NULL) {
      // really out of memory
      _palloc_error_message(ENOMEM, "unable to allocate thread local heap metadata (%zu bytes)\n", sizeof(palloc_thread_data_t));
      return NULL;
    }
  }
  td->memid = memid;
  return td;
}

static void palloc_thread_data_free( palloc_thread_data_t* tdfree ) {
  // try to add the thread metadata to the cache
  for (int i = 0; i < TD_CACHE_SIZE; i++) {
    palloc_thread_data_t* td = palloc_atomic_load_ptr_relaxed(palloc_thread_data_t, &td_cache[i]);
    if (td == NULL) {
      palloc_thread_data_t* expected = NULL;
      if (palloc_atomic_cas_ptr_weak_acq_rel(palloc_thread_data_t, &td_cache[i], &expected, tdfree)) {
        return;
      }
    }
  }
  // if that fails, just free it directly
  _palloc_os_free(tdfree, sizeof(palloc_thread_data_t), tdfree->memid);
}

void _palloc_thread_data_collect(void) {
  // free all thread metadata from the cache
  for (int i = 0; i < TD_CACHE_SIZE; i++) {
    palloc_thread_data_t* td = palloc_atomic_load_ptr_relaxed(palloc_thread_data_t, &td_cache[i]);
    if (td != NULL) {
      td = palloc_atomic_exchange_ptr_acq_rel(palloc_thread_data_t, &td_cache[i], NULL);
      if (td != NULL) {
        _palloc_os_free(td, sizeof(palloc_thread_data_t), td->memid);
      }
    }
  }
}

// Initialize the thread local default heap, called from `palloc_thread_init`
static bool _palloc_thread_heap_init(void) {
  if (palloc_heap_is_initialized(palloc_prim_get_default_heap())) return true;
  if (_palloc_is_main_thread()) {
    // palloc_assert_internal(_palloc_heap_main.thread_id != 0);  // can happen on freeBSD where alloc is called before any initialization
    // the main heap is statically allocated
    palloc_heap_main_init();
    _palloc_heap_set_default_direct(&_palloc_heap_main);
    //palloc_assert_internal(_palloc_heap_default->tld->heap_backing == palloc_prim_get_default_heap());
  }
  else {
    // use `_palloc_os_alloc` to allocate directly from the OS
    palloc_thread_data_t* td = palloc_thread_data_zalloc();
    if (td == NULL) return false;

    palloc_tld_t*  tld = &td->tld;
    palloc_heap_t* heap = &td->heap;
    _palloc_tld_init(tld, heap);  // must be before `_palloc_heap_init`
    _palloc_heap_init(heap, tld, _palloc_arena_id_none(), false /* can reclaim */, 0 /* default tag */);
    _palloc_heap_set_default_direct(heap);
  }
  return false;
}

// initialize thread local data
void _palloc_tld_init(palloc_tld_t* tld, palloc_heap_t* bheap) {
  _palloc_memcpy_aligned(tld, &tld_empty, sizeof(palloc_tld_t));
  tld->heap_backing = bheap;
  tld->heaps = NULL;
  tld->segments.subproc = &palloc_subproc_default;
  tld->segments.stats = &tld->stats;
}

// Free the thread local default heap (called from `palloc_thread_done`)
static bool _palloc_thread_heap_done(palloc_heap_t* heap) {
  if (!palloc_heap_is_initialized(heap)) return true;

  // reset default heap
  _palloc_heap_set_default_direct(_palloc_is_main_thread() ? &_palloc_heap_main : (palloc_heap_t*)&_palloc_heap_empty);

  // switch to backing heap
  heap = heap->tld->heap_backing;
  if (!palloc_heap_is_initialized(heap)) return false;

  // delete all non-backing heaps in this thread
  palloc_heap_t* curr = heap->tld->heaps;
  while (curr != NULL) {
    palloc_heap_t* next = curr->next; // save `next` as `curr` will be freed
    if (curr != heap) {
      palloc_assert_internal(!palloc_heap_is_backing(curr));
      palloc_heap_delete(curr);
    }
    curr = next;
  }
  palloc_assert_internal(heap->tld->heaps == heap && heap->next == NULL);
  palloc_assert_internal(palloc_heap_is_backing(heap));

  // collect if not the main thread
  if (heap != &_palloc_heap_main) {
    _palloc_heap_collect_abandon(heap);
  }

  // merge stats
  _palloc_stats_done(&heap->tld->stats);

  // free if not the main thread
  if (heap != &_palloc_heap_main) {
    // the following assertion does not always hold for huge segments as those are always treated
    // as abondened: one may allocate it in one thread, but deallocate in another in which case
    // the count can be too large or negative. todo: perhaps not count huge segments? see issue #363
    // palloc_assert_internal(heap->tld->segments.count == 0 || heap->thread_id != _palloc_thread_id());
    palloc_thread_data_free((palloc_thread_data_t*)heap);
  }
  else {
    #if 0
    // never free the main thread even in debug mode; if a dll is linked statically with palloc,
    // there may still be delete/free calls after the palloc_fls_done is called. Issue #207
    _palloc_heap_destroy_pages(heap);
    palloc_assert_internal(heap->tld->heap_backing == &_palloc_heap_main);
    #endif
  }
  return false;
}



// --------------------------------------------------------
// Try to run `palloc_thread_done()` automatically so any memory
// owned by the thread but not yet released can be abandoned
// and re-owned by another thread.
//
// 1. windows dynamic library:
//     call from DllMain on DLL_THREAD_DETACH
// 2. windows static library:
//     use `FlsAlloc` to call a destructor when the thread is done
// 3. unix, pthreads:
//     use a pthread key to call a destructor when a pthread is done
//
// In the last two cases we also need to call `palloc_process_init`
// to set up the thread local keys.
// --------------------------------------------------------

// Set up handlers so `palloc_thread_done` is called automatically
static void palloc_process_setup_auto_thread_done(void) {
  static bool tls_initialized = false; // fine if it races
  if (tls_initialized) return;
  tls_initialized = true;
  _palloc_prim_thread_init_auto_done();
  _palloc_heap_set_default_direct(&_palloc_heap_main);
}


bool _palloc_is_main_thread(void) {
  return (_palloc_heap_main.thread_id==0 || _palloc_heap_main.thread_id == _palloc_thread_id());
}

static _Atomic(size_t) thread_count = PALLOC_ATOMIC_VAR_INIT(1);

size_t  _palloc_current_thread_count(void) {
  return palloc_atomic_load_relaxed(&thread_count);
}

// This is called from the `palloc_malloc_generic`
void palloc_thread_init(void) palloc_attr_noexcept
{
  // ensure our process has started already
  palloc_process_init();

  // initialize the thread local default heap
  // (this will call `_palloc_heap_set_default_direct` and thus set the
  //  fiber/pthread key to a non-zero value, ensuring `_palloc_thread_done` is called)
  if (_palloc_thread_heap_init()) return;  // returns true if already initialized

  _palloc_stat_increase(&_palloc_stats_main.threads, 1);
  palloc_atomic_increment_relaxed(&thread_count);
  //_palloc_verbose_message("thread init: 0x%zx\n", _palloc_thread_id());
}

void palloc_thread_done(void) palloc_attr_noexcept {
  _palloc_thread_done(NULL);
}

void _palloc_thread_done(palloc_heap_t* heap)
{
  // calling with NULL implies using the default heap
  if (heap == NULL) {
    heap = palloc_prim_get_default_heap();
    if (heap == NULL) return;
  }

  // prevent re-entrancy through heap_done/heap_set_default_direct (issue #699)
  if (!palloc_heap_is_initialized(heap)) {
    return;
  }

  // adjust stats
  palloc_atomic_decrement_relaxed(&thread_count);
  _palloc_stat_decrease(&_palloc_stats_main.threads, 1);

  // check thread-id as on Windows shutdown with FLS the main (exit) thread may call this on thread-local heaps...
  if (heap->thread_id != _palloc_thread_id()) return;

  // abandon the thread local heap
  if (_palloc_thread_heap_done(heap)) return;  // returns true if already ran
}

void _palloc_heap_set_default_direct(palloc_heap_t* heap)  {
  palloc_assert_internal(heap != NULL);
  #if defined(PALLOC_TLS_SLOT)
  palloc_prim_tls_slot_set(PALLOC_TLS_SLOT,heap);
  #elif defined(PALLOC_TLS_PTHREAD_SLOT_OFS)
  *palloc_prim_tls_pthread_heap_slot() = heap;
  #elif defined(PALLOC_TLS_PTHREAD)
  // we use _palloc_heap_default_key
  #else
  _palloc_heap_default = heap;
  #endif

  // ensure the default heap is passed to `_palloc_thread_done`
  // setting to a non-NULL value also ensures `palloc_thread_done` is called.
  _palloc_prim_thread_associate_default_heap(heap);
}

void palloc_thread_set_in_threadpool(void) palloc_attr_noexcept {
  // nothing
}

// --------------------------------------------------------
// Run functions on process init/done, and thread init/done
// --------------------------------------------------------
static bool os_preloading = true;    // true until this module is initialized

// Returns true if this module has not been initialized; Don't use C runtime routines until it returns false.
bool palloc_decl_noinline _palloc_preloading(void) {
  return os_preloading;
}

// Returns true if palloc was redirected
palloc_decl_nodiscard bool palloc_is_redirected(void) palloc_attr_noexcept {
  return _palloc_is_redirected();
}

// Called once by the process loader from `src/prim/prim.c`
void _palloc_auto_process_init(void) {
  palloc_heap_main_init();
  #if defined(__APPLE__) || defined(PALLOC_TLS_RECURSE_GUARD)
  volatile palloc_heap_t* dummy = _palloc_heap_default; // access TLS to allocate it before setting tls_initialized to true;
  if (dummy == NULL) return;                    // use dummy or otherwise the access may get optimized away (issue #697)
  #endif
  os_preloading = false;
  palloc_assert_internal(_palloc_is_main_thread());
  _palloc_options_init();
  palloc_process_setup_auto_thread_done();
  palloc_process_init();
  if (_palloc_is_redirected()) _palloc_verbose_message("malloc is redirected.\n");

  // show message from the redirector (if present)
  const char* msg = NULL;
  _palloc_allocator_init(&msg);
  if (msg != NULL && (palloc_option_is_enabled(palloc_option_verbose) || palloc_option_is_enabled(palloc_option_show_errors))) {
    _palloc_fputs(NULL,NULL,NULL,msg);
  }

  // reseed random
  _palloc_random_reinit_if_weak(&_palloc_heap_main.random);
}

#if defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
palloc_decl_cache_align bool _palloc_cpu_has_fsrm = false;
palloc_decl_cache_align bool _palloc_cpu_has_erms = false;

static void palloc_detect_cpu_features(void) {
  // FSRM for fast short rep movsb/stosb support (AMD Zen3+ (~2020) or Intel Ice Lake+ (~2017))
  // EMRS for fast enhanced rep movsb/stosb support
  int32_t cpu_info[4];
  __cpuid(cpu_info, 7);
  _palloc_cpu_has_fsrm = ((cpu_info[3] & (1 << 4)) != 0); // bit 4 of EDX : see <https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features>
  _palloc_cpu_has_erms = ((cpu_info[1] & (1 << 9)) != 0); // bit 9 of EBX : see <https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features>
}
#else
static void palloc_detect_cpu_features(void) {
  // nothing
}
#endif

// Initialize the process; called by thread_init or the process loader
void palloc_process_init(void) palloc_attr_noexcept {
  // ensure we are called once
  static palloc_atomic_once_t process_init;
	#if _MSC_VER < 1920
	palloc_heap_main_init(); // vs2017 can dynamically re-initialize _palloc_heap_main
	#endif
  if (!palloc_atomic_once(&process_init)) return;
  _palloc_process_is_initialized = true;
  _palloc_verbose_message("process init: 0x%zx\n", _palloc_thread_id());
  palloc_process_setup_auto_thread_done();

  palloc_detect_cpu_features();
  _palloc_os_init();
  palloc_heap_main_init();
  palloc_thread_init();

  #if defined(_WIN32)
  // On windows, when building as a static lib the FLS cleanup happens to early for the main thread.
  // To avoid this, set the FLS value for the main thread to NULL so the fls cleanup
  // will not call _palloc_thread_done on the (still executing) main thread. See issue #508.
  _palloc_prim_thread_associate_default_heap(NULL);
  #endif

  palloc_stats_reset();  // only call stat reset *after* thread init (or the heap tld == NULL)
  palloc_track_init();

  if (palloc_option_is_enabled(palloc_option_reserve_huge_os_pages)) {
    size_t pages = palloc_option_get_clamp(palloc_option_reserve_huge_os_pages, 0, 128*1024);
    int reserve_at  = (int)palloc_option_get_clamp(palloc_option_reserve_huge_os_pages_at, -1, INT_MAX);
    if (reserve_at != -1) {
      palloc_reserve_huge_os_pages_at(pages, reserve_at, pages*500);
    } else {
      palloc_reserve_huge_os_pages_interleave(pages, 0, pages*500);
    }
  }
  if (palloc_option_is_enabled(palloc_option_reserve_os_memory)) {
    long ksize = palloc_option_get(palloc_option_reserve_os_memory);
    if (ksize > 0) {
      palloc_reserve_os_memory((size_t)ksize*PALLOC_KiB, true /* commit? */, true /* allow large pages? */);
    }
  }
}

// Called when the process is done (cdecl as it is used with `at_exit` on some platforms)
void palloc_cdecl palloc_process_done(void) palloc_attr_noexcept {
  // only shutdown if we were initialized
  if (!_palloc_process_is_initialized) return;
  // ensure we are called once
  static bool process_done = false;
  if (process_done) return;
  process_done = true;

  // get the default heap so we don't need to acces thread locals anymore
  palloc_heap_t* heap = palloc_prim_get_default_heap();  // use prim to not initialize any heap
  palloc_assert_internal(heap != NULL);

  // release any thread specific resources and ensure _palloc_thread_done is called on all but the main thread
  _palloc_prim_thread_done_auto_done();


  #ifndef PALLOC_SKIP_COLLECT_ON_EXIT
    #if (PALLOC_DEBUG || !defined(PALLOC_SHARED_LIB))
    // free all memory if possible on process exit. This is not needed for a stand-alone process
    // but should be done if palloc is statically linked into another shared library which
    // is repeatedly loaded/unloaded, see issue #281.
    palloc_heap_collect(heap, true /* force */ );
    #endif
  #endif

  // Forcefully release all retained memory; this can be dangerous in general if overriding regular malloc/free
  // since after process_done there might still be other code running that calls `free` (like at_exit routines,
  // or C-runtime termination code.
  if (palloc_option_is_enabled(palloc_option_destroy_on_exit)) {
    palloc_heap_collect(heap, true /* force */);
    _palloc_heap_unsafe_destroy_all(heap);     // forcefully release all memory held by all heaps (of this thread only!)
    _palloc_arena_unsafe_destroy_all();
    _palloc_segment_map_unsafe_destroy();
  }

  if (palloc_option_is_enabled(palloc_option_show_stats) || palloc_option_is_enabled(palloc_option_verbose)) {
    palloc_stats_print(NULL);
  }
  _palloc_allocator_done();
  _palloc_verbose_message("process done: 0x%zx\n", _palloc_heap_main.thread_id);
  os_preloading = true; // don't call the C runtime anymore
}

void palloc_cdecl _palloc_auto_process_done(void) palloc_attr_noexcept {
  if (_palloc_option_get_fast(palloc_option_destroy_on_exit)>1) return;
  palloc_process_done();
}
