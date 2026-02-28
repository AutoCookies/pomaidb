/*----------------------------------------------------------------------------
Copyright (c) 2018-2021, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"
#include "palloc/prim.h"  // palloc_prim_get_default_heap

#include <string.h>  // memset, memcpy

#if defined(_MSC_VER) && (_MSC_VER < 1920)
#pragma warning(disable:4204)  // non-constant aggregate initializer
#endif

/* -----------------------------------------------------------
  Helpers
----------------------------------------------------------- */

// return `true` if ok, `false` to break
typedef bool (heap_page_visitor_fun)(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* arg1, void* arg2);

// Visit all pages in a heap; returns `false` if break was called.
static bool palloc_heap_visit_pages(palloc_heap_t* heap, heap_page_visitor_fun* fn, void* arg1, void* arg2)
{
  if (heap==NULL || heap->page_count==0) return 0;

  // visit all pages
  #if PALLOC_DEBUG>1
  size_t total = heap->page_count;
  size_t count = 0;
  #endif

  for (size_t i = 0; i <= PALLOC_BIN_FULL; i++) {
    palloc_page_queue_t* pq = &heap->pages[i];
    palloc_page_t* page = pq->first;
    while(page != NULL) {
      palloc_page_t* next = page->next; // save next in case the page gets removed from the queue
      palloc_assert_internal(palloc_page_heap(page) == heap);
      #if PALLOC_DEBUG>1
      count++;
      #endif
      if (!fn(heap, pq, page, arg1, arg2)) return false;
      page = next; // and continue
    }
  }
  palloc_assert_internal(count == total);
  return true;
}


#if PALLOC_DEBUG>=2
static bool palloc_heap_page_is_valid(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* arg1, void* arg2) {
  PALLOC_UNUSED(arg1);
  PALLOC_UNUSED(arg2);
  PALLOC_UNUSED(pq);
  palloc_assert_internal(palloc_page_heap(page) == heap);
  palloc_segment_t* segment = _palloc_page_segment(page);
  palloc_assert_internal(palloc_atomic_load_relaxed(&segment->thread_id) == heap->thread_id);
  palloc_assert_expensive(_palloc_page_is_valid(page));
  return true;
}
#endif
#if PALLOC_DEBUG>=3
static bool palloc_heap_is_valid(palloc_heap_t* heap) {
  palloc_assert_internal(heap!=NULL);
  palloc_heap_visit_pages(heap, &palloc_heap_page_is_valid, NULL, NULL);
  return true;
}
#endif




/* -----------------------------------------------------------
  "Collect" pages by migrating `local_free` and `thread_free`
  lists and freeing empty pages. This is done when a thread
  stops (and in that case abandons pages if there are still
  blocks alive)
----------------------------------------------------------- */

typedef enum palloc_collect_e {
  PALLOC_NORMAL,
  PALLOC_FORCE,
  PALLOC_ABANDON
} palloc_collect_t;


static bool palloc_heap_page_collect(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* arg_collect, void* arg2 ) {
  PALLOC_UNUSED(arg2);
  PALLOC_UNUSED(heap);
  palloc_assert_internal(palloc_heap_page_is_valid(heap, pq, page, NULL, NULL));
  palloc_collect_t collect = *((palloc_collect_t*)arg_collect);
  _palloc_page_free_collect(page, collect >= PALLOC_FORCE);
  if (collect == PALLOC_FORCE) {
    // note: call before a potential `_palloc_page_free` as the segment may be freed if this was the last used page in that segment.
    palloc_segment_t* segment = _palloc_page_segment(page);
    _palloc_segment_collect(segment, true /* force? */);
  }
  if (palloc_page_all_free(page)) {
    // no more used blocks, free the page.
    // note: this will free retired pages as well.
    _palloc_page_free(page, pq, collect >= PALLOC_FORCE);
  }
  else if (collect == PALLOC_ABANDON) {
    // still used blocks but the thread is done; abandon the page
    _palloc_page_abandon(page, pq);
  }
  return true; // don't break
}

static bool palloc_heap_page_never_delayed_free(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* arg1, void* arg2) {
  PALLOC_UNUSED(arg1);
  PALLOC_UNUSED(arg2);
  PALLOC_UNUSED(heap);
  PALLOC_UNUSED(pq);
  _palloc_page_use_delayed_free(page, PALLOC_NEVER_DELAYED_FREE, false);
  return true; // don't break
}

static void palloc_heap_collect_ex(palloc_heap_t* heap, palloc_collect_t collect)
{
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return;

  const bool force = (collect >= PALLOC_FORCE);
  _palloc_deferred_free(heap, force);

  // python/cpython#112532: we may be called from a thread that is not the owner of the heap
  const bool is_main_thread = (_palloc_is_main_thread() && heap->thread_id == _palloc_thread_id());

  // note: never reclaim on collect but leave it to threads that need storage to reclaim
  const bool force_main =
    #ifdef NDEBUG
      collect == PALLOC_FORCE
    #else
      collect >= PALLOC_FORCE
    #endif
      && is_main_thread && palloc_heap_is_backing(heap) && !heap->no_reclaim;

  if (force_main) {
    // the main thread is abandoned (end-of-program), try to reclaim all abandoned segments.
    // if all memory is freed by now, all segments should be freed.
    // note: this only collects in the current subprocess
    _palloc_abandoned_reclaim_all(heap, &heap->tld->segments);
  }

  // if abandoning, mark all pages to no longer add to delayed_free
  if (collect == PALLOC_ABANDON) {
    palloc_heap_visit_pages(heap, &palloc_heap_page_never_delayed_free, NULL, NULL);
  }

  // free all current thread delayed blocks.
  // (if abandoning, after this there are no more thread-delayed references into the pages.)
  _palloc_heap_delayed_free_all(heap);

  // collect retired pages
  _palloc_heap_collect_retired(heap, force);

  // collect all pages owned by this thread
  palloc_heap_visit_pages(heap, &palloc_heap_page_collect, &collect, NULL);
  palloc_assert_internal( collect != PALLOC_ABANDON || palloc_atomic_load_ptr_acquire(palloc_block_t,&heap->thread_delayed_free) == NULL );

  // collect abandoned segments (in particular, purge expired parts of segments in the abandoned segment list)
  // note: forced purge can be quite expensive if many threads are created/destroyed so we do not force on abandonment
  _palloc_abandoned_collect(heap, collect == PALLOC_FORCE /* force? */, &heap->tld->segments);

  // if forced, collect thread data cache on program-exit (or shared library unload)
  if (force && is_main_thread && palloc_heap_is_backing(heap)) {
    _palloc_thread_data_collect();  // collect thread data cache
  }

  // collect arenas (this is program wide so don't force purges on abandonment of threads)
  _palloc_arenas_collect(collect == PALLOC_FORCE /* force purge? */);

  // merge statistics
  if (collect <= PALLOC_FORCE) { _palloc_stats_merge_thread(heap->tld); }
}

void _palloc_heap_collect_abandon(palloc_heap_t* heap) {
  palloc_heap_collect_ex(heap, PALLOC_ABANDON);
}

void palloc_heap_collect(palloc_heap_t* heap, bool force) palloc_attr_noexcept {
  palloc_heap_collect_ex(heap, (force ? PALLOC_FORCE : PALLOC_NORMAL));
}

void palloc_collect(bool force) palloc_attr_noexcept {
  palloc_heap_collect(palloc_prim_get_default_heap(), force);
}


/* -----------------------------------------------------------
  Heap new
----------------------------------------------------------- */

palloc_heap_t* palloc_heap_get_default(void) {
  palloc_thread_init();
  return palloc_prim_get_default_heap();
}

static bool palloc_heap_is_default(const palloc_heap_t* heap) {
  return (heap == palloc_prim_get_default_heap());
}


palloc_heap_t* palloc_heap_get_backing(void) {
  palloc_heap_t* heap = palloc_heap_get_default();
  palloc_assert_internal(heap!=NULL);
  palloc_heap_t* bheap = heap->tld->heap_backing;
  palloc_assert_internal(bheap!=NULL);
  palloc_assert_internal(bheap->thread_id == _palloc_thread_id());
  return bheap;
}

void _palloc_heap_init(palloc_heap_t* heap, palloc_tld_t* tld, palloc_arena_id_t arena_id, bool noreclaim, uint8_t tag) {
  _palloc_memcpy_aligned(heap, &_palloc_heap_empty, sizeof(palloc_heap_t));
  heap->tld = tld;
  heap->thread_id  = _palloc_thread_id();
  heap->arena_id   = arena_id;
  heap->no_reclaim = noreclaim;
  heap->tag        = tag;
  if (heap == tld->heap_backing) {
    #if defined(_WIN32) && !defined(PALLOC_SHARED_LIB)
      _palloc_random_init_weak(&heap->random);    // prevent allocation failure during bcrypt dll initialization with static linking (issue #1185)
    #else
      _palloc_random_init(&heap->random);
    #endif
  }
  else {
    _palloc_random_split(&tld->heap_backing->random, &heap->random);
  }
  heap->cookie  = _palloc_heap_random_next(heap) | 1;
  heap->keys[0] = _palloc_heap_random_next(heap);
  heap->keys[1] = _palloc_heap_random_next(heap);
  _palloc_heap_guarded_init(heap);
  // push on the thread local heaps list
  heap->next = heap->tld->heaps;
  heap->tld->heaps = heap;
}

palloc_decl_nodiscard palloc_heap_t* palloc_heap_new_ex(int heap_tag, bool allow_destroy, palloc_arena_id_t arena_id) {
  palloc_heap_t* bheap = palloc_heap_get_backing();
  palloc_heap_t* heap = palloc_heap_malloc_tp(bheap, palloc_heap_t);  // todo: OS allocate in secure mode?
  if (heap == NULL) return NULL;
  palloc_assert(heap_tag >= 0 && heap_tag < 256);
  _palloc_heap_init(heap, bheap->tld, arena_id, allow_destroy /* no reclaim? */, (uint8_t)heap_tag /* heap tag */);
  return heap;
}

palloc_decl_nodiscard palloc_heap_t* palloc_heap_new_in_arena(palloc_arena_id_t arena_id) {
  return palloc_heap_new_ex(0 /* default heap tag */, false /* don't allow `palloc_heap_destroy` */, arena_id);
}

palloc_decl_nodiscard palloc_heap_t* palloc_heap_new(void) {
  // don't reclaim abandoned memory or otherwise destroy is unsafe
  return palloc_heap_new_ex(0 /* default heap tag */, true /* no reclaim */, _palloc_arena_id_none());
}

bool _palloc_heap_memid_is_suitable(palloc_heap_t* heap, palloc_memid_t memid) {
  return _palloc_arena_memid_is_suitable(memid, heap->arena_id);
}

uintptr_t _palloc_heap_random_next(palloc_heap_t* heap) {
  return _palloc_random_next(&heap->random);
}

// zero out the page queues
static void palloc_heap_reset_pages(palloc_heap_t* heap) {
  palloc_assert_internal(heap != NULL);
  palloc_assert_internal(palloc_heap_is_initialized(heap));
  // TODO: copy full empty heap instead?
  memset(&heap->pages_free_direct, 0, sizeof(heap->pages_free_direct));
  _palloc_memcpy_aligned(&heap->pages, &_palloc_heap_empty.pages, sizeof(heap->pages));
  heap->thread_delayed_free = NULL;
  heap->page_count = 0;
}

// called from `palloc_heap_destroy` and `palloc_heap_delete` to free the internal heap resources.
static void palloc_heap_free(palloc_heap_t* heap) {
  palloc_assert(heap != NULL);
  palloc_assert_internal(palloc_heap_is_initialized(heap));
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return;
  if (palloc_heap_is_backing(heap)) return; // dont free the backing heap

  // reset default
  if (palloc_heap_is_default(heap)) {
    _palloc_heap_set_default_direct(heap->tld->heap_backing);
  }

  // remove ourselves from the thread local heaps list
  // linear search but we expect the number of heaps to be relatively small
  palloc_heap_t* prev = NULL;
  palloc_heap_t* curr = heap->tld->heaps;
  while (curr != heap && curr != NULL) {
    prev = curr;
    curr = curr->next;
  }
  palloc_assert_internal(curr == heap);
  if (curr == heap) {
    if (prev != NULL) { prev->next = heap->next; }
                 else { heap->tld->heaps = heap->next; }
  }
  palloc_assert_internal(heap->tld->heaps != NULL);

  // and free the used memory
  palloc_free(heap);
}

// return a heap on the same thread as `heap` specialized for the specified tag (if it exists)
palloc_heap_t* _palloc_heap_by_tag(palloc_heap_t* heap, uint8_t tag) {
  if (heap->tag == tag) {
    return heap;
  }
  for (palloc_heap_t *curr = heap->tld->heaps; curr != NULL; curr = curr->next) {
    if (curr->tag == tag) {
      return curr;
    }
  }
  return NULL;
}

/* -----------------------------------------------------------
  Heap destroy
----------------------------------------------------------- */

static bool _palloc_heap_page_destroy(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* arg1, void* arg2) {
  PALLOC_UNUSED(arg1);
  PALLOC_UNUSED(arg2);
  PALLOC_UNUSED(heap);
  PALLOC_UNUSED(pq);

  // ensure no more thread_delayed_free will be added
  _palloc_page_use_delayed_free(page, PALLOC_NEVER_DELAYED_FREE, false);

  // stats
  const size_t bsize = palloc_page_block_size(page);
  if (bsize > PALLOC_MEDIUM_OBJ_SIZE_MAX) {
    //if (bsize <= PALLOC_LARGE_OBJ_SIZE_MAX) {
    //  palloc_heap_stat_decrease(heap, malloc_large, bsize);
    //}
    //else 
    {
      palloc_heap_stat_decrease(heap, malloc_huge, bsize);
    }
  }
  #if (PALLOC_STAT>0)
  _palloc_page_free_collect(page, false);  // update used count
  const size_t inuse = page->used;
  if (bsize <= PALLOC_LARGE_OBJ_SIZE_MAX) {
    palloc_heap_stat_decrease(heap, malloc_normal, bsize * inuse);
    #if (PALLOC_STAT>1)
    palloc_heap_stat_decrease(heap, malloc_bins[_palloc_bin(bsize)], inuse);
    #endif
  }
  // palloc_heap_stat_decrease(heap, malloc_requested, bsize * inuse);  // todo: off for aligned blocks...
  #endif

  /// pretend it is all free now
  palloc_assert_internal(palloc_page_thread_free(page) == NULL);
  page->used = 0;

  // and free the page
  // palloc_page_free(page,false);
  page->next = NULL;
  page->prev = NULL;
  _palloc_segment_page_free(page,false /* no force? */, &heap->tld->segments);

  return true; // keep going
}

void _palloc_heap_destroy_pages(palloc_heap_t* heap) {
  palloc_heap_visit_pages(heap, &_palloc_heap_page_destroy, NULL, NULL);
  palloc_heap_reset_pages(heap);
}

#if PALLOC_TRACK_HEAP_DESTROY
static bool palloc_cdecl palloc_heap_track_block_free(const palloc_heap_t* heap, const palloc_heap_area_t* area, void* block, size_t block_size, void* arg) {
  PALLOC_UNUSED(heap); PALLOC_UNUSED(area);  PALLOC_UNUSED(arg); PALLOC_UNUSED(block_size);
  palloc_track_free_size(block,palloc_usable_size(block));
  return true;
}
#endif

void palloc_heap_destroy(palloc_heap_t* heap) {
  palloc_assert(heap != NULL);
  palloc_assert(palloc_heap_is_initialized(heap));
  palloc_assert(heap->no_reclaim);
  palloc_assert_expensive(palloc_heap_is_valid(heap));
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return;
  #if PALLOC_GUARDED
  // _palloc_warning_message("'palloc_heap_destroy' called but PALLOC_GUARDED is enabled -- using `palloc_heap_delete` instead (heap at %p)\n", heap);
  palloc_heap_delete(heap);
  return;
  #else
  if (!heap->no_reclaim) {
    _palloc_warning_message("'palloc_heap_destroy' called but ignored as the heap was not created with 'allow_destroy' (heap at %p)\n", heap);
    // don't free in case it may contain reclaimed pages
    palloc_heap_delete(heap);
  }
  else {
    // track all blocks as freed
    #if PALLOC_TRACK_HEAP_DESTROY
    palloc_heap_visit_blocks(heap, true, palloc_heap_track_block_free, NULL);
    #endif
    // free all pages
    _palloc_heap_destroy_pages(heap);
    palloc_heap_free(heap);
  }
  #endif
}

// forcefully destroy all heaps in the current thread
void _palloc_heap_unsafe_destroy_all(palloc_heap_t* heap) {
  palloc_assert_internal(heap != NULL);
  if (heap == NULL) return;
  palloc_heap_t* curr = heap->tld->heaps;
  while (curr != NULL) {
    palloc_heap_t* next = curr->next;
    if (curr->no_reclaim) {
      palloc_heap_destroy(curr);
    }
    else {
      _palloc_heap_destroy_pages(curr);
    }
    curr = next;
  }
}

/* -----------------------------------------------------------
  Safe Heap delete
----------------------------------------------------------- */

// Transfer the pages from one heap to the other
static void palloc_heap_absorb(palloc_heap_t* heap, palloc_heap_t* from) {
  palloc_assert_internal(heap!=NULL);
  if (from==NULL || from->page_count == 0) return;

  // reduce the size of the delayed frees
  _palloc_heap_delayed_free_partial(from);

  // transfer all pages by appending the queues; this will set a new heap field
  // so threads may do delayed frees in either heap for a while.
  // note: appending waits for each page to not be in the `PALLOC_DELAYED_FREEING` state
  // so after this only the new heap will get delayed frees
  for (size_t i = 0; i <= PALLOC_BIN_FULL; i++) {
    palloc_page_queue_t* pq = &heap->pages[i];
    palloc_page_queue_t* append = &from->pages[i];
    size_t pcount = _palloc_page_queue_append(heap, pq, append);
    heap->page_count += pcount;
    from->page_count -= pcount;
  }
  palloc_assert_internal(from->page_count == 0);

  // and do outstanding delayed frees in the `from` heap
  // note: be careful here as the `heap` field in all those pages no longer point to `from`,
  // turns out to be ok as `_palloc_heap_delayed_free` only visits the list and calls a
  // the regular `_palloc_free_delayed_block` which is safe.
  _palloc_heap_delayed_free_all(from);
  #if !defined(_MSC_VER) || (_MSC_VER > 1900) // somehow the following line gives an error in VS2015, issue #353
  palloc_assert_internal(palloc_atomic_load_ptr_relaxed(palloc_block_t,&from->thread_delayed_free) == NULL);
  #endif

  // and reset the `from` heap
  palloc_heap_reset_pages(from);
}

// are two heaps compatible with respect to heap-tag, exclusive arena etc.
static bool palloc_heaps_are_compatible(palloc_heap_t* heap1, palloc_heap_t* heap2) {
  return (heap1->tag == heap2->tag &&                   // store same kind of objects
          heap1->arena_id == heap2->arena_id);          // same arena preference
}

// Safe delete a heap without freeing any still allocated blocks in that heap.
void palloc_heap_delete(palloc_heap_t* heap)
{
  palloc_assert(heap != NULL);
  palloc_assert(palloc_heap_is_initialized(heap));
  palloc_assert_expensive(palloc_heap_is_valid(heap));
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return;

  palloc_heap_t* bheap = heap->tld->heap_backing;
  if (bheap != heap && palloc_heaps_are_compatible(bheap,heap)) {
    // transfer still used pages to the backing heap
    palloc_heap_absorb(bheap, heap);
  }
  else {
    // the backing heap abandons its pages
    _palloc_heap_collect_abandon(heap);
  }
  palloc_assert_internal(heap->page_count==0);
  palloc_heap_free(heap);
}

palloc_heap_t* palloc_heap_set_default(palloc_heap_t* heap) {
  palloc_assert(heap != NULL);
  palloc_assert(palloc_heap_is_initialized(heap));
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return NULL;
  palloc_assert_expensive(palloc_heap_is_valid(heap));
  palloc_heap_t* old = palloc_prim_get_default_heap();
  _palloc_heap_set_default_direct(heap);
  return old;
}




/* -----------------------------------------------------------
  Analysis
----------------------------------------------------------- */

// static since it is not thread safe to access heaps from other threads.
static palloc_heap_t* palloc_heap_of_block(const void* p) {
  if (p == NULL) return NULL;
  palloc_segment_t* segment = _palloc_ptr_segment(p);
  bool valid = (_palloc_ptr_cookie(segment) == segment->cookie);
  palloc_assert_internal(valid);
  if palloc_unlikely(!valid) return NULL;
  return palloc_page_heap(_palloc_segment_page_of(segment,p));
}

bool palloc_heap_contains_block(palloc_heap_t* heap, const void* p) {
  palloc_assert(heap != NULL);
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return false;
  return (heap == palloc_heap_of_block(p));
}


static bool palloc_heap_page_check_owned(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* p, void* vfound) {
  PALLOC_UNUSED(heap);
  PALLOC_UNUSED(pq);
  bool* found = (bool*)vfound;
  void* start = palloc_page_start(page);
  void* end   = (uint8_t*)start + (page->capacity * palloc_page_block_size(page));
  *found = (p >= start && p < end);
  return (!*found); // continue if not found
}

bool palloc_heap_check_owned(palloc_heap_t* heap, const void* p) {
  palloc_assert(heap != NULL);
  if (heap==NULL || !palloc_heap_is_initialized(heap)) return false;
  if (((uintptr_t)p & (PALLOC_INTPTR_SIZE - 1)) != 0) return false;  // only aligned pointers
  bool found = false;
  palloc_heap_visit_pages(heap, &palloc_heap_page_check_owned, (void*)p, &found);
  return found;
}

bool palloc_check_owned(const void* p) {
  return palloc_heap_check_owned(palloc_prim_get_default_heap(), p);
}

/* -----------------------------------------------------------
  Visit all heap blocks and areas
  Todo: enable visiting abandoned pages, and
        enable visiting all blocks of all heaps across threads
----------------------------------------------------------- */

void _palloc_heap_area_init(palloc_heap_area_t* area, palloc_page_t* page) {
  const size_t bsize = palloc_page_block_size(page);
  const size_t ubsize = palloc_page_usable_block_size(page);
  area->reserved = page->reserved * bsize;
  area->committed = page->capacity * bsize;
  area->blocks = palloc_page_start(page);
  area->used = page->used;   // number of blocks in use (#553)
  area->block_size = ubsize;
  area->full_block_size = bsize;
  area->heap_tag = page->heap_tag;
}


static void palloc_get_fast_divisor(size_t divisor, uint64_t* magic, size_t* shift) {
  palloc_assert_internal(divisor > 0 && divisor <= UINT32_MAX);
  *shift = PALLOC_SIZE_BITS - palloc_clz(divisor - 1);
  *magic = ((((uint64_t)1 << 32) * (((uint64_t)1 << *shift) - divisor)) / divisor + 1);
}

static size_t palloc_fast_divide(size_t n, uint64_t magic, size_t shift) {
  palloc_assert_internal(n <= UINT32_MAX);
  const uint64_t hi = ((uint64_t)n * magic) >> 32;
  return (size_t)((hi + n) >> shift);
}

bool _palloc_heap_area_visit_blocks(const palloc_heap_area_t* area, palloc_page_t* page, palloc_block_visit_fun* visitor, void* arg) {
  palloc_assert(area != NULL);
  if (area==NULL) return true;
  palloc_assert(page != NULL);
  if (page == NULL) return true;

  _palloc_page_free_collect(page,true);              // collect both thread_delayed and local_free
  palloc_assert_internal(page->local_free == NULL);
  if (page->used == 0) return true;

  size_t psize;
  uint8_t* const pstart = _palloc_segment_page_start(_palloc_page_segment(page), page, &psize);
  palloc_heap_t* const heap = palloc_page_heap(page);
  const size_t bsize    = palloc_page_block_size(page);
  const size_t ubsize   = palloc_page_usable_block_size(page); // without padding

  // optimize page with one block
  if (page->capacity == 1) {
    palloc_assert_internal(page->used == 1 && page->free == NULL);
    return visitor(palloc_page_heap(page), area, pstart, ubsize, arg);
  }
  palloc_assert(bsize <= UINT32_MAX);

  // optimize full pages
  if (page->used == page->capacity) {
    uint8_t* block = pstart;
    for (size_t i = 0; i < page->capacity; i++) {
      if (!visitor(heap, area, block, ubsize, arg)) return false;
      block += bsize;
    }
    return true;
  }

  // create a bitmap of free blocks.
  #define PALLOC_MAX_BLOCKS   (PALLOC_SMALL_PAGE_SIZE / sizeof(void*))
  uintptr_t free_map[PALLOC_MAX_BLOCKS / PALLOC_INTPTR_BITS];
  const uintptr_t bmapsize = _palloc_divide_up(page->capacity, PALLOC_INTPTR_BITS);
  memset(free_map, 0, bmapsize * sizeof(intptr_t));
  if (page->capacity % PALLOC_INTPTR_BITS != 0) {
    // mark left-over bits at the end as free
    size_t shift   = (page->capacity % PALLOC_INTPTR_BITS);
    uintptr_t mask = (UINTPTR_MAX << shift);
    free_map[bmapsize - 1] = mask;
  }

  // fast repeated division by the block size
  uint64_t magic;
  size_t   shift;
  palloc_get_fast_divisor(bsize, &magic, &shift);

  #if PALLOC_DEBUG>1
  size_t free_count = 0;
  #endif
  for (palloc_block_t* block = page->free; block != NULL; block = palloc_block_next(page, block)) {
    #if PALLOC_DEBUG>1
    free_count++;
    #endif
    palloc_assert_internal((uint8_t*)block >= pstart && (uint8_t*)block < (pstart + psize));
    size_t offset = (uint8_t*)block - pstart;
    palloc_assert_internal(offset % bsize == 0);
    palloc_assert_internal(offset <= UINT32_MAX);
    size_t blockidx = palloc_fast_divide(offset, magic, shift);
    palloc_assert_internal(blockidx == offset / bsize);
    palloc_assert_internal(blockidx < PALLOC_MAX_BLOCKS);
    size_t bitidx = (blockidx / PALLOC_INTPTR_BITS);
    size_t bit = blockidx - (bitidx * PALLOC_INTPTR_BITS);
    free_map[bitidx] |= ((uintptr_t)1 << bit);
  }
  palloc_assert_internal(page->capacity == (free_count + page->used));

  // walk through all blocks skipping the free ones
  #if PALLOC_DEBUG>1
  size_t used_count = 0;
  #endif
  uint8_t* block = pstart;
  for (size_t i = 0; i < bmapsize; i++) {
    if (free_map[i] == 0) {
      // every block is in use
      for (size_t j = 0; j < PALLOC_INTPTR_BITS; j++) {
        #if PALLOC_DEBUG>1
        used_count++;
        #endif
        if (!visitor(heap, area, block, ubsize, arg)) return false;
        block += bsize;
      }
    }
    else {
      // visit the used blocks in the mask
      uintptr_t m = ~free_map[i];
      while (m != 0) {
        #if PALLOC_DEBUG>1
        used_count++;
        #endif
        size_t bitidx = palloc_ctz(m);
        if (!visitor(heap, area, block + (bitidx * bsize), ubsize, arg)) return false;
        m &= m - 1;  // clear least significant bit
      }
      block += bsize * PALLOC_INTPTR_BITS;
    }
  }
  palloc_assert_internal(page->used == used_count);
  return true;
}



// Separate struct to keep `palloc_page_t` out of the public interface
typedef struct palloc_heap_area_ex_s {
  palloc_heap_area_t area;
  palloc_page_t* page;
} palloc_heap_area_ex_t;

typedef bool (palloc_heap_area_visit_fun)(const palloc_heap_t* heap, const palloc_heap_area_ex_t* area, void* arg);

static bool palloc_heap_visit_areas_page(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_t* page, void* vfun, void* arg) {
  PALLOC_UNUSED(heap);
  PALLOC_UNUSED(pq);
  palloc_heap_area_visit_fun* fun = (palloc_heap_area_visit_fun*)vfun;
  palloc_heap_area_ex_t xarea;
  xarea.page = page;
  _palloc_heap_area_init(&xarea.area, page);
  return fun(heap, &xarea, arg);
}

// Visit all heap pages as areas
static bool palloc_heap_visit_areas(const palloc_heap_t* heap, palloc_heap_area_visit_fun* visitor, void* arg) {
  if (visitor == NULL) return false;
  return palloc_heap_visit_pages((palloc_heap_t*)heap, &palloc_heap_visit_areas_page, (void*)(visitor), arg); // note: function pointer to void* :-{
}

// Just to pass arguments
typedef struct palloc_visit_blocks_args_s {
  bool  visit_blocks;
  palloc_block_visit_fun* visitor;
  void* arg;
} palloc_visit_blocks_args_t;

static bool palloc_heap_area_visitor(const palloc_heap_t* heap, const palloc_heap_area_ex_t* xarea, void* arg) {
  palloc_visit_blocks_args_t* args = (palloc_visit_blocks_args_t*)arg;
  if (!args->visitor(heap, &xarea->area, NULL, xarea->area.block_size, args->arg)) return false;
  if (args->visit_blocks) {
    return _palloc_heap_area_visit_blocks(&xarea->area, xarea->page, args->visitor, args->arg);
  }
  else {
    return true;
  }
}

// Visit all blocks in a heap
bool palloc_heap_visit_blocks(const palloc_heap_t* heap, bool visit_blocks, palloc_block_visit_fun* visitor, void* arg) {
  palloc_visit_blocks_args_t args = { visit_blocks, visitor, arg };
  return palloc_heap_visit_areas(heap, &palloc_heap_area_visitor, &args);
}
