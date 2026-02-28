/*----------------------------------------------------------------------------
Copyright (c) 2018-2024, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* -----------------------------------------------------------
  The core of the allocator. Every segment contains
  pages of a certain block size. The main function
  exported is `palloc_malloc_generic`.
----------------------------------------------------------- */

#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"

/* -----------------------------------------------------------
  Definition of page queues for each block size
----------------------------------------------------------- */

#define PALLOC_IN_PAGE_C
#include "page-queue.c"
#undef PALLOC_IN_PAGE_C


/* -----------------------------------------------------------
  Page helpers
----------------------------------------------------------- */

// Index a block in a page
static inline palloc_block_t* palloc_page_block_at(const palloc_page_t* page, void* page_start, size_t block_size, size_t i) {
  PALLOC_UNUSED(page);
  palloc_assert_internal(page != NULL);
  palloc_assert_internal(i <= page->reserved);
  return (palloc_block_t*)((uint8_t*)page_start + (i * block_size));
}

static void palloc_page_init(palloc_heap_t* heap, palloc_page_t* page, size_t size, palloc_tld_t* tld);
static bool palloc_page_extend_free(palloc_heap_t* heap, palloc_page_t* page, palloc_tld_t* tld);

#if (PALLOC_DEBUG>=3)
static size_t palloc_page_list_count(palloc_page_t* page, palloc_block_t* head) {
  size_t count = 0;
  while (head != NULL) {
    palloc_assert_internal(page == _palloc_ptr_page(head));
    count++;
    head = palloc_block_next(page, head);
  }
  return count;
}

/*
// Start of the page available memory
static inline uint8_t* palloc_page_area(const palloc_page_t* page) {
  return _palloc_page_start(_palloc_page_segment(page), page, NULL);
}
*/

static bool palloc_page_list_is_valid(palloc_page_t* page, palloc_block_t* p) {
  size_t psize;
  uint8_t* page_area = _palloc_segment_page_start(_palloc_page_segment(page), page, &psize);
  palloc_block_t* start = (palloc_block_t*)page_area;
  palloc_block_t* end   = (palloc_block_t*)(page_area + psize);
  while(p != NULL) {
    if (p < start || p >= end) return false;
    p = palloc_block_next(page, p);
  }
#if PALLOC_DEBUG>3 // generally too expensive to check this
  if (page->free_is_zero) {
    const size_t ubsize = palloc_page_usable_block_size(page);
    for (palloc_block_t* block = page->free; block != NULL; block = palloc_block_next(page, block)) {
      palloc_assert_expensive(palloc_mem_is_zero(block + 1, ubsize - sizeof(palloc_block_t)));
    }
  }
#endif
  return true;
}

static bool palloc_page_is_valid_init(palloc_page_t* page) {
  palloc_assert_internal(palloc_page_block_size(page) > 0);
  palloc_assert_internal(page->used <= page->capacity);
  palloc_assert_internal(page->capacity <= page->reserved);

  uint8_t* start = palloc_page_start(page);
  palloc_assert_internal(start == _palloc_segment_page_start(_palloc_page_segment(page), page, NULL));
  palloc_assert_internal(page->is_huge == (_palloc_page_segment(page)->kind == PALLOC_SEGMENT_HUGE));
  //palloc_assert_internal(start + page->capacity*page->block_size == page->top);

  palloc_assert_internal(palloc_page_list_is_valid(page,page->free));
  palloc_assert_internal(palloc_page_list_is_valid(page,page->local_free));

  #if PALLOC_DEBUG>3 // generally too expensive to check this
  if (page->free_is_zero) {
    const size_t ubsize = palloc_page_usable_block_size(page);
    for(palloc_block_t* block = page->free; block != NULL; block = palloc_block_next(page,block)) {
      palloc_assert_expensive(palloc_mem_is_zero(block + 1, ubsize - sizeof(palloc_block_t)));
    }
  }
  #endif

  #if !PALLOC_TRACK_ENABLED && !PALLOC_TSAN
  palloc_block_t* tfree = palloc_page_thread_free(page);
  palloc_assert_internal(palloc_page_list_is_valid(page, tfree));
  //size_t tfree_count = palloc_page_list_count(page, tfree);
  //palloc_assert_internal(tfree_count <= page->thread_freed + 1);
  #endif

  size_t free_count = palloc_page_list_count(page, page->free) + palloc_page_list_count(page, page->local_free);
  palloc_assert_internal(page->used + free_count == page->capacity);

  return true;
}

extern palloc_decl_hidden bool _palloc_process_is_initialized;             // has palloc_process_init been called?

bool _palloc_page_is_valid(palloc_page_t* page) {
  palloc_assert_internal(palloc_page_is_valid_init(page));
  #if PALLOC_SECURE
  palloc_assert_internal(page->keys[0] != 0);
  #endif
  if (palloc_page_heap(page)!=NULL) {
    palloc_segment_t* segment = _palloc_page_segment(page);

    palloc_assert_internal(!_palloc_process_is_initialized || segment->thread_id==0 || segment->thread_id == palloc_page_heap(page)->thread_id);
    #if PALLOC_HUGE_PAGE_ABANDON
    if (segment->kind != PALLOC_SEGMENT_HUGE)
    #endif
    {
      palloc_page_queue_t* pq = palloc_page_queue_of(page);
      palloc_assert_internal(palloc_page_queue_contains(pq, page));
      palloc_assert_internal(pq->block_size==palloc_page_block_size(page) || palloc_page_block_size(page) > PALLOC_MEDIUM_OBJ_SIZE_MAX || palloc_page_is_in_full(page));
      palloc_assert_internal(palloc_heap_contains_queue(palloc_page_heap(page),pq));
    }
  }
  return true;
}
#endif

void _palloc_page_use_delayed_free(palloc_page_t* page, palloc_delayed_t delay, bool override_never) {
  while (!_palloc_page_try_use_delayed_free(page, delay, override_never)) {
    palloc_atomic_yield();
  }
}

bool _palloc_page_try_use_delayed_free(palloc_page_t* page, palloc_delayed_t delay, bool override_never) {
  palloc_thread_free_t tfreex;
  palloc_delayed_t     old_delay;
  palloc_thread_free_t tfree;
  size_t yield_count = 0;
  do {
    tfree = palloc_atomic_load_acquire(&page->xthread_free); // note: must acquire as we can break/repeat this loop and not do a CAS;
    tfreex = palloc_tf_set_delayed(tfree, delay);
    old_delay = palloc_tf_delayed(tfree);
    if palloc_unlikely(old_delay == PALLOC_DELAYED_FREEING) {
      if (yield_count >= 4) return false;  // give up after 4 tries
      yield_count++;
      palloc_atomic_yield(); // delay until outstanding PALLOC_DELAYED_FREEING are done.
      // tfree = palloc_tf_set_delayed(tfree, PALLOC_NO_DELAYED_FREE); // will cause CAS to busy fail
    }
    else if (delay == old_delay) {
      break; // avoid atomic operation if already equal
    }
    else if (!override_never && old_delay == PALLOC_NEVER_DELAYED_FREE) {
      break; // leave never-delayed flag set
    }
  } while ((old_delay == PALLOC_DELAYED_FREEING) ||
           !palloc_atomic_cas_weak_release(&page->xthread_free, &tfree, tfreex));

  return true; // success
}

/* -----------------------------------------------------------
  Page collect the `local_free` and `thread_free` lists
----------------------------------------------------------- */

// Collect the local `thread_free` list using an atomic exchange.
// Note: The exchange must be done atomically as this is used right after
// moving to the full list in `palloc_page_collect_ex` and we need to
// ensure that there was no race where the page became unfull just before the move.
static void _palloc_page_thread_free_collect(palloc_page_t* page)
{
  palloc_block_t* head;
  palloc_thread_free_t tfreex;
  palloc_thread_free_t tfree = palloc_atomic_load_relaxed(&page->xthread_free);
  do {
    head = palloc_tf_block(tfree);
    tfreex = palloc_tf_set_block(tfree,NULL);
  } while (!palloc_atomic_cas_weak_acq_rel(&page->xthread_free, &tfree, tfreex));

  // return if the list is empty
  if (head == NULL) return;

  // find the tail -- also to get a proper count (without data races)
  size_t max_count = page->capacity; // cannot collect more than capacity
  size_t count = 1;
  palloc_block_t* tail = head;
  palloc_block_t* next;
  while ((next = palloc_block_next(page,tail)) != NULL && count <= max_count) {
    count++;
    tail = next;
  }
  // if `count > max_count` there was a memory corruption (possibly infinite list due to double multi-threaded free)
  if (count > max_count) {
    _palloc_error_message(EFAULT, "corrupted thread-free list\n");
    return; // the thread-free items cannot be freed
  }

  // and append the current local free list
  palloc_block_set_next(page,tail, page->local_free);
  page->local_free = head;

  // update counts now
  page->used -= (uint16_t)count;
}

void _palloc_page_free_collect(palloc_page_t* page, bool force) {
  palloc_assert_internal(page!=NULL);

  // collect the thread free list
  if (force || palloc_page_thread_free(page) != NULL) {  // quick test to avoid an atomic operation
    _palloc_page_thread_free_collect(page);
  }

  // and the local free list
  if (page->local_free != NULL) {
    if palloc_likely(page->free == NULL) {
      // usual case
      page->free = page->local_free;
      page->local_free = NULL;
      page->free_is_zero = false;
    }
    else if (force) {
      // append -- only on shutdown (force) as this is a linear operation
      palloc_block_t* tail = page->local_free;
      palloc_block_t* next;
      while ((next = palloc_block_next(page, tail)) != NULL) {
        tail = next;
      }
      palloc_block_set_next(page, tail, page->free);
      page->free = page->local_free;
      page->local_free = NULL;
      page->free_is_zero = false;
    }
  }

  palloc_assert_internal(!force || page->local_free == NULL);
}



/* -----------------------------------------------------------
  Page fresh and retire
----------------------------------------------------------- */

// called from segments when reclaiming abandoned pages
void _palloc_page_reclaim(palloc_heap_t* heap, palloc_page_t* page) {
  palloc_assert_expensive(palloc_page_is_valid_init(page));

  palloc_assert_internal(palloc_page_heap(page) == heap);
  palloc_assert_internal(palloc_page_thread_free_flag(page) != PALLOC_NEVER_DELAYED_FREE);
  #if PALLOC_HUGE_PAGE_ABANDON
  palloc_assert_internal(_palloc_page_segment(page)->kind != PALLOC_SEGMENT_HUGE);
  #endif

  // TODO: push on full queue immediately if it is full?
  palloc_page_queue_t* pq = palloc_page_queue(heap, palloc_page_block_size(page));
  palloc_page_queue_push(heap, pq, page);
  palloc_assert_expensive(_palloc_page_is_valid(page));
}

// allocate a fresh page from a segment
static palloc_page_t* palloc_page_fresh_alloc(palloc_heap_t* heap, palloc_page_queue_t* pq, size_t block_size, size_t page_alignment) {
  #if !PALLOC_HUGE_PAGE_ABANDON
  palloc_assert_internal(pq != NULL);
  palloc_assert_internal(palloc_heap_contains_queue(heap, pq));
  palloc_assert_internal(page_alignment > 0 || block_size > PALLOC_MEDIUM_OBJ_SIZE_MAX || block_size == pq->block_size);
  #endif
  palloc_page_t* page = _palloc_segment_page_alloc(heap, block_size, page_alignment, &heap->tld->segments);
  if (page == NULL) {
    // this may be out-of-memory, or an abandoned page was reclaimed (and in our queue)
    return NULL;
  }
  #if PALLOC_HUGE_PAGE_ABANDON
  palloc_assert_internal(pq==NULL || _palloc_page_segment(page)->page_kind != PALLOC_PAGE_HUGE);
  #endif
  palloc_assert_internal(page_alignment >0 || block_size > PALLOC_MEDIUM_OBJ_SIZE_MAX || _palloc_page_segment(page)->kind != PALLOC_SEGMENT_HUGE);
  palloc_assert_internal(pq!=NULL || palloc_page_block_size(page) >= block_size);
  // a fresh page was found, initialize it
  const size_t full_block_size = (pq == NULL || palloc_page_is_huge(page) ? palloc_page_block_size(page) : block_size); // see also: palloc_segment_huge_page_alloc
  palloc_assert_internal(full_block_size >= block_size);
  palloc_page_init(heap, page, full_block_size, heap->tld);
  palloc_heap_stat_increase(heap, pages, 1);
  palloc_heap_stat_increase(heap, page_bins[_palloc_page_stats_bin(page)], 1);
  if (pq != NULL) { palloc_page_queue_push(heap, pq, page); }
  palloc_assert_expensive(_palloc_page_is_valid(page));
  return page;
}

// Get a fresh page to use
static palloc_page_t* palloc_page_fresh(palloc_heap_t* heap, palloc_page_queue_t* pq) {
  palloc_assert_internal(palloc_heap_contains_queue(heap, pq));
  palloc_page_t* page = palloc_page_fresh_alloc(heap, pq, pq->block_size, 0);
  if (page==NULL) return NULL;
  palloc_assert_internal(pq->block_size==palloc_page_block_size(page));
  palloc_assert_internal(pq==palloc_page_queue(heap, palloc_page_block_size(page)));
  return page;
}

/* -----------------------------------------------------------
   Do any delayed frees
   (put there by other threads if they deallocated in a full page)
----------------------------------------------------------- */
void _palloc_heap_delayed_free_all(palloc_heap_t* heap) {
  while (!_palloc_heap_delayed_free_partial(heap)) {
    palloc_atomic_yield();
  }
}

// returns true if all delayed frees were processed
bool _palloc_heap_delayed_free_partial(palloc_heap_t* heap) {
  // take over the list (note: no atomic exchange since it is often NULL)
  palloc_block_t* block = palloc_atomic_load_ptr_relaxed(palloc_block_t, &heap->thread_delayed_free);
  while (block != NULL && !palloc_atomic_cas_ptr_weak_acq_rel(palloc_block_t, &heap->thread_delayed_free, &block, NULL)) { /* nothing */ };
  bool all_freed = true;

  // and free them all
  while(block != NULL) {
    palloc_block_t* next = palloc_block_nextx(heap,block, heap->keys);
    // use internal free instead of regular one to keep stats etc correct
    if (!_palloc_free_delayed_block(block)) {
      // we might already start delayed freeing while another thread has not yet
      // reset the delayed_freeing flag; in that case delay it further by reinserting the current block
      // into the delayed free list
      all_freed = false;
      palloc_block_t* dfree = palloc_atomic_load_ptr_relaxed(palloc_block_t, &heap->thread_delayed_free);
      do {
        palloc_block_set_nextx(heap, block, dfree, heap->keys);
      } while (!palloc_atomic_cas_ptr_weak_release(palloc_block_t,&heap->thread_delayed_free, &dfree, block));
    }
    block = next;
  }
  return all_freed;
}

/* -----------------------------------------------------------
  Unfull, abandon, free and retire
----------------------------------------------------------- */

// Move a page from the full list back to a regular list
void _palloc_page_unfull(palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  palloc_assert_expensive(_palloc_page_is_valid(page));
  palloc_assert_internal(palloc_page_is_in_full(page));
  if (!palloc_page_is_in_full(page)) return;

  palloc_heap_t* heap = palloc_page_heap(page);
  palloc_page_queue_t* pqfull = &heap->pages[PALLOC_BIN_FULL];
  palloc_page_set_in_full(page, false); // to get the right queue
  palloc_page_queue_t* pq = palloc_heap_page_queue_of(heap, page);
  palloc_page_set_in_full(page, true);
  palloc_page_queue_enqueue_from_full(pq, pqfull, page);
}

static void palloc_page_to_full(palloc_page_t* page, palloc_page_queue_t* pq) {
  palloc_assert_internal(pq == palloc_page_queue_of(page));
  palloc_assert_internal(!palloc_page_immediate_available(page));
  palloc_assert_internal(!palloc_page_is_in_full(page));

  if (palloc_page_is_in_full(page)) return;
  palloc_page_queue_enqueue_from(&palloc_page_heap(page)->pages[PALLOC_BIN_FULL], pq, page);
  _palloc_page_free_collect(page,false);  // try to collect right away in case another thread freed just before PALLOC_USE_DELAYED_FREE was set
}


// Abandon a page with used blocks at the end of a thread.
// Note: only call if it is ensured that no references exist from
// the `page->heap->thread_delayed_free` into this page.
// Currently only called through `palloc_heap_collect_ex` which ensures this.
void _palloc_page_abandon(palloc_page_t* page, palloc_page_queue_t* pq) {
  palloc_assert_internal(page != NULL);
  palloc_assert_expensive(_palloc_page_is_valid(page));
  palloc_assert_internal(pq == palloc_page_queue_of(page));
  palloc_assert_internal(palloc_page_heap(page) != NULL);

  palloc_heap_t* pheap = palloc_page_heap(page);

  // remove from our page list
  palloc_segments_tld_t* segments_tld = &pheap->tld->segments;
  palloc_page_queue_remove(pq, page);

  // page is no longer associated with our heap
  palloc_assert_internal(palloc_page_thread_free_flag(page)==PALLOC_NEVER_DELAYED_FREE);
  palloc_page_set_heap(page, NULL);

#if (PALLOC_DEBUG>1) && !PALLOC_TRACK_ENABLED
  // check there are no references left..
  for (palloc_block_t* block = (palloc_block_t*)pheap->thread_delayed_free; block != NULL; block = palloc_block_nextx(pheap, block, pheap->keys)) {
    palloc_assert_internal(_palloc_ptr_page(block) != page);
  }
#endif

  // and abandon it
  palloc_assert_internal(palloc_page_heap(page) == NULL);
  _palloc_segment_page_abandon(page,segments_tld);
}

// force abandon a page
void _palloc_page_force_abandon(palloc_page_t* page) {
  palloc_heap_t* heap = palloc_page_heap(page);
  // mark page as not using delayed free
  _palloc_page_use_delayed_free(page, PALLOC_NEVER_DELAYED_FREE, false);

  // ensure this page is no longer in the heap delayed free list
  _palloc_heap_delayed_free_all(heap);
  // We can still access the page meta-info even if it is freed as we ensure
  // in `palloc_segment_force_abandon` that the segment is not freed (yet)
  if (page->capacity == 0) return; // it may have been freed now

  // and now unlink it from the page queue and abandon (or free)
  palloc_page_queue_t* pq = palloc_heap_page_queue_of(heap, page);
  if (palloc_page_all_free(page)) {
    _palloc_page_free(page, pq, false);
  }
  else {
    _palloc_page_abandon(page, pq);
  }
}


// Free a page with no more free blocks
void _palloc_page_free(palloc_page_t* page, palloc_page_queue_t* pq, bool force) {
  palloc_assert_internal(page != NULL);
  palloc_assert_expensive(_palloc_page_is_valid(page));
  palloc_assert_internal(pq == palloc_page_queue_of(page));
  palloc_assert_internal(palloc_page_all_free(page));
  palloc_assert_internal(palloc_page_thread_free_flag(page)!=PALLOC_DELAYED_FREEING);

  // no more aligned blocks in here
  palloc_page_set_has_aligned(page, false);

  // remove from the page list
  // (no need to do _palloc_heap_delayed_free first as all blocks are already free)
  palloc_heap_t* heap = palloc_page_heap(page);
  palloc_segments_tld_t* segments_tld = &heap->tld->segments;
  palloc_page_queue_remove(pq, page);

  // and free it  
  palloc_page_set_heap(page,NULL);
  _palloc_segment_page_free(page, force, segments_tld);
}

#define PALLOC_MAX_RETIRE_SIZE    PALLOC_MEDIUM_OBJ_SIZE_MAX   // should be less than size for PALLOC_BIN_HUGE
#define PALLOC_RETIRE_CYCLES      (16)

// Retire a page with no more used blocks
// Important to not retire too quickly though as new
// allocations might coming.
// Note: called from `palloc_free` and benchmarks often
// trigger this due to freeing everything and then
// allocating again so careful when changing this.
void _palloc_page_retire(palloc_page_t* page) palloc_attr_noexcept {
  palloc_assert_internal(page != NULL);
  palloc_assert_expensive(_palloc_page_is_valid(page));
  palloc_assert_internal(palloc_page_all_free(page));

  palloc_page_set_has_aligned(page, false);

  // don't retire too often..
  // (or we end up retiring and re-allocating most of the time)
  // NOTE: refine this more: we should not retire if this
  // is the only page left with free blocks. It is not clear
  // how to check this efficiently though...
  // for now, we don't retire if it is the only page left of this size class.
  palloc_page_queue_t* pq = palloc_page_queue_of(page);
  #if PALLOC_RETIRE_CYCLES > 0
  const size_t bsize = palloc_page_block_size(page);
  if palloc_likely( /* bsize < PALLOC_MAX_RETIRE_SIZE && */ !palloc_page_queue_is_special(pq)) {  // not full or huge queue?
    if (pq->last==page && pq->first==page) { // the only page in the queue?
      palloc_stat_counter_increase(_palloc_stats_main.pages_retire,1);
      page->retire_expire = (bsize <= PALLOC_SMALL_OBJ_SIZE_MAX ? PALLOC_RETIRE_CYCLES : PALLOC_RETIRE_CYCLES/4);
      palloc_heap_t* heap = palloc_page_heap(page);
      palloc_assert_internal(pq >= heap->pages);
      const size_t index = pq - heap->pages;
      palloc_assert_internal(index < PALLOC_BIN_FULL && index < PALLOC_BIN_HUGE);
      if (index < heap->page_retired_min) heap->page_retired_min = index;
      if (index > heap->page_retired_max) heap->page_retired_max = index;
      palloc_assert_internal(palloc_page_all_free(page));
      return; // don't free after all
    }
  }
  #endif
  _palloc_page_free(page, pq, false);
}

// free retired pages: we don't need to look at the entire queues
// since we only retire pages that are at the head position in a queue.
void _palloc_heap_collect_retired(palloc_heap_t* heap, bool force) {
  size_t min = PALLOC_BIN_FULL;
  size_t max = 0;
  for(size_t bin = heap->page_retired_min; bin <= heap->page_retired_max; bin++) {
    palloc_page_queue_t* pq   = &heap->pages[bin];
    palloc_page_t*       page = pq->first;
    if (page != NULL && page->retire_expire != 0) {
      if (palloc_page_all_free(page)) {
        page->retire_expire--;
        if (force || page->retire_expire == 0) {
          _palloc_page_free(pq->first, pq, force);
        }
        else {
          // keep retired, update min/max
          if (bin < min) min = bin;
          if (bin > max) max = bin;
        }
      }
      else {
        page->retire_expire = 0;
      }
    }
  }
  heap->page_retired_min = min;
  heap->page_retired_max = max;
}


/* -----------------------------------------------------------
  Initialize the initial free list in a page.
  In secure mode we initialize a randomized list by
  alternating between slices.
----------------------------------------------------------- */

#define PALLOC_MAX_SLICE_SHIFT  (6)   // at most 64 slices
#define PALLOC_MAX_SLICES       (1UL << PALLOC_MAX_SLICE_SHIFT)
#define PALLOC_MIN_SLICES       (2)

static void palloc_page_free_list_extend_secure(palloc_heap_t* const heap, palloc_page_t* const page, const size_t bsize, const size_t extend, palloc_stats_t* const stats) {
  PALLOC_UNUSED(stats);
  #if (PALLOC_SECURE<=2)
  palloc_assert_internal(page->free == NULL);
  palloc_assert_internal(page->local_free == NULL);
  #endif
  palloc_assert_internal(page->capacity + extend <= page->reserved);
  palloc_assert_internal(bsize == palloc_page_block_size(page));
  void* const page_area = palloc_page_start(page);

  // initialize a randomized free list
  // set up `slice_count` slices to alternate between
  size_t shift = PALLOC_MAX_SLICE_SHIFT;
  while ((extend >> shift) == 0) {
    shift--;
  }
  const size_t slice_count = (size_t)1U << shift;
  const size_t slice_extend = extend / slice_count;
  palloc_assert_internal(slice_extend >= 1);
  palloc_block_t* blocks[PALLOC_MAX_SLICES];   // current start of the slice
  size_t      counts[PALLOC_MAX_SLICES];   // available objects in the slice
  for (size_t i = 0; i < slice_count; i++) {
    blocks[i] = palloc_page_block_at(page, page_area, bsize, page->capacity + i*slice_extend);
    counts[i] = slice_extend;
  }
  counts[slice_count-1] += (extend % slice_count);  // final slice holds the modulus too (todo: distribute evenly?)

  // and initialize the free list by randomly threading through them
  // set up first element
  const uintptr_t r = _palloc_heap_random_next(heap);
  size_t current = r % slice_count;
  counts[current]--;
  palloc_block_t* const free_start = blocks[current];
  // and iterate through the rest; use `random_shuffle` for performance
  uintptr_t rnd = _palloc_random_shuffle(r|1); // ensure not 0
  for (size_t i = 1; i < extend; i++) {
    // call random_shuffle only every INTPTR_SIZE rounds
    const size_t round = i%PALLOC_INTPTR_SIZE;
    if (round == 0) rnd = _palloc_random_shuffle(rnd);
    // select a random next slice index
    size_t next = ((rnd >> 8*round) & (slice_count-1));
    while (counts[next]==0) {                            // ensure it still has space
      next++;
      if (next==slice_count) next = 0;
    }
    // and link the current block to it
    counts[next]--;
    palloc_block_t* const block = blocks[current];
    blocks[current] = (palloc_block_t*)((uint8_t*)block + bsize);  // bump to the following block
    palloc_block_set_next(page, block, blocks[next]);   // and set next; note: we may have `current == next`
    current = next;
  }
  // prepend to the free list (usually NULL)
  palloc_block_set_next(page, blocks[current], page->free);  // end of the list
  page->free = free_start;
}

static palloc_decl_noinline void palloc_page_free_list_extend( palloc_page_t* const page, const size_t bsize, const size_t extend, palloc_stats_t* const stats)
{
  PALLOC_UNUSED(stats);
  #if (PALLOC_SECURE <= 2)
  palloc_assert_internal(page->free == NULL);
  palloc_assert_internal(page->local_free == NULL);
  #endif
  palloc_assert_internal(page->capacity + extend <= page->reserved);
  palloc_assert_internal(bsize == palloc_page_block_size(page));
  void* const page_area = palloc_page_start(page);

  palloc_block_t* const start = palloc_page_block_at(page, page_area, bsize, page->capacity);

  // initialize a sequential free list
  palloc_block_t* const last = palloc_page_block_at(page, page_area, bsize, page->capacity + extend - 1);
  palloc_block_t* block = start;
  while(block <= last) {
    palloc_block_t* next = (palloc_block_t*)((uint8_t*)block + bsize);
    palloc_block_set_next(page,block,next);
    block = next;
  }
  // prepend to free list (usually `NULL`)
  palloc_block_set_next(page, last, page->free);
  page->free = start;
}

/* -----------------------------------------------------------
  Page initialize and extend the capacity
----------------------------------------------------------- */

#define PALLOC_MAX_EXTEND_SIZE    (4*1024)      // heuristic, one OS page seems to work well.
#if (PALLOC_SECURE>0)
#define PALLOC_MIN_EXTEND         (8*PALLOC_SECURE) // extend at least by this many
#else
#define PALLOC_MIN_EXTEND         (4)
#endif

// Extend the capacity (up to reserved) by initializing a free list
// We do at most `PALLOC_MAX_EXTEND` to avoid touching too much memory
// Note: we also experimented with "bump" allocation on the first
// allocations but this did not speed up any benchmark (due to an
// extra test in malloc? or cache effects?)
static bool palloc_page_extend_free(palloc_heap_t* heap, palloc_page_t* page, palloc_tld_t* tld) {
  palloc_assert_expensive(palloc_page_is_valid_init(page));
  #if (PALLOC_SECURE<=2)
  palloc_assert(page->free == NULL);
  palloc_assert(page->local_free == NULL);
  if (page->free != NULL) return true;
  #endif
  if (page->capacity >= page->reserved) return true;

  palloc_stat_counter_increase(tld->stats.pages_extended, 1);

  // calculate the extend count
  const size_t bsize = palloc_page_block_size(page);
  size_t extend = page->reserved - page->capacity;
  palloc_assert_internal(extend > 0);

  size_t max_extend = (bsize >= PALLOC_MAX_EXTEND_SIZE ? PALLOC_MIN_EXTEND : PALLOC_MAX_EXTEND_SIZE/bsize);
  if (max_extend < PALLOC_MIN_EXTEND) { max_extend = PALLOC_MIN_EXTEND; }
  palloc_assert_internal(max_extend > 0);

  if (extend > max_extend) {
    // ensure we don't touch memory beyond the page to reduce page commit.
    // the `lean` benchmark tests this. Going from 1 to 8 increases rss by 50%.
    extend = max_extend;
  }

  palloc_assert_internal(extend > 0 && extend + page->capacity <= page->reserved);
  palloc_assert_internal(extend < (1UL<<16));

  // and append the extend the free list
  if (extend < PALLOC_MIN_SLICES || PALLOC_SECURE==0) { //!palloc_option_is_enabled(palloc_option_secure)) {
    palloc_page_free_list_extend(page, bsize, extend, &tld->stats );
  }
  else {
    palloc_page_free_list_extend_secure(heap, page, bsize, extend, &tld->stats);
  }
  // enable the new free list
  page->capacity += (uint16_t)extend;
  palloc_stat_increase(tld->stats.page_committed, extend * bsize);
  palloc_assert_expensive(palloc_page_is_valid_init(page));
  return true;
}

// Initialize a fresh page
static void palloc_page_init(palloc_heap_t* heap, palloc_page_t* page, size_t block_size, palloc_tld_t* tld) {
  palloc_assert(page != NULL);
  palloc_segment_t* segment = _palloc_page_segment(page);
  palloc_assert(segment != NULL);
  palloc_assert_internal(block_size > 0);
  // set fields
  palloc_page_set_heap(page, heap);
  page->block_size = block_size;
  size_t page_size;
  page->page_start = _palloc_segment_page_start(segment, page, &page_size);
  palloc_track_mem_noaccess(page->page_start,page_size);
  palloc_assert_internal(palloc_page_block_size(page) <= page_size);
  palloc_assert_internal(page_size <= page->slice_count*PALLOC_SEGMENT_SLICE_SIZE);
  palloc_assert_internal(page_size / block_size < (1L<<16));
  page->reserved = (uint16_t)(page_size / block_size);
  palloc_assert_internal(page->reserved > 0);
  #if (PALLOC_PADDING || PALLOC_ENCODE_FREELIST)
  page->keys[0] = _palloc_heap_random_next(heap);
  page->keys[1] = _palloc_heap_random_next(heap);
  #endif
  page->free_is_zero = page->is_zero_init;
  #if PALLOC_DEBUG>2
  if (page->is_zero_init) {
    palloc_track_mem_defined(page->page_start, page_size);
    palloc_assert_expensive(palloc_mem_is_zero(page->page_start, page_size));
  }
  #endif
  palloc_assert_internal(page->is_committed);
  if (block_size > 0 && _palloc_is_power_of_two(block_size)) {
    page->block_size_shift = (uint8_t)(palloc_ctz((uintptr_t)block_size));
  }
  else {
    page->block_size_shift = 0;
  }

  palloc_assert_internal(page->capacity == 0);
  palloc_assert_internal(page->free == NULL);
  palloc_assert_internal(page->used == 0);
  palloc_assert_internal(page->xthread_free == 0);
  palloc_assert_internal(page->next == NULL);
  palloc_assert_internal(page->prev == NULL);
  palloc_assert_internal(page->retire_expire == 0);
  palloc_assert_internal(!palloc_page_has_aligned(page));
  #if (PALLOC_PADDING || PALLOC_ENCODE_FREELIST)
  palloc_assert_internal(page->keys[0] != 0);
  palloc_assert_internal(page->keys[1] != 0);
  #endif
  palloc_assert_internal(page->block_size_shift == 0 || (block_size == ((size_t)1 << page->block_size_shift)));
  palloc_assert_expensive(palloc_page_is_valid_init(page));

  // initialize an initial free list
  if (palloc_page_extend_free(heap,page,tld)) {
    palloc_assert(palloc_page_immediate_available(page));
  }
  return;
}


/* -----------------------------------------------------------
  Find pages with free blocks
-------------------------------------------------------------*/

// search for a best next page to use for at most N pages (often cut short if immediate blocks are available)
#define PALLOC_MAX_CANDIDATE_SEARCH  (4)

// is the page not yet used up to its reserved space?
static bool palloc_page_is_expandable(const palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  palloc_assert_internal(page->capacity <= page->reserved);
  return (page->capacity < page->reserved);
}


// Find a page with free blocks of `page->block_size`.
static palloc_page_t* palloc_page_queue_find_free_ex(palloc_heap_t* heap, palloc_page_queue_t* pq, bool first_try)
{
  // search through the pages in "next fit" order
  #if PALLOC_STAT
  size_t count = 0;
  #endif
  size_t candidate_count = 0;        // we reset this on the first candidate to limit the search
  palloc_page_t* page_candidate = NULL;  // a page with free space
  palloc_page_t* page = pq->first;

  while (page != NULL)
  {
    palloc_page_t* next = page->next; // remember next
    #if PALLOC_STAT
    count++;
    #endif
    candidate_count++;

    // collect freed blocks by us and other threads
    _palloc_page_free_collect(page, false);

  #if PALLOC_MAX_CANDIDATE_SEARCH > 1
    // search up to N pages for a best candidate

    // is the local free list non-empty?
    const bool immediate_available = palloc_page_immediate_available(page);

    // if the page is completely full, move it to the `palloc_pages_full`
    // queue so we don't visit long-lived pages too often.
    if (!immediate_available && !palloc_page_is_expandable(page)) {
      palloc_assert_internal(!palloc_page_is_in_full(page) && !palloc_page_immediate_available(page));
      palloc_page_to_full(page, pq);
    }
    else {
      // the page has free space, make it a candidate
      // we prefer non-expandable pages with high usage as candidates (to reduce commit, and increase chances of free-ing up pages)
      if (page_candidate == NULL) {
        page_candidate = page;
        candidate_count = 0;
      }
      // prefer to reuse fuller pages (in the hope the less used page gets freed)
      else if (page->used >= page_candidate->used && !palloc_page_is_mostly_used(page) && !palloc_page_is_expandable(page)) {
        page_candidate = page;
      }
      // if we find a non-expandable candidate, or searched for N pages, return with the best candidate
      if (immediate_available || candidate_count > PALLOC_MAX_CANDIDATE_SEARCH) {
        palloc_assert_internal(page_candidate!=NULL);
        break;
      }
    }
  #else
    // first-fit algorithm
    // If the page contains free blocks, we are done
    if (palloc_page_immediate_available(page) || palloc_page_is_expandable(page)) {
      break;  // pick this one
    }

    // If the page is completely full, move it to the `palloc_pages_full`
    // queue so we don't visit long-lived pages too often.
    palloc_assert_internal(!palloc_page_is_in_full(page) && !palloc_page_immediate_available(page));
    palloc_page_to_full(page, pq);
  #endif

    page = next;
  } // for each page

  palloc_heap_stat_counter_increase(heap, page_searches, count);
  palloc_heap_stat_counter_increase(heap, page_searches_count, 1);

  // set the page to the best candidate
  if (page_candidate != NULL) {
    page = page_candidate;
  }
  if (page != NULL) {
    if (!palloc_page_immediate_available(page)) {
      palloc_assert_internal(palloc_page_is_expandable(page));
      if (!palloc_page_extend_free(heap, page, heap->tld)) {
        page = NULL; // failed to extend
      }
    }
    palloc_assert_internal(page == NULL || palloc_page_immediate_available(page));
  }

  if (page == NULL) {
    _palloc_heap_collect_retired(heap, false); // perhaps make a page available?
    page = palloc_page_fresh(heap, pq);
    if (page == NULL && first_try) {
      // out-of-memory _or_ an abandoned page with free blocks was reclaimed, try once again
      page = palloc_page_queue_find_free_ex(heap, pq, false);
    }
  }
  else {
    // move the page to the front of the queue
    palloc_page_queue_move_to_front(heap, pq, page);
    page->retire_expire = 0;
    // _palloc_heap_collect_retired(heap, false); // update retire counts; note: increases rss on MemoryLoad bench so don't do this
  }
  palloc_assert_internal(page == NULL || palloc_page_immediate_available(page));


  return page;
}



// Find a page with free blocks of `size`.
static inline palloc_page_t* palloc_find_free_page(palloc_heap_t* heap, size_t size) {
  palloc_page_queue_t* pq = palloc_page_queue(heap, size);

  // check the first page: we even do this with candidate search or otherwise we re-search every time
  palloc_page_t* page = pq->first;
  if (page != NULL) {
   #if (PALLOC_SECURE>=3) // in secure mode, we extend half the time to increase randomness
    if (page->capacity < page->reserved && ((_palloc_heap_random_next(heap) & 1) == 1)) {
      palloc_page_extend_free(heap, page, heap->tld);
      palloc_assert_internal(palloc_page_immediate_available(page));
    }
    else
   #endif
    {
      _palloc_page_free_collect(page,false);
    }

    if (palloc_page_immediate_available(page)) {
      page->retire_expire = 0;
      return page; // fast path
    }
  }

  return palloc_page_queue_find_free_ex(heap, pq, true);
}


/* -----------------------------------------------------------
  Users can register a deferred free function called
  when the `free` list is empty. Since the `local_free`
  is separate this is deterministically called after
  a certain number of allocations.
----------------------------------------------------------- */

static palloc_deferred_free_fun* volatile deferred_free = NULL;
static _Atomic(void*) deferred_arg; // = NULL

void _palloc_deferred_free(palloc_heap_t* heap, bool force) {
  heap->tld->heartbeat++;
  if (deferred_free != NULL && !heap->tld->recurse) {
    heap->tld->recurse = true;
    deferred_free(force, heap->tld->heartbeat, palloc_atomic_load_ptr_relaxed(void,&deferred_arg));
    heap->tld->recurse = false;
  }
}

void palloc_register_deferred_free(palloc_deferred_free_fun* fn, void* arg) palloc_attr_noexcept {
  deferred_free = fn;
  palloc_atomic_store_ptr_release(void,&deferred_arg, arg);
}


/* -----------------------------------------------------------
  General allocation
----------------------------------------------------------- */

// Large and huge page allocation.
// Huge pages contain just one block, and the segment contains just that page (as `PALLOC_SEGMENT_HUGE`).
// Huge pages are also use if the requested alignment is very large (> PALLOC_BLOCK_ALIGNMENT_MAX)
// so their size is not always `> PALLOC_LARGE_OBJ_SIZE_MAX`.
static palloc_page_t* palloc_large_huge_page_alloc(palloc_heap_t* heap, size_t size, size_t page_alignment) {
  size_t block_size = _palloc_os_good_alloc_size(size);
  palloc_assert_internal(palloc_bin(block_size) == PALLOC_BIN_HUGE || page_alignment > 0);
  bool is_huge = (block_size > PALLOC_LARGE_OBJ_SIZE_MAX || page_alignment > 0);
  #if PALLOC_HUGE_PAGE_ABANDON
  palloc_page_queue_t* pq = (is_huge ? NULL : palloc_page_queue(heap, block_size));
  #else
  palloc_page_queue_t* pq = palloc_page_queue(heap, is_huge ? PALLOC_LARGE_OBJ_SIZE_MAX+1 : block_size);
  palloc_assert_internal(!is_huge || palloc_page_queue_is_huge(pq));
  #endif
  palloc_page_t* page = palloc_page_fresh_alloc(heap, pq, block_size, page_alignment);
  if (page != NULL) {
    palloc_assert_internal(palloc_page_immediate_available(page));

    if (is_huge) {
      palloc_assert_internal(palloc_page_is_huge(page));
      palloc_assert_internal(_palloc_page_segment(page)->kind == PALLOC_SEGMENT_HUGE);
      palloc_assert_internal(_palloc_page_segment(page)->used==1);
      #if PALLOC_HUGE_PAGE_ABANDON
      palloc_assert_internal(_palloc_page_segment(page)->thread_id==0); // abandoned, not in the huge queue
      palloc_page_set_heap(page, NULL);
      #endif
    }
    else {
      palloc_assert_internal(!palloc_page_is_huge(page));
    }

    const size_t bsize = palloc_page_usable_block_size(page);  // note: not `palloc_page_block_size` to account for padding
    /*if (bsize <= PALLOC_LARGE_OBJ_SIZE_MAX) {
      palloc_heap_stat_increase(heap, malloc_large, bsize);
      palloc_heap_stat_counter_increase(heap, malloc_large_count, 1);
    }
    else */
    {
      _palloc_stat_increase(&heap->tld->stats.malloc_huge, bsize);
      _palloc_stat_counter_increase(&heap->tld->stats.malloc_huge_count, 1);
    }
  }
  return page;
}


// Allocate a page
// Note: in debug mode the size includes PALLOC_PADDING_SIZE and might have overflowed.
static palloc_page_t* palloc_find_page(palloc_heap_t* heap, size_t size, size_t huge_alignment) palloc_attr_noexcept {
  // huge allocation?
  const size_t req_size = size - PALLOC_PADDING_SIZE;  // correct for padding_size in case of an overflow on `size`
  if palloc_unlikely(req_size > (PALLOC_MEDIUM_OBJ_SIZE_MAX - PALLOC_PADDING_SIZE) || huge_alignment > 0) {
    if palloc_unlikely(req_size > PALLOC_MAX_ALLOC_SIZE) {
      _palloc_error_message(EOVERFLOW, "allocation request is too large (%zu bytes)\n", req_size);
      return NULL;
    }
    else {
      return palloc_large_huge_page_alloc(heap,size,huge_alignment);
    }
  }
  else {
    // otherwise find a page with free blocks in our size segregated queues
    #if PALLOC_PADDING
    palloc_assert_internal(size >= PALLOC_PADDING_SIZE);
    #endif
    return palloc_find_free_page(heap, size);
  }
}

// Generic allocation routine if the fast path (`alloc.c:palloc_page_malloc`) does not succeed.
// Note: in debug mode the size includes PALLOC_PADDING_SIZE and might have overflowed.
// The `huge_alignment` is normally 0 but is set to a multiple of PALLOC_SLICE_SIZE for
// very large requested alignments in which case we use a huge singleton page.
void* _palloc_malloc_generic(palloc_heap_t* heap, size_t size, bool zero, size_t huge_alignment, size_t* usable) palloc_attr_noexcept
{
  palloc_assert_internal(heap != NULL);

  // initialize if necessary
  if palloc_unlikely(!palloc_heap_is_initialized(heap)) {
    heap = palloc_heap_get_default(); // calls palloc_thread_init
    if palloc_unlikely(!palloc_heap_is_initialized(heap)) { return NULL; }
  }
  palloc_assert_internal(palloc_heap_is_initialized(heap));

  // do administrative tasks every N generic mallocs
  if palloc_unlikely(++heap->generic_count >= 100) {
    heap->generic_collect_count += heap->generic_count;
    heap->generic_count = 0;
    // call potential deferred free routines
    _palloc_deferred_free(heap, false);

    // free delayed frees from other threads (but skip contended ones)
    _palloc_heap_delayed_free_partial(heap);

    // collect every once in a while (10000 by default)
    const long generic_collect = palloc_option_get_clamp(palloc_option_generic_collect, 1, 1000000L);
    if (heap->generic_collect_count >= generic_collect) {
      heap->generic_collect_count = 0;
      palloc_heap_collect(heap, false /* force? */);
    }
  }

  // find (or allocate) a page of the right size
  palloc_page_t* page = palloc_find_page(heap, size, huge_alignment);
  if palloc_unlikely(page == NULL) { // first time out of memory, try to collect and retry the allocation once more
    palloc_heap_collect(heap, true /* force */);
    page = palloc_find_page(heap, size, huge_alignment);
  }

  if palloc_unlikely(page == NULL) { // out of memory
    const size_t req_size = size - PALLOC_PADDING_SIZE;  // correct for padding_size in case of an overflow on `size`
    _palloc_error_message(ENOMEM, "unable to allocate memory (%zu bytes)\n", req_size);
    return NULL;
  }

  palloc_assert_internal(palloc_page_immediate_available(page));
  palloc_assert_internal(palloc_page_block_size(page) >= size);

  // and try again, this time succeeding! (i.e. this should never recurse through _palloc_page_malloc)
  void* const p = _palloc_page_malloc_zero(heap, page, size, zero, usable);
  palloc_assert_internal(p != NULL);

  // move singleton pages to the full queue
  if (page->reserved == page->used) {
    palloc_page_to_full(page, palloc_page_queue_of(page));
  }
  return p;
}
