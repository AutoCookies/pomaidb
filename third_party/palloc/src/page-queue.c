/*----------------------------------------------------------------------------
Copyright (c) 2018-2024, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* -----------------------------------------------------------
  Definition of page queues for each block size
----------------------------------------------------------- */

#ifndef PALLOC_IN_PAGE_C
#error "this file should be included from 'page.c'"
// include to help an IDE
#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"
#endif

/* -----------------------------------------------------------
  Minimal alignment in machine words (i.e. `sizeof(void*)`)
----------------------------------------------------------- */

#if (PALLOC_MAX_ALIGN_SIZE > 4*PALLOC_INTPTR_SIZE)
  #error "define alignment for more than 4x word size for this platform"
#elif (PALLOC_MAX_ALIGN_SIZE > 2*PALLOC_INTPTR_SIZE)
  #define PALLOC_ALIGN4W   // 4 machine words minimal alignment
#elif (PALLOC_MAX_ALIGN_SIZE > PALLOC_INTPTR_SIZE)
  #define PALLOC_ALIGN2W   // 2 machine words minimal alignment
#else
  // ok, default alignment is 1 word
#endif


/* -----------------------------------------------------------
  Queue query
----------------------------------------------------------- */


static inline bool palloc_page_queue_is_huge(const palloc_page_queue_t* pq) {
  return (pq->block_size == (PALLOC_MEDIUM_OBJ_SIZE_MAX+sizeof(uintptr_t)));
}

static inline bool palloc_page_queue_is_full(const palloc_page_queue_t* pq) {
  return (pq->block_size == (PALLOC_MEDIUM_OBJ_SIZE_MAX+(2*sizeof(uintptr_t))));
}

static inline bool palloc_page_queue_is_special(const palloc_page_queue_t* pq) {
  return (pq->block_size > PALLOC_MEDIUM_OBJ_SIZE_MAX);
}

/* -----------------------------------------------------------
  Bins
----------------------------------------------------------- */

// Return the bin for a given field size.
// Returns PALLOC_BIN_HUGE if the size is too large.
// We use `wsize` for the size in "machine word sizes",
// i.e. byte size == `wsize*sizeof(void*)`.
static inline size_t palloc_bin(size_t size) {
  size_t wsize = _palloc_wsize_from_size(size);
#if defined(PALLOC_ALIGN4W)
  if palloc_likely(wsize <= 4) {
    return (wsize <= 1 ? 1 : (wsize+1)&~1); // round to double word sizes
  }
#elif defined(PALLOC_ALIGN2W)
  if palloc_likely(wsize <= 8) {
    return (wsize <= 1 ? 1 : (wsize+1)&~1); // round to double word sizes
  }
#else
  if palloc_likely(wsize <= 8) {
    return (wsize == 0 ? 1 : wsize);
  }
#endif
  else if palloc_unlikely(wsize > PALLOC_MEDIUM_OBJ_WSIZE_MAX) {
    return PALLOC_BIN_HUGE;
  }
  else {
    #if defined(PALLOC_ALIGN4W)
    if (wsize <= 16) { wsize = (wsize+3)&~3; } // round to 4x word sizes
    #endif
    wsize--;
    // find the highest bit
    const size_t b = (PALLOC_SIZE_BITS - 1 - palloc_clz(wsize));  // note: wsize != 0
    // and use the top 3 bits to determine the bin (~12.5% worst internal fragmentation).
    // - adjust with 3 because we use do not round the first 8 sizes
    //   which each get an exact bin
    const size_t bin = ((b << 2) + ((wsize >> (b - 2)) & 0x03)) - 3;
    palloc_assert_internal(bin > 0 && bin < PALLOC_BIN_HUGE);
    return bin;
  }
}



/* -----------------------------------------------------------
  Queue of pages with free blocks
----------------------------------------------------------- */

size_t _palloc_bin(size_t size) {
  return palloc_bin(size);
}

size_t _palloc_bin_size(size_t bin) {
  return _palloc_heap_empty.pages[bin].block_size;
}

// Good size for allocation
size_t palloc_good_size(size_t size) palloc_attr_noexcept {
  if (size <= PALLOC_MEDIUM_OBJ_SIZE_MAX) {
    return _palloc_bin_size(palloc_bin(size + PALLOC_PADDING_SIZE));
  }
  else {
    return _palloc_align_up(size + PALLOC_PADDING_SIZE,_palloc_os_page_size());
  }
}

#if (PALLOC_DEBUG>1)
static bool palloc_page_queue_contains(palloc_page_queue_t* queue, const palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  palloc_page_t* list = queue->first;
  while (list != NULL) {
    palloc_assert_internal(list->next == NULL || list->next->prev == list);
    palloc_assert_internal(list->prev == NULL || list->prev->next == list);
    if (list == page) break;
    list = list->next;
  }
  return (list == page);
}

#endif

#if (PALLOC_DEBUG>1)
static bool palloc_heap_contains_queue(const palloc_heap_t* heap, const palloc_page_queue_t* pq) {
  return (pq >= &heap->pages[0] && pq <= &heap->pages[PALLOC_BIN_FULL]);
}
#endif

static inline bool palloc_page_is_large_or_huge(const palloc_page_t* page) {
  return (palloc_page_block_size(page) > PALLOC_MEDIUM_OBJ_SIZE_MAX || palloc_page_is_huge(page));
}

static size_t palloc_page_bin(const palloc_page_t* page) {
  const size_t bin = (palloc_page_is_in_full(page) ? PALLOC_BIN_FULL : (palloc_page_is_huge(page) ? PALLOC_BIN_HUGE : palloc_bin(palloc_page_block_size(page))));
  palloc_assert_internal(bin <= PALLOC_BIN_FULL);
  return bin;
}

// returns the page bin without using PALLOC_BIN_FULL for statistics
size_t _palloc_page_stats_bin(const palloc_page_t* page) {
  const size_t bin = (palloc_page_is_huge(page) ? PALLOC_BIN_HUGE : palloc_bin(palloc_page_block_size(page)));
  palloc_assert_internal(bin <= PALLOC_BIN_HUGE);
  return bin;
}

static palloc_page_queue_t* palloc_heap_page_queue_of(palloc_heap_t* heap, const palloc_page_t* page) {
  palloc_assert_internal(heap!=NULL);
  const size_t bin = palloc_page_bin(page);
  palloc_page_queue_t* pq = &heap->pages[bin];
  palloc_assert_internal((palloc_page_block_size(page) == pq->block_size) ||
                       (palloc_page_is_large_or_huge(page) && palloc_page_queue_is_huge(pq)) ||
                         (palloc_page_is_in_full(page) && palloc_page_queue_is_full(pq)));
  return pq;
}

static palloc_page_queue_t* palloc_page_queue_of(const palloc_page_t* page) {
  palloc_heap_t* heap = palloc_page_heap(page);
  palloc_page_queue_t* pq = palloc_heap_page_queue_of(heap, page);
  palloc_assert_expensive(palloc_page_queue_contains(pq, page));
  return pq;
}

// The current small page array is for efficiency and for each
// small size (up to 256) it points directly to the page for that
// size without having to compute the bin. This means when the
// current free page queue is updated for a small bin, we need to update a
// range of entries in `_palloc_page_small_free`.
static inline void palloc_heap_queue_first_update(palloc_heap_t* heap, const palloc_page_queue_t* pq) {
  palloc_assert_internal(palloc_heap_contains_queue(heap,pq));
  size_t size = pq->block_size;
  if (size > PALLOC_SMALL_SIZE_MAX) return;

  palloc_page_t* page = pq->first;
  if (pq->first == NULL) page = (palloc_page_t*)&_palloc_page_empty;

  // find index in the right direct page array
  size_t start;
  size_t idx = _palloc_wsize_from_size(size);
  palloc_page_t** pages_free = heap->pages_free_direct;

  if (pages_free[idx] == page) return;  // already set

  // find start slot
  if (idx<=1) {
    start = 0;
  }
  else {
    // find previous size; due to minimal alignment upto 3 previous bins may need to be skipped
    size_t bin = palloc_bin(size);
    const palloc_page_queue_t* prev = pq - 1;
    while( bin == palloc_bin(prev->block_size) && prev > &heap->pages[0]) {
      prev--;
    }
    start = 1 + _palloc_wsize_from_size(prev->block_size);
    if (start > idx) start = idx;
  }

  // set size range to the right page
  palloc_assert(start <= idx);
  for (size_t sz = start; sz <= idx; sz++) {
    pages_free[sz] = page;
  }
}

/*
static bool palloc_page_queue_is_empty(palloc_page_queue_t* queue) {
  return (queue->first == NULL);
}
*/

static void palloc_page_queue_remove(palloc_page_queue_t* queue, palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  palloc_assert_expensive(palloc_page_queue_contains(queue, page));
  palloc_assert_internal(palloc_page_block_size(page) == queue->block_size ||
                      (palloc_page_is_large_or_huge(page) && palloc_page_queue_is_huge(queue)) ||
                        (palloc_page_is_in_full(page) && palloc_page_queue_is_full(queue)));
  palloc_heap_t* heap = palloc_page_heap(page);

  if (page->prev != NULL) page->prev->next = page->next;
  if (page->next != NULL) page->next->prev = page->prev;
  if (page == queue->last)  queue->last = page->prev;
  if (page == queue->first) {
    queue->first = page->next;
    // update first
    palloc_assert_internal(palloc_heap_contains_queue(heap, queue));
    palloc_heap_queue_first_update(heap,queue);
  }
  heap->page_count--;
  page->next = NULL;
  page->prev = NULL;
  // palloc_atomic_store_ptr_release(palloc_atomic_cast(void*, &page->heap), NULL);
  palloc_page_set_in_full(page,false);
}


static void palloc_page_queue_push(palloc_heap_t* heap, palloc_page_queue_t* queue, palloc_page_t* page) {
  palloc_assert_internal(palloc_page_heap(page) == heap);
  palloc_assert_internal(!palloc_page_queue_contains(queue, page));
  #if PALLOC_HUGE_PAGE_ABANDON
  palloc_assert_internal(_palloc_page_segment(page)->kind != PALLOC_SEGMENT_HUGE);
  #endif
  palloc_assert_internal(palloc_page_block_size(page) == queue->block_size ||
                      (palloc_page_is_large_or_huge(page) && palloc_page_queue_is_huge(queue)) ||
                        (palloc_page_is_in_full(page) && palloc_page_queue_is_full(queue)));

  palloc_page_set_in_full(page, palloc_page_queue_is_full(queue));
  // palloc_atomic_store_ptr_release(palloc_atomic_cast(void*, &page->heap), heap);
  page->next = queue->first;
  page->prev = NULL;
  if (queue->first != NULL) {
    palloc_assert_internal(queue->first->prev == NULL);
    queue->first->prev = page;
    queue->first = page;
  }
  else {
    queue->first = queue->last = page;
  }

  // update direct
  palloc_heap_queue_first_update(heap, queue);
  heap->page_count++;
}

static void palloc_page_queue_move_to_front(palloc_heap_t* heap, palloc_page_queue_t* queue, palloc_page_t* page) {
  palloc_assert_internal(palloc_page_heap(page) == heap);
  palloc_assert_internal(palloc_page_queue_contains(queue, page));
  if (queue->first == page) return;
  palloc_page_queue_remove(queue, page);
  palloc_page_queue_push(heap, queue, page);
  palloc_assert_internal(queue->first == page);
}

static void palloc_page_queue_enqueue_from_ex(palloc_page_queue_t* to, palloc_page_queue_t* from, bool enqueue_at_end, palloc_page_t* page) {
  palloc_assert_internal(page != NULL);
  palloc_assert_expensive(palloc_page_queue_contains(from, page));
  palloc_assert_expensive(!palloc_page_queue_contains(to, page));
  const size_t bsize = palloc_page_block_size(page);
  PALLOC_UNUSED(bsize);
  palloc_assert_internal((bsize == to->block_size && bsize == from->block_size) ||
                     (bsize == to->block_size && palloc_page_queue_is_full(from)) ||
                     (bsize == from->block_size && palloc_page_queue_is_full(to)) ||
                     (palloc_page_is_large_or_huge(page) && palloc_page_queue_is_huge(to)) ||
                     (palloc_page_is_large_or_huge(page) && palloc_page_queue_is_full(to)));

  palloc_heap_t* heap = palloc_page_heap(page);

  // delete from `from`
  if (page->prev != NULL) page->prev->next = page->next;
  if (page->next != NULL) page->next->prev = page->prev;
  if (page == from->last)  from->last = page->prev;
  if (page == from->first) {
    from->first = page->next;
    // update first
    palloc_assert_internal(palloc_heap_contains_queue(heap, from));
    palloc_heap_queue_first_update(heap, from);
  }

  // insert into `to`
  if (enqueue_at_end) {
    // enqueue at the end
    page->prev = to->last;
    page->next = NULL;
    if (to->last != NULL) {
      palloc_assert_internal(heap == palloc_page_heap(to->last));
      to->last->next = page;
      to->last = page;
    }
    else {
      to->first = page;
      to->last = page;
      palloc_heap_queue_first_update(heap, to);
    }
  }
  else {
    if (to->first != NULL) {
      // enqueue at 2nd place
      palloc_assert_internal(heap == palloc_page_heap(to->first));
      palloc_page_t* next = to->first->next;
      page->prev = to->first;
      page->next = next;
      to->first->next = page;
      if (next != NULL) {
        next->prev = page;
      }
      else {
        to->last = page;
      }
    }
    else {
      // enqueue at the head (singleton list)
      page->prev = NULL;
      page->next = NULL;
      to->first = page;
      to->last = page;
      palloc_heap_queue_first_update(heap, to);
    }
  }

  palloc_page_set_in_full(page, palloc_page_queue_is_full(to));
}

static void palloc_page_queue_enqueue_from(palloc_page_queue_t* to, palloc_page_queue_t* from, palloc_page_t* page) {
  palloc_page_queue_enqueue_from_ex(to, from, true /* enqueue at the end */, page);
}

static void palloc_page_queue_enqueue_from_full(palloc_page_queue_t* to, palloc_page_queue_t* from, palloc_page_t* page) {
  // note: we could insert at the front to increase reuse, but it slows down certain benchmarks (like `alloc-test`)
  palloc_page_queue_enqueue_from_ex(to, from, true /* enqueue at the end of the `to` queue? */, page);
}

// Only called from `palloc_heap_absorb`.
size_t _palloc_page_queue_append(palloc_heap_t* heap, palloc_page_queue_t* pq, palloc_page_queue_t* append) {
  palloc_assert_internal(palloc_heap_contains_queue(heap,pq));
  palloc_assert_internal(pq->block_size == append->block_size);

  if (append->first==NULL) return 0;

  // set append pages to new heap and count
  size_t count = 0;
  for (palloc_page_t* page = append->first; page != NULL; page = page->next) {
    // inline `palloc_page_set_heap` to avoid wrong assertion during absorption;
    // in this case it is ok to be delayed freeing since both "to" and "from" heap are still alive.
    palloc_atomic_store_release(&page->xheap, (uintptr_t)heap);
    // set the flag to delayed free (not overriding NEVER_DELAYED_FREE) which has as a
    // side effect that it spins until any DELAYED_FREEING is finished. This ensures
    // that after appending only the new heap will be used for delayed free operations.
    _palloc_page_use_delayed_free(page, PALLOC_USE_DELAYED_FREE, false);
    count++;
  }

  if (pq->last==NULL) {
    // take over afresh
    palloc_assert_internal(pq->first==NULL);
    pq->first = append->first;
    pq->last = append->last;
    palloc_heap_queue_first_update(heap, pq);
  }
  else {
    // append to end
    palloc_assert_internal(pq->last!=NULL);
    palloc_assert_internal(append->first!=NULL);
    pq->last->next = append->first;
    append->first->prev = pq->last;
    pq->last = append->last;
  }
  return count;
}
