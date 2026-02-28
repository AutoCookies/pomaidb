/* ----------------------------------------------------------------------------
Copyright (c) 2018-2024, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#if !defined(PALLOC_IN_ALLOC_C)
#error "this file should be included from 'alloc.c' (so aliases can work from alloc-override)"
// add includes help an IDE
#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/prim.h"   // _palloc_prim_thread_id()
#endif

// forward declarations
static void   palloc_check_padding(const palloc_page_t* page, const palloc_block_t* block);
static bool   palloc_check_is_double_free(const palloc_page_t* page, const palloc_block_t* block);
static size_t palloc_page_usable_size_of(const palloc_page_t* page, const palloc_block_t* block);
static void   palloc_stat_free(const palloc_page_t* page, const palloc_block_t* block);


// ------------------------------------------------------
// Free
// ------------------------------------------------------

// forward declaration of multi-threaded free (`_mt`) (or free in huge block if compiled with PALLOC_HUGE_PAGE_ABANDON)
static palloc_decl_noinline void palloc_free_block_mt(palloc_page_t* page, palloc_segment_t* segment, palloc_block_t* block);

// regular free of a (thread local) block pointer
// fast path written carefully to prevent spilling on the stack
static inline void palloc_free_block_local(palloc_page_t* page, palloc_block_t* block, bool track_stats, bool check_full)
{
  // checks
  if palloc_unlikely(palloc_check_is_double_free(page, block)) return;
  palloc_check_padding(page, block);
  if (track_stats) { palloc_stat_free(page, block); }
  #if (PALLOC_DEBUG>0) && !PALLOC_TRACK_ENABLED  && !PALLOC_TSAN && !PALLOC_GUARDED
  if (!palloc_page_is_huge(page)) {   // huge page content may be already decommitted
    memset(block, PALLOC_DEBUG_FREED, palloc_page_block_size(page));
  }
  #endif
  if (track_stats) { palloc_track_free_size(block, palloc_page_usable_size_of(page, block)); } // faster then palloc_usable_size as we already know the page and that p is unaligned

  // actual free: push on the local free list
  palloc_block_set_next(page, block, page->local_free);
  page->local_free = block;
  if palloc_unlikely(--page->used == 0) {
    _palloc_page_retire(page);
  }
  else if palloc_unlikely(check_full && palloc_page_is_in_full(page)) {
    _palloc_page_unfull(page);
  }
}

// Adjust a block that was allocated aligned, to the actual start of the block in the page.
// note: this can be called from `palloc_free_generic_mt` where a non-owning thread accesses the
// `page_start` and `block_size` fields; however these are constant and the page won't be
// deallocated (as the block we are freeing keeps it alive) and thus safe to read concurrently.
palloc_block_t* _palloc_page_ptr_unalign(const palloc_page_t* page, const void* p) {
  palloc_assert_internal(page!=NULL && p!=NULL);

  size_t diff = (uint8_t*)p - page->page_start;
  size_t adjust;
  if palloc_likely(page->block_size_shift != 0) {
    adjust = diff & (((size_t)1 << page->block_size_shift) - 1);
  }
  else {
    adjust = diff % palloc_page_block_size(page);
  }

  return (palloc_block_t*)((uintptr_t)p - adjust);
}

// forward declaration for a PALLOC_GUARDED build
#if PALLOC_GUARDED
static void palloc_block_unguard(palloc_page_t* page, palloc_block_t* block, void* p); // forward declaration
static inline void palloc_block_check_unguard(palloc_page_t* page, palloc_block_t* block, void* p) {
  if (palloc_block_ptr_is_guarded(block, p)) { palloc_block_unguard(page, block, p); }
}
#else
static inline void palloc_block_check_unguard(palloc_page_t* page, palloc_block_t* block, void* p) {
  PALLOC_UNUSED(page); PALLOC_UNUSED(block); PALLOC_UNUSED(p);
}
#endif

// free a local pointer  (page parameter comes first for better codegen)
static void palloc_decl_noinline palloc_free_generic_local(palloc_page_t* page, palloc_segment_t* segment, void* p) palloc_attr_noexcept {
  PALLOC_UNUSED(segment);
  palloc_block_t* const block = (palloc_page_has_aligned(page) ? _palloc_page_ptr_unalign(page, p) : (palloc_block_t*)p);
  palloc_block_check_unguard(page, block, p);
  palloc_free_block_local(page, block, true /* track stats */, true /* check for a full page */);
}

// free a pointer owned by another thread (page parameter comes first for better codegen)
static void palloc_decl_noinline palloc_free_generic_mt(palloc_page_t* page, palloc_segment_t* segment, void* p) palloc_attr_noexcept {
  palloc_block_t* const block = _palloc_page_ptr_unalign(page, p); // don't check `has_aligned` flag to avoid a race (issue #865)
  palloc_block_check_unguard(page, block, p);
  palloc_free_block_mt(page, segment, block);
}

// generic free (for runtime integration)
void palloc_decl_noinline _palloc_free_generic(palloc_segment_t* segment, palloc_page_t* page, bool is_local, void* p) palloc_attr_noexcept {
  if (is_local) palloc_free_generic_local(page,segment,p);
           else palloc_free_generic_mt(page,segment,p);
}

// Get the segment data belonging to a pointer
// This is just a single `and` in release mode but does further checks in debug mode
// (and secure mode) to see if this was a valid pointer.
static inline palloc_segment_t* palloc_checked_ptr_segment(const void* p, const char* msg)
{
  PALLOC_UNUSED(msg);

  #if (PALLOC_DEBUG>0)
  if palloc_unlikely(((uintptr_t)p & (PALLOC_INTPTR_SIZE - 1)) != 0 && !palloc_option_is_enabled(palloc_option_guarded_precise)) {
    _palloc_error_message(EINVAL, "%s: invalid (unaligned) pointer: %p\n", msg, p);
    return NULL;
  }
  #endif

  palloc_segment_t* const segment = _palloc_ptr_segment(p);
  if palloc_unlikely(segment==NULL) return segment;

  #if (PALLOC_DEBUG>0)
  if palloc_unlikely(!palloc_is_in_heap_region(p)) {
  #if (PALLOC_INTPTR_SIZE == 8 && defined(__linux__))
    if (((uintptr_t)p >> 40) != 0x7F) { // linux tends to align large blocks above 0x7F000000000 (issue #640)
  #else
    {
  #endif
      _palloc_warning_message("%s: pointer might not point to a valid heap region: %p\n"
        "(this may still be a valid very large allocation (over 64MiB))\n", msg, p);
      if palloc_likely(_palloc_ptr_cookie(segment) == segment->cookie) {
        _palloc_warning_message("(yes, the previous pointer %p was valid after all)\n", p);
      }
    }
  }
  #endif
  #if (PALLOC_DEBUG>0 || PALLOC_SECURE>=4)
  if palloc_unlikely(_palloc_ptr_cookie(segment) != segment->cookie) {
    _palloc_error_message(EINVAL, "%s: pointer does not point to a valid heap space: %p\n", msg, p);
    return NULL;
  }
  #endif

  return segment;
}

// Free a block
// Fast path written carefully to prevent register spilling on the stack
static inline void palloc_free_ex(void* p, size_t* usable) palloc_attr_noexcept
{
  palloc_segment_t* const segment = palloc_checked_ptr_segment(p,"palloc_free");
  if palloc_unlikely(segment==NULL) return;

  const bool is_local = (_palloc_prim_thread_id() == palloc_atomic_load_relaxed(&segment->thread_id));
  palloc_page_t* const page = _palloc_segment_page_of(segment, p);
  if (usable!=NULL) { *usable = palloc_page_usable_block_size(page); }
  
  if palloc_likely(is_local) {                        // thread-local free?
    if palloc_likely(page->flags.full_aligned == 0) { // and it is not a full page (full pages need to move from the full bin), nor has aligned blocks (aligned blocks need to be unaligned)
      // thread-local, aligned, and not a full page
      palloc_block_t* const block = (palloc_block_t*)p;
      palloc_free_block_local(page, block, true /* track stats */, false /* no need to check if the page is full */);
    }
    else {
      // page is full or contains (inner) aligned blocks; use generic path
      palloc_free_generic_local(page, segment, p);
    }
  }
  else {
    // not thread-local; use generic path
    palloc_free_generic_mt(page, segment, p);
  }
}

void palloc_free(void* p) palloc_attr_noexcept {
  palloc_free_ex(p,NULL);
}

void palloc_ufree(void* p, size_t* usable) palloc_attr_noexcept {
  palloc_free_ex(p,usable);
}

// return true if successful
bool _palloc_free_delayed_block(palloc_block_t* block) {
  // get segment and page
  palloc_assert_internal(block!=NULL);
  const palloc_segment_t* const segment = _palloc_ptr_segment(block);
  palloc_assert_internal(_palloc_ptr_cookie(segment) == segment->cookie);
  palloc_assert_internal(_palloc_thread_id() == segment->thread_id);
  palloc_page_t* const page = _palloc_segment_page_of(segment, block);

  // Clear the no-delayed flag so delayed freeing is used again for this page.
  // This must be done before collecting the free lists on this page -- otherwise
  // some blocks may end up in the page `thread_free` list with no blocks in the
  // heap `thread_delayed_free` list which may cause the page to be never freed!
  // (it would only be freed if we happen to scan it in `palloc_page_queue_find_free_ex`)
  if (!_palloc_page_try_use_delayed_free(page, PALLOC_USE_DELAYED_FREE, false /* dont overwrite never delayed */)) {
    return false;
  }

  // collect all other non-local frees (move from `thread_free` to `free`) to ensure up-to-date `used` count
  _palloc_page_free_collect(page, false);

  // and free the block (possibly freeing the page as well since `used` is updated)
  palloc_free_block_local(page, block, false /* stats have already been adjusted */, true /* check for a full page */);
  return true;
}

// ------------------------------------------------------
// Multi-threaded Free (`_mt`)
// ------------------------------------------------------

// Push a block that is owned by another thread on its page-local thread free
// list or it's heap delayed free list. Such blocks are later collected by
// the owning thread in `_palloc_free_delayed_block`.
static void palloc_decl_noinline palloc_free_block_delayed_mt( palloc_page_t* page, palloc_block_t* block )
{
  // Try to put the block on either the page-local thread free list,
  // or the heap delayed free list (if this is the first non-local free in that page)
  palloc_thread_free_t tfreex;
  bool use_delayed;
  palloc_thread_free_t tfree = palloc_atomic_load_relaxed(&page->xthread_free);
  do {
    use_delayed = (palloc_tf_delayed(tfree) == PALLOC_USE_DELAYED_FREE);
    if palloc_unlikely(use_delayed) {
      // unlikely: this only happens on the first concurrent free in a page that is in the full list
      tfreex = palloc_tf_set_delayed(tfree,PALLOC_DELAYED_FREEING);
    }
    else {
      // usual: directly add to page thread_free list
      palloc_block_set_next(page, block, palloc_tf_block(tfree));
      tfreex = palloc_tf_set_block(tfree,block);
    }
  } while (!palloc_atomic_cas_weak_release(&page->xthread_free, &tfree, tfreex));

  // If this was the first non-local free, we need to push it on the heap delayed free list instead
  if palloc_unlikely(use_delayed) {
    // racy read on `heap`, but ok because PALLOC_DELAYED_FREEING is set (see `palloc_heap_delete` and `palloc_heap_collect_abandon`)
    palloc_heap_t* const heap = (palloc_heap_t*)(palloc_atomic_load_acquire(&page->xheap)); //palloc_page_heap(page);
    palloc_assert_internal(heap != NULL);
    if (heap != NULL) {
      // add to the delayed free list of this heap. (do this atomically as the lock only protects heap memory validity)
      palloc_block_t* dfree = palloc_atomic_load_ptr_relaxed(palloc_block_t, &heap->thread_delayed_free);
      do {
        palloc_block_set_nextx(heap,block,dfree, heap->keys);
      } while (!palloc_atomic_cas_ptr_weak_release(palloc_block_t,&heap->thread_delayed_free, &dfree, block));
    }

    // and reset the PALLOC_DELAYED_FREEING flag
    tfree = palloc_atomic_load_relaxed(&page->xthread_free);
    do {
      tfreex = tfree;
      palloc_assert_internal(palloc_tf_delayed(tfree) == PALLOC_DELAYED_FREEING);
      tfreex = palloc_tf_set_delayed(tfree,PALLOC_NO_DELAYED_FREE);
    } while (!palloc_atomic_cas_weak_release(&page->xthread_free, &tfree, tfreex));
  }
}

// Multi-threaded free (`_mt`) (or free in huge block if compiled with PALLOC_HUGE_PAGE_ABANDON)
static void palloc_decl_noinline palloc_free_block_mt(palloc_page_t* page, palloc_segment_t* segment, palloc_block_t* block)
{
  // first see if the segment was abandoned and if we can reclaim it into our thread
  if (_palloc_option_get_fast(palloc_option_abandoned_reclaim_on_free) != 0 &&
      #if PALLOC_HUGE_PAGE_ABANDON
      segment->page_kind != PALLOC_PAGE_HUGE &&
      #endif
      palloc_atomic_load_relaxed(&segment->thread_id) == 0 &&  // segment is abandoned?
      palloc_prim_get_default_heap() != (palloc_heap_t*)&_palloc_heap_empty) // and we did not already exit this thread (without this check, a fresh heap will be initalized (issue #944))
  {
    // the segment is abandoned, try to reclaim it into our heap
    if (_palloc_segment_attempt_reclaim(palloc_heap_get_default(), segment)) {
      palloc_assert_internal(_palloc_thread_id() == palloc_atomic_load_relaxed(&segment->thread_id));
      palloc_assert_internal(palloc_heap_get_default()->tld->segments.subproc == segment->subproc);
      palloc_free(block);  // recursively free as now it will be a local free in our heap
      return;
    }
  }

  // The padding check may access the non-thread-owned page for the key values.
  // that is safe as these are constant and the page won't be freed (as the block is not freed yet).
  palloc_check_padding(page, block);

  // adjust stats (after padding check and potentially recursive `palloc_free` above)
  palloc_stat_free(page, block);    // stat_free may access the padding
  palloc_track_free_size(block, palloc_page_usable_size_of(page,block));

  // for small size, ensure we can fit the delayed thread pointers without triggering overflow detection
  _palloc_padding_shrink(page, block, sizeof(palloc_block_t));

  if (segment->kind == PALLOC_SEGMENT_HUGE) {
    #if PALLOC_HUGE_PAGE_ABANDON
    // huge page segments are always abandoned and can be freed immediately
    _palloc_segment_huge_page_free(segment, page, block);
    return;
    #else
    // huge pages are special as they occupy the entire segment
    // as these are large we reset the memory occupied by the page so it is available to other threads
    // (as the owning thread needs to actually free the memory later).
    _palloc_segment_huge_page_reset(segment, page, block);
    #endif
  }
  else {
    #if (PALLOC_DEBUG>0) && !PALLOC_TRACK_ENABLED  && !PALLOC_TSAN       // note: when tracking, cannot use palloc_usable_size with multi-threading
    memset(block, PALLOC_DEBUG_FREED, palloc_usable_size(block));
    #endif
  }

  // and finally free the actual block by pushing it on the owning heap
  // thread_delayed free list (or heap delayed free list)
  palloc_free_block_delayed_mt(page,block);
}


// ------------------------------------------------------
// Usable size
// ------------------------------------------------------

// Bytes available in a block
static size_t palloc_decl_noinline palloc_page_usable_aligned_size_of(const palloc_page_t* page, const void* p) palloc_attr_noexcept {
  const palloc_block_t* block = _palloc_page_ptr_unalign(page, p);
  const size_t size = palloc_page_usable_size_of(page, block);
  const ptrdiff_t adjust = (uint8_t*)p - (uint8_t*)block;
  palloc_assert_internal(adjust >= 0 && (size_t)adjust <= size);
  const size_t aligned_size = (size - adjust);
  #if PALLOC_GUARDED
  if (palloc_block_ptr_is_guarded(block, p)) {
    return aligned_size - _palloc_os_page_size();
  }
  #endif
  return aligned_size;
}

static inline palloc_page_t* palloc_validate_ptr_page(const void* p, const char* msg) {
  const palloc_segment_t* const segment = palloc_checked_ptr_segment(p, msg);
  if palloc_unlikely(segment==NULL) return NULL;
  palloc_page_t* const page = _palloc_segment_page_of(segment, p);
  return page;
}

static inline size_t _palloc_usable_size(const void* p, const palloc_page_t* page) palloc_attr_noexcept {
  if palloc_unlikely(page==NULL) return 0;
  if palloc_likely(!palloc_page_has_aligned(page)) {
    const palloc_block_t* block = (const palloc_block_t*)p;
    return palloc_page_usable_size_of(page, block);
  }
  else {
    // split out to separate routine for improved code generation
    return palloc_page_usable_aligned_size_of(page, p);
  }
}

palloc_decl_nodiscard size_t palloc_usable_size(const void* p) palloc_attr_noexcept {
  const palloc_page_t* const page = palloc_validate_ptr_page(p,"palloc_usable_size");
  return _palloc_usable_size(p,page);
}


// ------------------------------------------------------
// Free variants
// ------------------------------------------------------

void palloc_free_size(void* p, size_t size) palloc_attr_noexcept {
  PALLOC_UNUSED_RELEASE(size);
  #if PALLOC_DEBUG
  const palloc_page_t* const page = palloc_validate_ptr_page(p,"palloc_free_size");  
  const size_t available = _palloc_usable_size(p,page);
  palloc_assert(p == NULL || size <= available || available == 0 /* invalid pointer */ );
  #endif
  palloc_free(p);
}

void palloc_free_size_aligned(void* p, size_t size, size_t alignment) palloc_attr_noexcept {
  PALLOC_UNUSED_RELEASE(alignment);
  palloc_assert(((uintptr_t)p % alignment) == 0);
  palloc_free_size(p,size);
}

void palloc_free_aligned(void* p, size_t alignment) palloc_attr_noexcept {
  PALLOC_UNUSED_RELEASE(alignment);
  palloc_assert(((uintptr_t)p % alignment) == 0);
  palloc_free(p);
}


// ------------------------------------------------------
// Check for double free in secure and debug mode
// This is somewhat expensive so only enabled for secure mode 4
// ------------------------------------------------------

#if (PALLOC_ENCODE_FREELIST && (PALLOC_SECURE>=4 || PALLOC_DEBUG!=0))
// linear check if the free list contains a specific element
static bool palloc_list_contains(const palloc_page_t* page, const palloc_block_t* list, const palloc_block_t* elem) {
  while (list != NULL) {
    if (elem==list) return true;
    list = palloc_block_next(page, list);
  }
  return false;
}

static palloc_decl_noinline bool palloc_check_is_double_freex(const palloc_page_t* page, const palloc_block_t* block) {
  // The decoded value is in the same page (or NULL).
  // Walk the free lists to verify positively if it is already freed
  if (palloc_list_contains(page, page->free, block) ||
      palloc_list_contains(page, page->local_free, block) ||
      palloc_list_contains(page, palloc_page_thread_free(page), block))
  {
    _palloc_error_message(EAGAIN, "double free detected of block %p with size %zu\n", block, palloc_page_block_size(page));
    return true;
  }
  return false;
}

#define palloc_track_page(page,access)  { size_t psize; void* pstart = _palloc_page_start(_palloc_page_segment(page),page,&psize); palloc_track_mem_##access( pstart, psize); }

static inline bool palloc_check_is_double_free(const palloc_page_t* page, const palloc_block_t* block) {
  bool is_double_free = false;
  palloc_block_t* n = palloc_block_nextx(page, block, page->keys); // pretend it is freed, and get the decoded first field
  if (((uintptr_t)n & (PALLOC_INTPTR_SIZE-1))==0 &&  // quick check: aligned pointer?
      (n==NULL || palloc_is_in_same_page(block, n))) // quick check: in same page or NULL?
  {
    // Suspicious: decoded value a in block is in the same page (or NULL) -- maybe a double free?
    // (continue in separate function to improve code generation)
    is_double_free = palloc_check_is_double_freex(page, block);
  }
  return is_double_free;
}
#else
static inline bool palloc_check_is_double_free(const palloc_page_t* page, const palloc_block_t* block) {
  PALLOC_UNUSED(page);
  PALLOC_UNUSED(block);
  return false;
}
#endif


// ---------------------------------------------------------------------------
// Check for heap block overflow by setting up padding at the end of the block
// ---------------------------------------------------------------------------

#if PALLOC_PADDING // && !PALLOC_TRACK_ENABLED
static bool palloc_page_decode_padding(const palloc_page_t* page, const palloc_block_t* block, size_t* delta, size_t* bsize) {
  *bsize = palloc_page_usable_block_size(page);
  const palloc_padding_t* const padding = (palloc_padding_t*)((uint8_t*)block + *bsize);
  palloc_track_mem_defined(padding,sizeof(palloc_padding_t));
  *delta = padding->delta;
  uint32_t canary = padding->canary;
  uintptr_t keys[2];
  keys[0] = page->keys[0];
  keys[1] = page->keys[1];
  bool ok = (palloc_ptr_encode_canary(page,block,keys) == canary && *delta <= *bsize);
  palloc_track_mem_noaccess(padding,sizeof(palloc_padding_t));
  return ok;
}

// Return the exact usable size of a block.
static size_t palloc_page_usable_size_of(const palloc_page_t* page, const palloc_block_t* block) {
  size_t bsize;
  size_t delta;
  bool ok = palloc_page_decode_padding(page, block, &delta, &bsize);
  palloc_assert_internal(ok); palloc_assert_internal(delta <= bsize);
  return (ok ? bsize - delta : 0);
}

// When a non-thread-local block is freed, it becomes part of the thread delayed free
// list that is freed later by the owning heap. If the exact usable size is too small to
// contain the pointer for the delayed list, then shrink the padding (by decreasing delta)
// so it will later not trigger an overflow error in `palloc_free_block`.
void _palloc_padding_shrink(const palloc_page_t* page, const palloc_block_t* block, const size_t min_size) {
  size_t bsize;
  size_t delta;
  bool ok = palloc_page_decode_padding(page, block, &delta, &bsize);
  palloc_assert_internal(ok);
  if (!ok || (bsize - delta) >= min_size) return;  // usually already enough space
  palloc_assert_internal(bsize >= min_size);
  if (bsize < min_size) return;  // should never happen
  size_t new_delta = (bsize - min_size);
  palloc_assert_internal(new_delta < bsize);
  palloc_padding_t* padding = (palloc_padding_t*)((uint8_t*)block + bsize);
  palloc_track_mem_defined(padding,sizeof(palloc_padding_t));
  padding->delta = (uint32_t)new_delta;
  palloc_track_mem_noaccess(padding,sizeof(palloc_padding_t));
}
#else
static size_t palloc_page_usable_size_of(const palloc_page_t* page, const palloc_block_t* block) {
  PALLOC_UNUSED(block);
  return palloc_page_usable_block_size(page);
}

void _palloc_padding_shrink(const palloc_page_t* page, const palloc_block_t* block, const size_t min_size) {
  PALLOC_UNUSED(page);
  PALLOC_UNUSED(block);
  PALLOC_UNUSED(min_size);
}
#endif

#if PALLOC_PADDING && PALLOC_PADDING_CHECK

static bool palloc_verify_padding(const palloc_page_t* page, const palloc_block_t* block, size_t* size, size_t* wrong) {
  size_t bsize;
  size_t delta;
  bool ok = palloc_page_decode_padding(page, block, &delta, &bsize);
  *size = *wrong = bsize;
  if (!ok) return false;
  palloc_assert_internal(bsize >= delta);
  *size = bsize - delta;
  if (!palloc_page_is_huge(page)) {
    uint8_t* fill = (uint8_t*)block + bsize - delta;
    const size_t maxpad = (delta > PALLOC_MAX_ALIGN_SIZE ? PALLOC_MAX_ALIGN_SIZE : delta); // check at most the first N padding bytes
    palloc_track_mem_defined(fill, maxpad);
    for (size_t i = 0; i < maxpad; i++) {
      if (fill[i] != PALLOC_DEBUG_PADDING) {
        *wrong = bsize - delta + i;
        ok = false;
        break;
      }
    }
    palloc_track_mem_noaccess(fill, maxpad);
  }
  return ok;
}

static void palloc_check_padding(const palloc_page_t* page, const palloc_block_t* block) {
  size_t size;
  size_t wrong;
  if (!palloc_verify_padding(page,block,&size,&wrong)) {
    _palloc_error_message(EFAULT, "buffer overflow in heap block %p of size %zu: write after %zu bytes\n", block, size, wrong );
  }
}

#else

static void palloc_check_padding(const palloc_page_t* page, const palloc_block_t* block) {
  PALLOC_UNUSED(page);
  PALLOC_UNUSED(block);
}

#endif

// only maintain stats for smaller objects if requested
#if (PALLOC_STAT>0)
static void palloc_stat_free(const palloc_page_t* page, const palloc_block_t* block) {
  PALLOC_UNUSED(block);
  palloc_heap_t* const heap = palloc_heap_get_default();
  const size_t bsize = palloc_page_usable_block_size(page);
  // #if (PALLOC_STAT>1)
  // const size_t usize = palloc_page_usable_size_of(page, block);
  // palloc_heap_stat_decrease(heap, malloc_requested, usize);
  // #endif
  if (bsize <= PALLOC_MEDIUM_OBJ_SIZE_MAX) {
    palloc_heap_stat_decrease(heap, malloc_normal, bsize);
    #if (PALLOC_STAT > 1)
    palloc_heap_stat_decrease(heap, malloc_bins[_palloc_bin(bsize)], 1);
    #endif
  }
  //else if (bsize <= PALLOC_LARGE_OBJ_SIZE_MAX) {
  //  palloc_heap_stat_decrease(heap, malloc_large, bsize);
  //}
  else {
    palloc_heap_stat_decrease(heap, malloc_huge, bsize);
  }
}
#else
static void palloc_stat_free(const palloc_page_t* page, const palloc_block_t* block) {
  PALLOC_UNUSED(page); PALLOC_UNUSED(block);
}
#endif


// Remove guard page when building with PALLOC_GUARDED
#if PALLOC_GUARDED
static void palloc_block_unguard(palloc_page_t* page, palloc_block_t* block, void* p) {
  PALLOC_UNUSED(p);
  palloc_assert_internal(palloc_block_ptr_is_guarded(block, p));
  palloc_assert_internal(palloc_page_has_aligned(page));
  palloc_assert_internal((uint8_t*)p - (uint8_t*)block >= (ptrdiff_t)sizeof(palloc_block_t));
  palloc_assert_internal(block->next == PALLOC_BLOCK_TAG_GUARDED);

  const size_t bsize = palloc_page_block_size(page);
  const size_t psize = _palloc_os_page_size();
  palloc_assert_internal(bsize > psize);
  palloc_assert_internal(_palloc_page_segment(page)->allow_decommit);
  void* gpage = (uint8_t*)block + bsize - psize;
  palloc_assert_internal(_palloc_is_aligned(gpage, psize));
  _palloc_os_unprotect(gpage, psize);
}
#endif
