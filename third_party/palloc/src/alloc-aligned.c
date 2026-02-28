/* ----------------------------------------------------------------------------
Copyright (c) 2018-2021, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/prim.h"  // palloc_prim_get_default_heap

#include <string.h>     // memset

// ------------------------------------------------------
// Aligned Allocation
// ------------------------------------------------------

static bool palloc_malloc_is_naturally_aligned( size_t size, size_t alignment ) {
  // objects up to `PALLOC_MAX_ALIGN_GUARANTEE` are allocated aligned to their size (see `segment.c:_palloc_segment_page_start`).
  palloc_assert_internal(_palloc_is_power_of_two(alignment) && (alignment > 0));
  if (alignment > size) return false;
  if (alignment <= PALLOC_MAX_ALIGN_SIZE) return true;
  const size_t bsize = palloc_good_size(size);
  return (bsize <= PALLOC_MAX_ALIGN_GUARANTEE && (bsize & (alignment-1)) == 0);
}

#if PALLOC_GUARDED
static palloc_decl_restrict void* palloc_heap_malloc_guarded_aligned(palloc_heap_t* heap, size_t size, size_t alignment, bool zero) palloc_attr_noexcept {
  // use over allocation for guarded blocksl
  palloc_assert_internal(alignment > 0 && alignment < PALLOC_BLOCK_ALIGNMENT_MAX);
  const size_t oversize = size + alignment - 1;
  void* base = _palloc_heap_malloc_guarded(heap, oversize, zero);
  void* p = palloc_align_up_ptr(base, alignment);
  palloc_track_align(base, p, (uint8_t*)p - (uint8_t*)base, size);
  palloc_assert_internal(palloc_usable_size(p) >= size);
  palloc_assert_internal(_palloc_is_aligned(p, alignment));
  return p;
}

static void* palloc_heap_malloc_zero_no_guarded(palloc_heap_t* heap, size_t size, bool zero, size_t* usable) {
  const size_t rate = heap->guarded_sample_rate;
  // only write if `rate!=0` so we don't write to the constant `_palloc_heap_empty`
  if (rate != 0) { heap->guarded_sample_rate = 0; }
  void* p = _palloc_heap_malloc_zero_ex(heap, size, zero, 0, usable);
  if (rate != 0) { heap->guarded_sample_rate = rate; }
  return p;
}
#else
static void* palloc_heap_malloc_zero_no_guarded(palloc_heap_t* heap, size_t size, bool zero, size_t* usable) {
  return _palloc_heap_malloc_zero_ex(heap, size, zero, 0, usable);
}
#endif

// Fallback aligned allocation that over-allocates -- split out for better codegen
static palloc_decl_noinline void* palloc_heap_malloc_zero_aligned_at_overalloc(palloc_heap_t* const heap, const size_t size, const size_t alignment, const size_t offset, const bool zero, size_t* usable) palloc_attr_noexcept
{
  palloc_assert_internal(size <= (PALLOC_MAX_ALLOC_SIZE - PALLOC_PADDING_SIZE));
  palloc_assert_internal(alignment != 0 && _palloc_is_power_of_two(alignment));

  void* p;
  size_t oversize;
  if palloc_unlikely(alignment > PALLOC_BLOCK_ALIGNMENT_MAX) {
    // use OS allocation for very large alignment and allocate inside a huge page (dedicated segment with 1 page)
    // This can support alignments >= PALLOC_SEGMENT_SIZE by ensuring the object can be aligned at a point in the
    // first (and single) page such that the segment info is `PALLOC_SEGMENT_SIZE` bytes before it (so it can be found by aligning the pointer down)
    if palloc_unlikely(offset != 0) {
      // todo: cannot support offset alignment for very large alignments yet
#if PALLOC_DEBUG > 0
      _palloc_error_message(EOVERFLOW, "aligned allocation with a very large alignment cannot be used with an alignment offset (size %zu, alignment %zu, offset %zu)\n", size, alignment, offset);
#endif
      return NULL;
    }
    oversize = (size <= PALLOC_SMALL_SIZE_MAX ? PALLOC_SMALL_SIZE_MAX + 1 /* ensure we use generic malloc path */ : size);
    // note: no guarded as alignment > 0
    p = _palloc_heap_malloc_zero_ex(heap, oversize, false, alignment, usable); // the page block size should be large enough to align in the single huge page block
    // zero afterwards as only the area from the aligned_p may be committed!
    if (p == NULL) return NULL;
  }
  else {
    // otherwise over-allocate
    oversize = (size < PALLOC_MAX_ALIGN_SIZE ? PALLOC_MAX_ALIGN_SIZE : size) + alignment - 1;  // adjust for size <= 16; with size 0 and aligment 64k, we would allocate a 64k block and pointing just beyond that.
    p = palloc_heap_malloc_zero_no_guarded(heap, oversize, zero, usable);
    if (p == NULL) return NULL;
  }
  palloc_page_t* page = _palloc_ptr_page(p);

  // .. and align within the allocation
  const uintptr_t align_mask = alignment - 1;  // for any x, `(x & align_mask) == (x % alignment)`
  const uintptr_t poffset = ((uintptr_t)p + offset) & align_mask;
  const uintptr_t adjust  = (poffset == 0 ? 0 : alignment - poffset);
  palloc_assert_internal(adjust < alignment);
  void* aligned_p = (void*)((uintptr_t)p + adjust);
  if (aligned_p != p) {
    palloc_page_set_has_aligned(page, true);
    #if PALLOC_GUARDED
    // set tag to aligned so palloc_usable_size works with guard pages
    if (adjust >= sizeof(palloc_block_t)) {
      palloc_block_t* const block = (palloc_block_t*)p;
      block->next = PALLOC_BLOCK_TAG_ALIGNED;
    }
    #endif
    _palloc_padding_shrink(page, (palloc_block_t*)p, adjust + size);
  }
  // todo: expand padding if overallocated ?

  palloc_assert_internal(palloc_page_usable_block_size(page) >= adjust + size);
  palloc_assert_internal(((uintptr_t)aligned_p + offset) % alignment == 0);
  palloc_assert_internal(palloc_usable_size(aligned_p)>=size);
  palloc_assert_internal(palloc_usable_size(p) == palloc_usable_size(aligned_p)+adjust);
  #if PALLOC_DEBUG > 1
  palloc_page_t* const apage = _palloc_ptr_page(aligned_p);
  void* unalign_p = _palloc_page_ptr_unalign(apage, aligned_p);
  palloc_assert_internal(p == unalign_p);
  #endif

  // now zero the block if needed
  if (alignment > PALLOC_BLOCK_ALIGNMENT_MAX) {
    // for the tracker, on huge aligned allocations only the memory from the start of the large block is defined
    palloc_track_mem_undefined(aligned_p, size);
    if (zero) {
      _palloc_memzero_aligned(aligned_p, palloc_usable_size(aligned_p));
    }
  }

  if (p != aligned_p) {
    palloc_track_align(p,aligned_p,adjust,palloc_usable_size(aligned_p));
    #if PALLOC_GUARDED
    palloc_track_mem_defined(p, sizeof(palloc_block_t));
    #endif
  }
  return aligned_p;
}

// Generic primitive aligned allocation -- split out for better codegen
static palloc_decl_noinline void* palloc_heap_malloc_zero_aligned_at_generic(palloc_heap_t* const heap, const size_t size, const size_t alignment, const size_t offset, const bool zero, size_t* usable) palloc_attr_noexcept
{
  palloc_assert_internal(alignment != 0 && _palloc_is_power_of_two(alignment));
  // we don't allocate more than PALLOC_MAX_ALLOC_SIZE (see <https://sourceware.org/ml/libc-announce/2019/msg00001.html>)
  if palloc_unlikely(size > (PALLOC_MAX_ALLOC_SIZE - PALLOC_PADDING_SIZE)) {
    #if PALLOC_DEBUG > 0
    _palloc_error_message(EOVERFLOW, "aligned allocation request is too large (size %zu, alignment %zu)\n", size, alignment);
    #endif
    return NULL;
  }

  // use regular allocation if it is guaranteed to fit the alignment constraints.
  // this is important to try as the fast path in `palloc_heap_malloc_zero_aligned` only works when there exist
  // a page with the right block size, and if we always use the over-alloc fallback that would never happen.
  if (offset == 0 && palloc_malloc_is_naturally_aligned(size,alignment)) {
    void* p = palloc_heap_malloc_zero_no_guarded(heap, size, zero, usable);
    palloc_assert_internal(p == NULL || ((uintptr_t)p % alignment) == 0);
    const bool is_aligned_or_null = (((uintptr_t)p) & (alignment-1))==0;
    if palloc_likely(is_aligned_or_null) {
      return p;
    }
    else {
      // this should never happen if the `palloc_malloc_is_naturally_aligned` check is correct..
      palloc_assert(false);
      palloc_free(p);
    }
  }

  // fall back to over-allocation
  return palloc_heap_malloc_zero_aligned_at_overalloc(heap,size,alignment,offset,zero,usable);
}


// Primitive aligned allocation
static void* palloc_heap_malloc_zero_aligned_at(palloc_heap_t* const heap, const size_t size, 
                                            const size_t alignment, const size_t offset, const bool zero,
                                            size_t* usable) palloc_attr_noexcept
{
  // note: we don't require `size > offset`, we just guarantee that the address at offset is aligned regardless of the allocated size.
  if palloc_unlikely(alignment == 0 || !_palloc_is_power_of_two(alignment)) { // require power-of-two (see <https://en.cppreference.com/w/c/memory/aligned_alloc>)
    #if PALLOC_DEBUG > 0
    _palloc_error_message(EOVERFLOW, "aligned allocation requires the alignment to be a power-of-two (size %zu, alignment %zu)\n", size, alignment);
    #endif
    return NULL;
  }

  #if PALLOC_GUARDED
  if (offset==0 && alignment < PALLOC_BLOCK_ALIGNMENT_MAX && palloc_heap_malloc_use_guarded(heap,size)) {
    return palloc_heap_malloc_guarded_aligned(heap, size, alignment, zero);
  }
  #endif

  // try first if there happens to be a small block available with just the right alignment
  if palloc_likely(size <= PALLOC_SMALL_SIZE_MAX && alignment <= size) {
    const uintptr_t align_mask = alignment-1;       // for any x, `(x & align_mask) == (x % alignment)`
    const size_t padsize = size + PALLOC_PADDING_SIZE;
    palloc_page_t* page = _palloc_heap_get_free_small_page(heap, padsize);
    if palloc_likely(page->free != NULL) {
      const bool is_aligned = (((uintptr_t)page->free + offset) & align_mask)==0;
      if palloc_likely(is_aligned)
      {
        if (usable!=NULL) { *usable = palloc_page_usable_block_size(page); }
        void* p = (zero ? _palloc_page_malloc_zeroed(heap,page,padsize) : _palloc_page_malloc(heap,page,padsize)); // call specific page malloc for better codegen
        palloc_assert_internal(p != NULL);
        palloc_assert_internal(((uintptr_t)p + offset) % alignment == 0);
        palloc_track_malloc(p,size,zero);
        return p;
      }
    }
  }

  // fallback to generic aligned allocation
  return palloc_heap_malloc_zero_aligned_at_generic(heap, size, alignment, offset, zero, usable);
}


// ------------------------------------------------------
// Optimized palloc_heap_malloc_aligned / palloc_malloc_aligned
// ------------------------------------------------------

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_malloc_aligned_at(palloc_heap_t* heap, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_malloc_zero_aligned_at(heap, size, alignment, offset, false, NULL);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_malloc_aligned(palloc_heap_t* heap, size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_malloc_aligned_at(heap, size, alignment, 0);
}

// ensure a definition is emitted
#if defined(__cplusplus)
void* _palloc_extern_heap_malloc_aligned = (void*)&palloc_heap_malloc_aligned;
#endif

// ------------------------------------------------------
// Aligned Allocation
// ------------------------------------------------------

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_zalloc_aligned_at(palloc_heap_t* heap, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_malloc_zero_aligned_at(heap, size, alignment, offset, true, NULL);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_zalloc_aligned(palloc_heap_t* heap, size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_zalloc_aligned_at(heap, size, alignment, 0);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_calloc_aligned_at(palloc_heap_t* heap, size_t count, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(count, size, &total)) return NULL;
  return palloc_heap_zalloc_aligned_at(heap, total, alignment, offset);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_calloc_aligned(palloc_heap_t* heap, size_t count, size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_calloc_aligned_at(heap,count,size,alignment,0);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_malloc_aligned_at(size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_malloc_aligned_at(palloc_prim_get_default_heap(), size, alignment, offset);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_malloc_aligned(size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_malloc_aligned(palloc_prim_get_default_heap(), size, alignment);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_umalloc_aligned(size_t size, size_t alignment, size_t* block_size) palloc_attr_noexcept {
  return palloc_heap_malloc_zero_aligned_at(palloc_prim_get_default_heap(), size, alignment, 0, false, block_size);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_zalloc_aligned_at(size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_zalloc_aligned_at(palloc_prim_get_default_heap(), size, alignment, offset);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_zalloc_aligned(size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_zalloc_aligned(palloc_prim_get_default_heap(), size, alignment);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_uzalloc_aligned(size_t size, size_t alignment, size_t* block_size) palloc_attr_noexcept {
  return palloc_heap_malloc_zero_aligned_at(palloc_prim_get_default_heap(), size, alignment, 0, true, block_size);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_calloc_aligned_at(size_t count, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_calloc_aligned_at(palloc_prim_get_default_heap(), count, size, alignment, offset);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_calloc_aligned(size_t count, size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_calloc_aligned(palloc_prim_get_default_heap(), count, size, alignment);
}


// ------------------------------------------------------
// Aligned re-allocation
// ------------------------------------------------------

static void* palloc_heap_realloc_zero_aligned_at(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment, size_t offset, bool zero) palloc_attr_noexcept {
  palloc_assert(alignment > 0);
  if (alignment <= sizeof(uintptr_t)) return _palloc_heap_realloc_zero(heap,p,newsize,zero,NULL,NULL);
  if (p == NULL) return palloc_heap_malloc_zero_aligned_at(heap,newsize,alignment,offset,zero,NULL);
  size_t size = palloc_usable_size(p);
  if (newsize <= size && newsize >= (size - (size / 2))
      && (((uintptr_t)p + offset) % alignment) == 0) {
    return p;  // reallocation still fits, is aligned and not more than 50% waste
  }
  else {
    // note: we don't zero allocate upfront so we only zero initialize the expanded part
    void* newp = palloc_heap_malloc_aligned_at(heap,newsize,alignment,offset);
    if (newp != NULL) {
      if (zero && newsize > size) {
        // also set last word in the previous allocation to zero to ensure any padding is zero-initialized
        size_t start = (size >= sizeof(intptr_t) ? size - sizeof(intptr_t) : 0);
        _palloc_memzero((uint8_t*)newp + start, newsize - start);
      }
      _palloc_memcpy_aligned(newp, p, (newsize > size ? size : newsize));
      palloc_free(p); // only free if successful
    }
    return newp;
  }
}

static void* palloc_heap_realloc_zero_aligned(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment, bool zero) palloc_attr_noexcept {
  palloc_assert(alignment > 0);
  if (alignment <= sizeof(uintptr_t)) return _palloc_heap_realloc_zero(heap,p,newsize,zero,NULL,NULL);
  size_t offset = ((uintptr_t)p % alignment); // use offset of previous allocation (p can be NULL)
  return palloc_heap_realloc_zero_aligned_at(heap,p,newsize,alignment,offset,zero);
}

palloc_decl_nodiscard void* palloc_heap_realloc_aligned_at(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_realloc_zero_aligned_at(heap,p,newsize,alignment,offset,false);
}

palloc_decl_nodiscard void* palloc_heap_realloc_aligned(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_realloc_zero_aligned(heap,p,newsize,alignment,false);
}

palloc_decl_nodiscard void* palloc_heap_rezalloc_aligned_at(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_realloc_zero_aligned_at(heap, p, newsize, alignment, offset, true);
}

palloc_decl_nodiscard void* palloc_heap_rezalloc_aligned(palloc_heap_t* heap, void* p, size_t newsize, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_realloc_zero_aligned(heap, p, newsize, alignment, true);
}

palloc_decl_nodiscard void* palloc_heap_recalloc_aligned_at(palloc_heap_t* heap, void* p, size_t newcount, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(newcount, size, &total)) return NULL;
  return palloc_heap_rezalloc_aligned_at(heap, p, total, alignment, offset);
}

palloc_decl_nodiscard void* palloc_heap_recalloc_aligned(palloc_heap_t* heap, void* p, size_t newcount, size_t size, size_t alignment) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(newcount, size, &total)) return NULL;
  return palloc_heap_rezalloc_aligned(heap, p, total, alignment);
}

palloc_decl_nodiscard void* palloc_realloc_aligned_at(void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_realloc_aligned_at(palloc_prim_get_default_heap(), p, newsize, alignment, offset);
}

palloc_decl_nodiscard void* palloc_realloc_aligned(void* p, size_t newsize, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_realloc_aligned(palloc_prim_get_default_heap(), p, newsize, alignment);
}

palloc_decl_nodiscard void* palloc_rezalloc_aligned_at(void* p, size_t newsize, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_rezalloc_aligned_at(palloc_prim_get_default_heap(), p, newsize, alignment, offset);
}

palloc_decl_nodiscard void* palloc_rezalloc_aligned(void* p, size_t newsize, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_rezalloc_aligned(palloc_prim_get_default_heap(), p, newsize, alignment);
}

palloc_decl_nodiscard void* palloc_recalloc_aligned_at(void* p, size_t newcount, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept {
  return palloc_heap_recalloc_aligned_at(palloc_prim_get_default_heap(), p, newcount, size, alignment, offset);
}

palloc_decl_nodiscard void* palloc_recalloc_aligned(void* p, size_t newcount, size_t size, size_t alignment) palloc_attr_noexcept {
  return palloc_heap_recalloc_aligned(palloc_prim_get_default_heap(), p, newcount, size, alignment);
}


