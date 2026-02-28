/* ----------------------------------------------------------------------------
Copyright (c) 2019-2023, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* -----------------------------------------------------------
  The following functions are to reliably find the segment or
  block that encompasses any pointer p (or NULL if it is not
  in any of our segments).
  We maintain a bitmap of all memory with 1 bit per PALLOC_SEGMENT_SIZE (64MiB)
  set to 1 if it contains the segment meta data.
----------------------------------------------------------- */
#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"

// Reduce total address space to reduce .bss  (due to the `palloc_segment_map`)
#if (PALLOC_INTPTR_SIZE > 4) && PALLOC_TRACK_ASAN
#define PALLOC_SEGMENT_MAP_MAX_ADDRESS    (128*1024ULL*PALLOC_GiB)  // 128 TiB  (see issue #881)
#elif (PALLOC_INTPTR_SIZE > 4)
#define PALLOC_SEGMENT_MAP_MAX_ADDRESS    (48*1024ULL*PALLOC_GiB)   // 48 TiB
#else
#define PALLOC_SEGMENT_MAP_MAX_ADDRESS    (UINT32_MAX)
#endif

#define PALLOC_SEGMENT_MAP_PART_SIZE      (PALLOC_INTPTR_SIZE*PALLOC_KiB - 128)      // 128 > sizeof(palloc_memid_t) ! 
#define PALLOC_SEGMENT_MAP_PART_BITS      (8*PALLOC_SEGMENT_MAP_PART_SIZE)
#define PALLOC_SEGMENT_MAP_PART_ENTRIES   (PALLOC_SEGMENT_MAP_PART_SIZE / PALLOC_INTPTR_SIZE)
#define PALLOC_SEGMENT_MAP_PART_BIT_SPAN  (PALLOC_SEGMENT_ALIGN)                 // memory area covered by 1 bit

#if (PALLOC_SEGMENT_MAP_PART_BITS < (PALLOC_SEGMENT_MAP_MAX_ADDRESS / PALLOC_SEGMENT_MAP_PART_BIT_SPAN)) // prevent overflow on 32-bit (issue #1017)
#define PALLOC_SEGMENT_MAP_PART_SPAN      (PALLOC_SEGMENT_MAP_PART_BITS * PALLOC_SEGMENT_MAP_PART_BIT_SPAN)
#else
#define PALLOC_SEGMENT_MAP_PART_SPAN      PALLOC_SEGMENT_MAP_MAX_ADDRESS
#endif

#define PALLOC_SEGMENT_MAP_MAX_PARTS      ((PALLOC_SEGMENT_MAP_MAX_ADDRESS / PALLOC_SEGMENT_MAP_PART_SPAN) + 1)

// A part of the segment map.
typedef struct palloc_segmap_part_s {
  palloc_memid_t memid;
  _Atomic(uintptr_t) map[PALLOC_SEGMENT_MAP_PART_ENTRIES];
} palloc_segmap_part_t;

// Allocate parts on-demand to reduce .bss footprint
static _Atomic(palloc_segmap_part_t*) palloc_segment_map[PALLOC_SEGMENT_MAP_MAX_PARTS]; // = { NULL, .. }

static palloc_segmap_part_t* palloc_segment_map_index_of(const palloc_segment_t* segment, bool create_on_demand, size_t* idx, size_t* bitidx) {
  // note: segment can be invalid or NULL.
  palloc_assert_internal(_palloc_ptr_segment(segment + 1) == segment); // is it aligned on PALLOC_SEGMENT_SIZE?
  *idx = 0;
  *bitidx = 0;  
  if ((uintptr_t)segment >= PALLOC_SEGMENT_MAP_MAX_ADDRESS) return NULL;
  const uintptr_t segindex = ((uintptr_t)segment) / PALLOC_SEGMENT_MAP_PART_SPAN;
  if (segindex >= PALLOC_SEGMENT_MAP_MAX_PARTS) return NULL;
  palloc_segmap_part_t* part = palloc_atomic_load_ptr_relaxed(palloc_segmap_part_t, &palloc_segment_map[segindex]);

  // allocate on demand to reduce .bss footprint
  if palloc_unlikely(part == NULL) {
    if (!create_on_demand) return NULL;
    palloc_memid_t memid;
    part = (palloc_segmap_part_t*)_palloc_os_zalloc(sizeof(palloc_segmap_part_t), &memid);
    if (part == NULL) return NULL;
    part->memid = memid;
    palloc_segmap_part_t* expected = NULL;
    if (!palloc_atomic_cas_ptr_strong_release(palloc_segmap_part_t, &palloc_segment_map[segindex], &expected, part)) {
      _palloc_os_free(part, sizeof(palloc_segmap_part_t), memid);
      part = expected;
      if (part == NULL) return NULL;
    }
  }
  palloc_assert(part != NULL);
  const uintptr_t offset = ((uintptr_t)segment) % PALLOC_SEGMENT_MAP_PART_SPAN;
  const uintptr_t bitofs = offset / PALLOC_SEGMENT_MAP_PART_BIT_SPAN;
  *idx = bitofs / PALLOC_INTPTR_BITS;
  *bitidx = bitofs % PALLOC_INTPTR_BITS;
  return part;
}

void _palloc_segment_map_allocated_at(const palloc_segment_t* segment) {
  if (segment->memid.memkind == PALLOC_MEM_ARENA) return; // we lookup segments first in the arena's and don't need the segment map
  size_t index;
  size_t bitidx;
  palloc_segmap_part_t* part = palloc_segment_map_index_of(segment, true /* alloc map if needed */, &index, &bitidx);
  if (part == NULL) return; // outside our address range..
  uintptr_t mask = palloc_atomic_load_relaxed(&part->map[index]);
  uintptr_t newmask;
  do {
    newmask = (mask | ((uintptr_t)1 << bitidx));
  } while (!palloc_atomic_cas_weak_release(&part->map[index], &mask, newmask));
}

void _palloc_segment_map_freed_at(const palloc_segment_t* segment) {
  if (segment->memid.memkind == PALLOC_MEM_ARENA) return;
  size_t index;
  size_t bitidx;
  palloc_segmap_part_t* part = palloc_segment_map_index_of(segment, false /* don't alloc if not present */, &index, &bitidx);
  if (part == NULL) return; // outside our address range..
  uintptr_t mask = palloc_atomic_load_relaxed(&part->map[index]);
  uintptr_t newmask;
  do {
    newmask = (mask & ~((uintptr_t)1 << bitidx));
  } while (!palloc_atomic_cas_weak_release(&part->map[index], &mask, newmask));
}

// Determine the segment belonging to a pointer or NULL if it is not in a valid segment.
static palloc_segment_t* _palloc_segment_of(const void* p) {
  if (p == NULL) return NULL;
  palloc_segment_t* segment = _palloc_ptr_segment(p);  // segment can be NULL  
  size_t index;
  size_t bitidx;
  palloc_segmap_part_t* part = palloc_segment_map_index_of(segment, false /* dont alloc if not present */, &index, &bitidx);
  if (part == NULL) return NULL;  
  const uintptr_t mask = palloc_atomic_load_relaxed(&part->map[index]);
  if palloc_likely((mask & ((uintptr_t)1 << bitidx)) != 0) {
    bool cookie_ok = (_palloc_ptr_cookie(segment) == segment->cookie);
    palloc_assert_internal(cookie_ok); PALLOC_UNUSED(cookie_ok);
    return segment; // yes, allocated by us
  }
  return NULL;
}

// Is this a valid pointer in our heap?
static bool palloc_is_valid_pointer(const void* p) {
  // first check if it is in an arena, then check if it is OS allocated
  return (_palloc_arena_contains(p) || _palloc_segment_of(p) != NULL);
}

palloc_decl_nodiscard palloc_decl_export bool palloc_is_in_heap_region(const void* p) palloc_attr_noexcept {
  return palloc_is_valid_pointer(p);
}

void _palloc_segment_map_unsafe_destroy(void) {
  for (size_t i = 0; i < PALLOC_SEGMENT_MAP_MAX_PARTS; i++) {
    palloc_segmap_part_t* part = palloc_atomic_exchange_ptr_relaxed(palloc_segmap_part_t, &palloc_segment_map[i], NULL);
    if (part != NULL) {
      _palloc_os_free(part, sizeof(palloc_segmap_part_t), part->memid);
    }
  }
}
