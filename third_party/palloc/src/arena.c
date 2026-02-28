/* ----------------------------------------------------------------------------
Copyright (c) 2019-2024, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------------
"Arenas" are fixed area's of OS memory from which we can allocate
large blocks (>= PALLOC_ARENA_MIN_BLOCK_SIZE, 4MiB).
In contrast to the rest of palloc, the arenas are shared between
threads and need to be accessed using atomic operations.

Arenas are also used to for huge OS page (1GiB) reservations or for reserving
OS memory upfront which can be improve performance or is sometimes needed
on embedded devices. We can also employ this with WASI or `sbrk` systems
to reserve large arenas upfront and be able to reuse the memory more effectively.

The arena allocation needs to be thread safe and we use an atomic bitmap to allocate.
-----------------------------------------------------------------------------*/

#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"
#include "bitmap.h"


/* -----------------------------------------------------------
  Arena allocation
----------------------------------------------------------- */

// A memory arena descriptor
typedef struct palloc_arena_s {
  palloc_arena_id_t       id;                   // arena id; 0 for non-specific
  palloc_memid_t          memid;                // memid of the memory area
  _Atomic(uint8_t*)   start;                // the start of the memory area
  size_t              block_count;          // size of the area in arena blocks (of `PALLOC_ARENA_BLOCK_SIZE`)
  size_t              field_count;          // number of bitmap fields (where `field_count * PALLOC_BITMAP_FIELD_BITS >= block_count`)
  size_t              meta_size;            // size of the arena structure itself (including its bitmaps)
  palloc_memid_t          meta_memid;           // memid of the arena structure itself (OS or static allocation)
  int                 numa_node;            // associated NUMA node
  bool                exclusive;            // only allow allocations if specifically for this arena
  bool                is_large;             // memory area consists of large- or huge OS pages (always committed)
  palloc_lock_t           abandoned_visit_lock; // lock is only used when abandoned segments are being visited
  _Atomic(size_t)     search_idx;           // optimization to start the search for free blocks
  _Atomic(palloc_msecs_t) purge_expire;         // expiration time when blocks should be purged from `blocks_purge`.

  palloc_bitmap_field_t*  blocks_dirty;         // are the blocks potentially non-zero?
  palloc_bitmap_field_t*  blocks_committed;     // are the blocks committed? (can be NULL for memory that cannot be decommitted)
  palloc_bitmap_field_t*  blocks_purge;         // blocks that can be (reset) decommitted. (can be NULL for memory that cannot be (reset) decommitted)
  palloc_bitmap_field_t*  blocks_abandoned;     // blocks that start with an abandoned segment. (This crosses API's but it is convenient to have here)
  palloc_bitmap_field_t   blocks_inuse[1];      // in-place bitmap of in-use blocks (of size `field_count`)
  // do not add further fields here as the dirty, committed, purged, and abandoned bitmaps follow the inuse bitmap fields.
} palloc_arena_t;


#define PALLOC_ARENA_BLOCK_SIZE   (PALLOC_SEGMENT_SIZE)        // 64MiB  (must be at least PALLOC_SEGMENT_ALIGN)
#define PALLOC_ARENA_MIN_OBJ_SIZE (PALLOC_ARENA_BLOCK_SIZE/2)  // 32MiB
#define PALLOC_MAX_ARENAS         (132)                    // Limited as the reservation exponentially increases (and takes up .bss)

// The available arenas
static palloc_decl_cache_align _Atomic(palloc_arena_t*) palloc_arenas[PALLOC_MAX_ARENAS];
static palloc_decl_cache_align _Atomic(size_t)      palloc_arena_count; // = 0
static palloc_decl_cache_align _Atomic(int64_t)     palloc_arenas_purge_expire; // set if there exist purgeable arenas

#define PALLOC_IN_ARENA_C
#include "arena-abandon.c"
#undef PALLOC_IN_ARENA_C

/* -----------------------------------------------------------
  Arena id's
  id = arena_index + 1
----------------------------------------------------------- */

size_t palloc_arena_id_index(palloc_arena_id_t id) {
  return (size_t)(id <= 0 ? PALLOC_MAX_ARENAS : id - 1);
}

static palloc_arena_id_t palloc_arena_id_create(size_t arena_index) {
  palloc_assert_internal(arena_index < PALLOC_MAX_ARENAS);
  return (int)arena_index + 1;
}

palloc_arena_id_t _palloc_arena_id_none(void) {
  return 0;
}

static bool palloc_arena_id_is_suitable(palloc_arena_id_t arena_id, bool arena_is_exclusive, palloc_arena_id_t req_arena_id) {
  return ((!arena_is_exclusive && req_arena_id == _palloc_arena_id_none()) ||
          (arena_id == req_arena_id));
}

bool _palloc_arena_memid_is_suitable(palloc_memid_t memid, palloc_arena_id_t request_arena_id) {
  if (memid.memkind == PALLOC_MEM_ARENA) {
    return palloc_arena_id_is_suitable(memid.mem.arena.id, memid.mem.arena.is_exclusive, request_arena_id);
  }
  else {
    return palloc_arena_id_is_suitable(_palloc_arena_id_none(), false, request_arena_id);
  }
}

bool _palloc_arena_memid_is_os_allocated(palloc_memid_t memid) {
  return (memid.memkind == PALLOC_MEM_OS);
}

size_t palloc_arena_get_count(void) {
  return palloc_atomic_load_relaxed(&palloc_arena_count);
}

palloc_arena_t* palloc_arena_from_index(size_t idx) {
  palloc_assert_internal(idx < palloc_arena_get_count());
  return palloc_atomic_load_ptr_acquire(palloc_arena_t, &palloc_arenas[idx]);
}


/* -----------------------------------------------------------
  Arena allocations get a (currently) 16-bit memory id where the
  lower 8 bits are the arena id, and the upper bits the block index.
----------------------------------------------------------- */

static size_t palloc_block_count_of_size(size_t size) {
  return _palloc_divide_up(size, PALLOC_ARENA_BLOCK_SIZE);
}

static size_t palloc_arena_block_size(size_t bcount) {
  return (bcount * PALLOC_ARENA_BLOCK_SIZE);
}

static size_t palloc_arena_size(palloc_arena_t* arena) {
  return palloc_arena_block_size(arena->block_count);
}

static palloc_memid_t palloc_memid_create_arena(palloc_arena_id_t id, bool is_exclusive, palloc_bitmap_index_t bitmap_index) {
  palloc_memid_t memid = _palloc_memid_create(PALLOC_MEM_ARENA);
  memid.mem.arena.id = id;
  memid.mem.arena.block_index = bitmap_index;
  memid.mem.arena.is_exclusive = is_exclusive;
  return memid;
}

bool palloc_arena_memid_indices(palloc_memid_t memid, size_t* arena_index, palloc_bitmap_index_t* bitmap_index) {
  palloc_assert_internal(memid.memkind == PALLOC_MEM_ARENA);
  *arena_index = palloc_arena_id_index(memid.mem.arena.id);
  *bitmap_index = memid.mem.arena.block_index;
  return memid.mem.arena.is_exclusive;
}



/* -----------------------------------------------------------
  Special static area for palloc internal structures
  to avoid OS calls (for example, for the arena metadata (~= 256b))
----------------------------------------------------------- */

#define PALLOC_ARENA_STATIC_MAX  ((PALLOC_INTPTR_SIZE/2)*PALLOC_KiB)  // 4 KiB on 64-bit

static palloc_decl_cache_align uint8_t palloc_arena_static[PALLOC_ARENA_STATIC_MAX];  // must be cache aligned, see issue #895
static palloc_decl_cache_align _Atomic(size_t) palloc_arena_static_top;

static void* palloc_arena_static_zalloc(size_t size, size_t alignment, palloc_memid_t* memid) {
  *memid = _palloc_memid_none();
  if (size == 0 || size > PALLOC_ARENA_STATIC_MAX) return NULL;
  const size_t toplow = palloc_atomic_load_relaxed(&palloc_arena_static_top);
  if ((toplow + size) > PALLOC_ARENA_STATIC_MAX) return NULL;

  // try to claim space
  if (alignment < PALLOC_MAX_ALIGN_SIZE) { alignment = PALLOC_MAX_ALIGN_SIZE; }
  const size_t oversize = size + alignment - 1;
  if (toplow + oversize > PALLOC_ARENA_STATIC_MAX) return NULL;
  const size_t oldtop = palloc_atomic_add_acq_rel(&palloc_arena_static_top, oversize);
  size_t top = oldtop + oversize;
  if (top > PALLOC_ARENA_STATIC_MAX) {
    // try to roll back, ok if this fails
    palloc_atomic_cas_strong_acq_rel(&palloc_arena_static_top, &top, oldtop);
    return NULL;
  }

  // success
  *memid = _palloc_memid_create(PALLOC_MEM_STATIC);
  memid->initially_zero = true;
  const size_t start = _palloc_align_up(oldtop, alignment);
  uint8_t* const p = &palloc_arena_static[start];
  _palloc_memzero_aligned(p, size);
  return p;
}

void* _palloc_arena_meta_zalloc(size_t size, palloc_memid_t* memid) {
  *memid = _palloc_memid_none();

  // try static
  void* p = palloc_arena_static_zalloc(size, PALLOC_MAX_ALIGN_SIZE, memid);
  if (p != NULL) return p;

  // or fall back to the OS
  p = _palloc_os_zalloc(size, memid);
  if (p == NULL) return NULL;

  return p;
}

void _palloc_arena_meta_free(void* p, palloc_memid_t memid, size_t size) {
  if (palloc_memkind_is_os(memid.memkind)) {
    _palloc_os_free(p, size, memid);
  }
  else {
    palloc_assert(memid.memkind == PALLOC_MEM_STATIC);
  }
}

void* palloc_arena_block_start(palloc_arena_t* arena, palloc_bitmap_index_t bindex) {
  return (arena->start + palloc_arena_block_size(palloc_bitmap_index_bit(bindex)));
}


/* -----------------------------------------------------------
  Thread safe allocation in an arena
----------------------------------------------------------- */

// claim the `blocks_inuse` bits
static bool palloc_arena_try_claim(palloc_arena_t* arena, size_t blocks, palloc_bitmap_index_t* bitmap_idx)
{
  size_t idx = 0; // palloc_atomic_load_relaxed(&arena->search_idx);  // start from last search; ok to be relaxed as the exact start does not matter
  if (_palloc_bitmap_try_find_from_claim_across(arena->blocks_inuse, arena->field_count, idx, blocks, bitmap_idx)) {
    palloc_atomic_store_relaxed(&arena->search_idx, palloc_bitmap_index_field(*bitmap_idx));  // start search from found location next time around
    return true;
  };
  return false;
}


/* -----------------------------------------------------------
  Arena Allocation
----------------------------------------------------------- */

static palloc_decl_noinline void* palloc_arena_try_alloc_at(palloc_arena_t* arena, size_t arena_index, size_t needed_bcount,
                                                    bool commit, palloc_memid_t* memid)
{
  PALLOC_UNUSED(arena_index);
  palloc_assert_internal(palloc_arena_id_index(arena->id) == arena_index);

  palloc_bitmap_index_t bitmap_index;
  if (!palloc_arena_try_claim(arena, needed_bcount, &bitmap_index)) return NULL;

  // claimed it!
  void* p = palloc_arena_block_start(arena, bitmap_index);
  *memid = palloc_memid_create_arena(arena->id, arena->exclusive, bitmap_index);
  memid->is_pinned = arena->memid.is_pinned;

  // none of the claimed blocks should be scheduled for a decommit
  if (arena->blocks_purge != NULL) {
    // this is thread safe as a potential purge only decommits parts that are not yet claimed as used (in `blocks_inuse`).
    _palloc_bitmap_unclaim_across(arena->blocks_purge, arena->field_count, needed_bcount, bitmap_index);
  }

  // set the dirty bits (todo: no need for an atomic op here?)
  if (arena->memid.initially_zero && arena->blocks_dirty != NULL) {
    memid->initially_zero = _palloc_bitmap_claim_across(arena->blocks_dirty, arena->field_count, needed_bcount, bitmap_index, NULL, NULL);
  }

  // set commit state
  if (arena->blocks_committed == NULL) {
    // always committed
    memid->initially_committed = true;
  }
  else if (commit) {
    // commit requested, but the range may not be committed as a whole: ensure it is committed now
    memid->initially_committed = true;
    const size_t commit_size = palloc_arena_block_size(needed_bcount);      
    bool any_uncommitted;
    size_t already_committed = 0;
    _palloc_bitmap_claim_across(arena->blocks_committed, arena->field_count, needed_bcount, bitmap_index, &any_uncommitted, &already_committed);
    if (any_uncommitted) {
      palloc_assert_internal(already_committed < needed_bcount);
      const size_t stat_commit_size = commit_size - palloc_arena_block_size(already_committed);
      bool commit_zero = false;
      if (!_palloc_os_commit_ex(p, commit_size, &commit_zero, stat_commit_size)) {
        memid->initially_committed = false;
      }
      else {
        if (commit_zero) { memid->initially_zero = true; }
      }
    }
    else {
      // all are already committed: signal that we are reusing memory in case it was purged before
      _palloc_os_reuse( p, commit_size );
    }
  }
  else {
    // no need to commit, but check if already fully committed
    size_t already_committed = 0;
    memid->initially_committed = _palloc_bitmap_is_claimed_across(arena->blocks_committed, arena->field_count, needed_bcount, bitmap_index, &already_committed);
    if (!memid->initially_committed && already_committed > 0) {
      // partially committed: as it will be committed at some time, adjust the stats and pretend the range is fully uncommitted.
      palloc_assert_internal(already_committed < needed_bcount);
      _palloc_stat_decrease(&_palloc_stats_main.committed, palloc_arena_block_size(already_committed));
      _palloc_bitmap_unclaim_across(arena->blocks_committed, arena->field_count, needed_bcount, bitmap_index);
    }
  }

  return p;
}

// allocate in a specific arena
static void* palloc_arena_try_alloc_at_id(palloc_arena_id_t arena_id, bool match_numa_node, int numa_node, size_t size, size_t alignment,
                                       bool commit, bool allow_large, palloc_arena_id_t req_arena_id, palloc_memid_t* memid )
{
  PALLOC_UNUSED_RELEASE(alignment);
  palloc_assert(alignment <= PALLOC_SEGMENT_ALIGN);
  const size_t bcount = palloc_block_count_of_size(size);
  const size_t arena_index = palloc_arena_id_index(arena_id);
  palloc_assert_internal(arena_index < palloc_atomic_load_relaxed(&palloc_arena_count));
  palloc_assert_internal(size <= palloc_arena_block_size(bcount));

  // Check arena suitability
  palloc_arena_t* arena = palloc_arena_from_index(arena_index);
  if (arena == NULL) return NULL;
  if (!allow_large && arena->is_large) return NULL;
  if (!palloc_arena_id_is_suitable(arena->id, arena->exclusive, req_arena_id)) return NULL;
  if (req_arena_id == _palloc_arena_id_none()) { // in not specific, check numa affinity
    const bool numa_suitable = (numa_node < 0 || arena->numa_node < 0 || arena->numa_node == numa_node);
    if (match_numa_node) { if (!numa_suitable) return NULL; }
                    else { if (numa_suitable) return NULL; }
  }

  // try to allocate
  void* p = palloc_arena_try_alloc_at(arena, arena_index, bcount, commit, memid);
  palloc_assert_internal(p == NULL || _palloc_is_aligned(p, alignment));
  return p;
}


// allocate from an arena with fallback to the OS
static palloc_decl_noinline void* palloc_arena_try_alloc(int numa_node, size_t size, size_t alignment,
                                                  bool commit, bool allow_large,
                                                  palloc_arena_id_t req_arena_id, palloc_memid_t* memid )
{
  PALLOC_UNUSED(alignment);
  palloc_assert_internal(alignment <= PALLOC_SEGMENT_ALIGN);
  const size_t max_arena = palloc_atomic_load_relaxed(&palloc_arena_count);
  if palloc_likely(max_arena == 0) return NULL;

  if (req_arena_id != _palloc_arena_id_none()) {
    // try a specific arena if requested
    if (palloc_arena_id_index(req_arena_id) < max_arena) {
      void* p = palloc_arena_try_alloc_at_id(req_arena_id, true, numa_node, size, alignment, commit, allow_large, req_arena_id, memid);
      if (p != NULL) return p;
    }
  }
  else {
    // try numa affine allocation
    for (size_t i = 0; i < max_arena; i++) {
      void* p = palloc_arena_try_alloc_at_id(palloc_arena_id_create(i), true, numa_node, size, alignment, commit, allow_large, req_arena_id, memid);
      if (p != NULL) return p;
    }

    // try from another numa node instead..
    if (numa_node >= 0) {  // if numa_node was < 0 (no specific affinity requested), all arena's have been tried already
      for (size_t i = 0; i < max_arena; i++) {
        void* p = palloc_arena_try_alloc_at_id(palloc_arena_id_create(i), false /* only proceed if not numa local */, numa_node, size, alignment, commit, allow_large, req_arena_id, memid);
        if (p != NULL) return p;
      }
    }
  }
  return NULL;
}

// try to reserve a fresh arena space
static bool palloc_arena_reserve(size_t req_size, bool allow_large, palloc_arena_id_t *arena_id)
{
  if (_palloc_preloading()) return false;  // use OS only while pre loading

  const size_t arena_count = palloc_atomic_load_acquire(&palloc_arena_count);
  if (arena_count > (PALLOC_MAX_ARENAS - 4)) return false;

  size_t arena_reserve = palloc_option_get_size(palloc_option_arena_reserve);
  if (arena_reserve == 0) return false;

  if (!_palloc_os_has_virtual_reserve()) {
    arena_reserve = arena_reserve/4;  // be conservative if virtual reserve is not supported (for WASM for example)
  }
  arena_reserve = _palloc_align_up(arena_reserve, PALLOC_ARENA_BLOCK_SIZE);
  arena_reserve = _palloc_align_up(arena_reserve, PALLOC_SEGMENT_SIZE);
  if (arena_count >= 8 && arena_count <= 128) {
    // scale up the arena sizes exponentially every 8 entries (128 entries get to 589TiB)
    const size_t multiplier = (size_t)1 << _palloc_clamp(arena_count/8, 0, 16 );
    size_t reserve = 0;
    if (!palloc_mul_overflow(multiplier, arena_reserve, &reserve)) {
      arena_reserve = reserve;
    }
  }
  if (arena_reserve < req_size) return false;  // should be able to at least handle the current allocation size

  // commit eagerly?
  bool arena_commit = false;
  if (palloc_option_get(palloc_option_arena_eager_commit) == 2)      { arena_commit = _palloc_os_has_overcommit(); }
  else if (palloc_option_get(palloc_option_arena_eager_commit) == 1) { arena_commit = true; }

  return (palloc_reserve_os_memory_ex(arena_reserve, arena_commit, allow_large, false /* exclusive? */, arena_id) == 0);
}


void* _palloc_arena_alloc_aligned(size_t size, size_t alignment, size_t align_offset, bool commit, bool allow_large,
                              palloc_arena_id_t req_arena_id, palloc_memid_t* memid)
{
  palloc_assert_internal(memid != NULL);
  palloc_assert_internal(size > 0);
  *memid = _palloc_memid_none();

  const int numa_node = _palloc_os_numa_node(); // current numa node

  // try to allocate in an arena if the alignment is small enough and the object is not too small (as for heap meta data)
  if (!palloc_option_is_enabled(palloc_option_disallow_arena_alloc)) {  // is arena allocation allowed?
    if (size >= PALLOC_ARENA_MIN_OBJ_SIZE && alignment <= PALLOC_SEGMENT_ALIGN && align_offset == 0)
    {
      void* p = palloc_arena_try_alloc(numa_node, size, alignment, commit, allow_large, req_arena_id, memid);
      if (p != NULL) return p;

      // otherwise, try to first eagerly reserve a new arena
      if (req_arena_id == _palloc_arena_id_none()) {
        palloc_arena_id_t arena_id = 0;
        if (palloc_arena_reserve(size, allow_large, &arena_id)) {
          // and try allocate in there
          palloc_assert_internal(req_arena_id == _palloc_arena_id_none());
          p = palloc_arena_try_alloc_at_id(arena_id, true, numa_node, size, alignment, commit, allow_large, req_arena_id, memid);
          if (p != NULL) return p;
        }
      }
    }
  }

  // if we cannot use OS allocation, return NULL
  if (palloc_option_is_enabled(palloc_option_disallow_os_alloc) || req_arena_id != _palloc_arena_id_none()) {
    errno = ENOMEM;
    return NULL;
  }

  // finally, fall back to the OS
  if (align_offset > 0) {
    return _palloc_os_alloc_aligned_at_offset(size, alignment, align_offset, commit, allow_large, memid);
  }
  else {
    return _palloc_os_alloc_aligned(size, alignment, commit, allow_large, memid);
  }
}

void* _palloc_arena_alloc(size_t size, bool commit, bool allow_large, palloc_arena_id_t req_arena_id, palloc_memid_t* memid)
{
  return _palloc_arena_alloc_aligned(size, PALLOC_ARENA_BLOCK_SIZE, 0, commit, allow_large, req_arena_id, memid);
}


void* palloc_arena_area(palloc_arena_id_t arena_id, size_t* size) {
  if (size != NULL) *size = 0;
  size_t arena_index = palloc_arena_id_index(arena_id);
  if (arena_index >= PALLOC_MAX_ARENAS) return NULL;
  palloc_arena_t* arena = palloc_atomic_load_ptr_acquire(palloc_arena_t, &palloc_arenas[arena_index]);
  if (arena == NULL) return NULL;
  if (size != NULL) { *size = palloc_arena_block_size(arena->block_count); }
  return arena->start;
}


/* -----------------------------------------------------------
  Arena purge
----------------------------------------------------------- */

static long palloc_arena_purge_delay(void) {
  // <0 = no purging allowed, 0=immediate purging, >0=milli-second delay
  return (palloc_option_get(palloc_option_purge_delay) * palloc_option_get(palloc_option_arena_purge_mult));
}

// reset or decommit in an arena and update the committed/decommit bitmaps
// assumes we own the area (i.e. blocks_in_use is claimed by us)
static void palloc_arena_purge(palloc_arena_t* arena, size_t bitmap_idx, size_t blocks) {
  palloc_assert_internal(arena->blocks_committed != NULL);
  palloc_assert_internal(arena->blocks_purge != NULL);
  palloc_assert_internal(!arena->memid.is_pinned);
  const size_t size = palloc_arena_block_size(blocks);
  void* const p = palloc_arena_block_start(arena, bitmap_idx);
  bool needs_recommit;
  size_t already_committed = 0;
  if (_palloc_bitmap_is_claimed_across(arena->blocks_committed, arena->field_count, blocks, bitmap_idx, &already_committed)) {
    // all blocks are committed, we can purge freely
    palloc_assert_internal(already_committed == blocks);
    needs_recommit = _palloc_os_purge(p, size);
  }
  else {
    // some blocks are not committed -- this can happen when a partially committed block is freed
    // in `_palloc_arena_free` and it is conservatively marked as uncommitted but still scheduled for a purge
    // we need to ensure we do not try to reset (as that may be invalid for uncommitted memory).
    palloc_assert_internal(already_committed < blocks);
    palloc_assert_internal(palloc_option_is_enabled(palloc_option_purge_decommits));
    needs_recommit = _palloc_os_purge_ex(p, size, false /* allow reset? */, palloc_arena_block_size(already_committed));
  }

  // clear the purged blocks
  _palloc_bitmap_unclaim_across(arena->blocks_purge, arena->field_count, blocks, bitmap_idx);
  // update committed bitmap
  if (needs_recommit) {
    _palloc_bitmap_unclaim_across(arena->blocks_committed, arena->field_count, blocks, bitmap_idx);
  }
}

// Schedule a purge. This is usually delayed to avoid repeated decommit/commit calls.
// Note: assumes we (still) own the area as we may purge immediately
static void palloc_arena_schedule_purge(palloc_arena_t* arena, size_t bitmap_idx, size_t blocks) {
  palloc_assert_internal(arena->blocks_purge != NULL);
  const long delay = palloc_arena_purge_delay();
  if (delay < 0) return;  // is purging allowed at all?

  if (_palloc_preloading() || delay == 0) {
    // decommit directly
    palloc_arena_purge(arena, bitmap_idx, blocks);
  }
  else {
    // schedule purge
    const palloc_msecs_t expire = _palloc_clock_now() + delay;
    palloc_msecs_t expire0 = 0;
    if (palloc_atomic_casi64_strong_acq_rel(&arena->purge_expire, &expire0, expire)) {
      // expiration was not yet set
      // maybe set the global arenas expire as well (if it wasn't set already)
      palloc_atomic_casi64_strong_acq_rel(&palloc_arenas_purge_expire, &expire0, expire);
    }
    else {
      // already an expiration was set
    }
    _palloc_bitmap_claim_across(arena->blocks_purge, arena->field_count, blocks, bitmap_idx, NULL, NULL);
  }
}

// purge a range of blocks
// return true if the full range was purged.
// assumes we own the area (i.e. blocks_in_use is claimed by us)
static bool palloc_arena_purge_range(palloc_arena_t* arena, size_t idx, size_t startidx, size_t bitlen, size_t purge) {
  const size_t endidx = startidx + bitlen;
  size_t bitidx = startidx;
  bool all_purged = false;
  while (bitidx < endidx) {
    // count consecutive ones in the purge mask
    size_t count = 0;
    while (bitidx + count < endidx && (purge & ((size_t)1 << (bitidx + count))) != 0) {
      count++;
    }
    if (count > 0) {
      // found range to be purged
      const palloc_bitmap_index_t range_idx = palloc_bitmap_index_create(idx, bitidx);
      palloc_arena_purge(arena, range_idx, count);
      if (count == bitlen) {
        all_purged = true;
      }
    }
    bitidx += (count+1); // +1 to skip the zero bit (or end)
  }
  return all_purged;
}

// returns true if anything was purged
static bool palloc_arena_try_purge(palloc_arena_t* arena, palloc_msecs_t now, bool force)
{
  // check pre-conditions
  if (arena->memid.is_pinned) return false;

  // expired yet?
  palloc_msecs_t expire = palloc_atomic_loadi64_relaxed(&arena->purge_expire);
  if (!force && (expire == 0 || expire > now)) return false;

  // reset expire (if not already set concurrently)
  palloc_atomic_casi64_strong_acq_rel(&arena->purge_expire, &expire, (palloc_msecs_t)0);
  _palloc_stat_counter_increase(&_palloc_stats_main.arena_purges, 1);

  // potential purges scheduled, walk through the bitmap
  bool any_purged = false;
  bool full_purge = true;
  for (size_t i = 0; i < arena->field_count; i++) {
    size_t purge = palloc_atomic_load_relaxed(&arena->blocks_purge[i]);
    if (purge != 0) {
      size_t bitidx = 0;
      while (bitidx < PALLOC_BITMAP_FIELD_BITS) {
        // find consecutive range of ones in the purge mask
        size_t bitlen = 0;
        while (bitidx + bitlen < PALLOC_BITMAP_FIELD_BITS && (purge & ((size_t)1 << (bitidx + bitlen))) != 0) {
          bitlen++;
        }
        // temporarily claim the purge range as "in-use" to be thread-safe with allocation
        // try to claim the longest range of corresponding in_use bits
        const palloc_bitmap_index_t bitmap_index = palloc_bitmap_index_create(i, bitidx);
        while( bitlen > 0 ) {
          if (_palloc_bitmap_try_claim(arena->blocks_inuse, arena->field_count, bitlen, bitmap_index)) {
            break;
          }
          bitlen--;
        }
        // actual claimed bits at `in_use`
        if (bitlen > 0) {
          // read purge again now that we have the in_use bits
          purge = palloc_atomic_load_acquire(&arena->blocks_purge[i]);
          if (!palloc_arena_purge_range(arena, i, bitidx, bitlen, purge)) {
            full_purge = false;
          }
          any_purged = true;
          // release the claimed `in_use` bits again
          _palloc_bitmap_unclaim(arena->blocks_inuse, arena->field_count, bitlen, bitmap_index);
        }
        bitidx += (bitlen+1);  // +1 to skip the zero (or end)
      } // while bitidx
    } // purge != 0
  }
  // if not fully purged, make sure to purge again in the future
  if (!full_purge) {
    const long delay = palloc_arena_purge_delay();
    palloc_msecs_t expected = 0;
    palloc_atomic_casi64_strong_acq_rel(&arena->purge_expire,&expected,_palloc_clock_now() + delay);
  }
  return any_purged;
}

static void palloc_arenas_try_purge( bool force, bool visit_all )
{
  if (_palloc_preloading() || palloc_arena_purge_delay() <= 0) return;  // nothing will be scheduled

  // check if any arena needs purging?
  const palloc_msecs_t now = _palloc_clock_now();
  palloc_msecs_t arenas_expire = palloc_atomic_loadi64_acquire(&palloc_arenas_purge_expire);
  if (!force && (arenas_expire == 0 || arenas_expire < now)) return;

  const size_t max_arena = palloc_atomic_load_acquire(&palloc_arena_count);
  if (max_arena == 0) return;

  // allow only one thread to purge at a time
  static palloc_atomic_guard_t purge_guard;
  palloc_atomic_guard(&purge_guard)
  {
    // increase global expire: at most one purge per delay cycle
    palloc_atomic_storei64_release(&palloc_arenas_purge_expire, now + palloc_arena_purge_delay());
    size_t max_purge_count = (visit_all ? max_arena : 2);
    bool all_visited = true;
    for (size_t i = 0; i < max_arena; i++) {
      palloc_arena_t* arena = palloc_atomic_load_ptr_acquire(palloc_arena_t, &palloc_arenas[i]);
      if (arena != NULL) {
        if (palloc_arena_try_purge(arena, now, force)) {
          if (max_purge_count <= 1) {
            all_visited = false;
            break;
          }
          max_purge_count--;
        }
      }
    }
    if (all_visited) {
      // all arena's were visited and purged: reset global expire
      palloc_atomic_storei64_release(&palloc_arenas_purge_expire, 0);
    }
  }
}


/* -----------------------------------------------------------
  Arena free
----------------------------------------------------------- */

void _palloc_arena_free(void* p, size_t size, size_t committed_size, palloc_memid_t memid) {
  palloc_assert_internal(size > 0);
  palloc_assert_internal(committed_size <= size);
  if (p==NULL) return;
  if (size==0) return;
  const bool all_committed = (committed_size == size);
  const size_t decommitted_size = (committed_size <= size ? size - committed_size : 0);

  // need to set all memory to undefined as some parts may still be marked as no_access (like padding etc.)
  palloc_track_mem_undefined(p,size);

  if (palloc_memkind_is_os(memid.memkind)) {
    // was a direct OS allocation, pass through
    if (!all_committed && decommitted_size > 0) {
      // if partially committed, adjust the committed stats (as `_palloc_os_free` will decrease commit by the full size)
      _palloc_stat_increase(&_palloc_stats_main.committed, decommitted_size);
    }
    _palloc_os_free(p, size, memid);
  }
  else if (memid.memkind == PALLOC_MEM_ARENA) {
    // allocated in an arena
    size_t arena_idx;
    size_t bitmap_idx;
    palloc_arena_memid_indices(memid, &arena_idx, &bitmap_idx);
    palloc_assert_internal(arena_idx < PALLOC_MAX_ARENAS);
    palloc_arena_t* arena = palloc_atomic_load_ptr_acquire(palloc_arena_t,&palloc_arenas[arena_idx]);
    palloc_assert_internal(arena != NULL);
    const size_t blocks = palloc_block_count_of_size(size);

    // checks
    if (arena == NULL) {
      _palloc_error_message(EINVAL, "trying to free from an invalid arena: %p, size %zu, memid: 0x%zx\n", p, size, memid);
      return;
    }
    palloc_assert_internal(arena->field_count > palloc_bitmap_index_field(bitmap_idx));
    if (arena->field_count <= palloc_bitmap_index_field(bitmap_idx)) {
      _palloc_error_message(EINVAL, "trying to free from an invalid arena block: %p, size %zu, memid: 0x%zx\n", p, size, memid);
      return;
    }

    // potentially decommit
    if (arena->memid.is_pinned || arena->blocks_committed == NULL) {
      palloc_assert_internal(all_committed);
    }
    else {
      palloc_assert_internal(arena->blocks_committed != NULL);
      palloc_assert_internal(arena->blocks_purge != NULL);

      if (!all_committed) {
        // mark the entire range as no longer committed (so we will recommit the full range when re-using)
        _palloc_bitmap_unclaim_across(arena->blocks_committed, arena->field_count, blocks, bitmap_idx);
        palloc_track_mem_noaccess(p,size);
        //if (committed_size > 0) {
          // if partially committed, adjust the committed stats (is it will be recommitted when re-using)
          // in the delayed purge, we do no longer decrease the commit if the range is not marked entirely as committed.
          _palloc_stat_decrease(&_palloc_stats_main.committed, committed_size);
        //}
        // note: if not all committed, it may be that the purge will reset/decommit the entire range
        // that contains already decommitted parts. Since purge consistently uses reset or decommit that
        // works (as we should never reset decommitted parts).
      }
      // (delay) purge the entire range
      palloc_arena_schedule_purge(arena, bitmap_idx, blocks);
    }

    // and make it available to others again
    bool all_inuse = _palloc_bitmap_unclaim_across(arena->blocks_inuse, arena->field_count, blocks, bitmap_idx);
    if (!all_inuse) {
      _palloc_error_message(EAGAIN, "trying to free an already freed arena block: %p, size %zu\n", p, size);
      return;
    };
  }
  else {
    // arena was none, external, or static; nothing to do
    palloc_assert_internal(memid.memkind < PALLOC_MEM_OS);
  }

  // purge expired decommits
  palloc_arenas_try_purge(false, false);
}

// destroy owned arenas; this is unsafe and should only be done using `palloc_option_destroy_on_exit`
// for dynamic libraries that are unloaded and need to release all their allocated memory.
static void palloc_arenas_unsafe_destroy(void) {
  const size_t max_arena = palloc_atomic_load_relaxed(&palloc_arena_count);
  size_t new_max_arena = 0;
  for (size_t i = 0; i < max_arena; i++) {
    palloc_arena_t* arena = palloc_atomic_load_ptr_acquire(palloc_arena_t, &palloc_arenas[i]);
    if (arena != NULL) {
      palloc_lock_done(&arena->abandoned_visit_lock);
      if (arena->start != NULL && palloc_memkind_is_os(arena->memid.memkind)) {
        palloc_atomic_store_ptr_release(palloc_arena_t, &palloc_arenas[i], NULL);
        _palloc_os_free(arena->start, palloc_arena_size(arena), arena->memid);
      }
      else {
        new_max_arena = i;
      }
      _palloc_arena_meta_free(arena, arena->meta_memid, arena->meta_size);
    }
  }

  // try to lower the max arena.
  size_t expected = max_arena;
  palloc_atomic_cas_strong_acq_rel(&palloc_arena_count, &expected, new_max_arena);
}

// Purge the arenas; if `force_purge` is true, amenable parts are purged even if not yet expired
void _palloc_arenas_collect(bool force_purge) {
  palloc_arenas_try_purge(force_purge, force_purge /* visit all? */);
}

// destroy owned arenas; this is unsafe and should only be done using `palloc_option_destroy_on_exit`
// for dynamic libraries that are unloaded and need to release all their allocated memory.
void _palloc_arena_unsafe_destroy_all(void) {
  palloc_arenas_unsafe_destroy();
  _palloc_arenas_collect(true /* force purge */);  // purge non-owned arenas
}

// Is a pointer inside any of our arenas?
bool _palloc_arena_contains(const void* p) {
  const size_t max_arena = palloc_atomic_load_relaxed(&palloc_arena_count);
  for (size_t i = 0; i < max_arena; i++) {
    palloc_arena_t* arena = palloc_atomic_load_ptr_relaxed(palloc_arena_t, &palloc_arenas[i]);
    if (arena != NULL && arena->start <= (const uint8_t*)p && arena->start + palloc_arena_block_size(arena->block_count) > (const uint8_t*)p) {
      return true;
    }
  }
  return false;
}

/* -----------------------------------------------------------
  Add an arena.
----------------------------------------------------------- */

static bool palloc_arena_add(palloc_arena_t* arena, palloc_arena_id_t* arena_id, palloc_stats_t* stats) {
  palloc_assert_internal(arena != NULL);
  palloc_assert_internal((uintptr_t)palloc_atomic_load_ptr_relaxed(uint8_t,&arena->start) % PALLOC_SEGMENT_ALIGN == 0);
  palloc_assert_internal(arena->block_count > 0);
  if (arena_id != NULL) { *arena_id = -1; }

  size_t i = palloc_atomic_load_relaxed(&palloc_arena_count);
  while (i < PALLOC_MAX_ARENAS) {
    if (palloc_atomic_cas_strong_acq_rel(&palloc_arena_count, &i, i+1)) {
      _palloc_stat_counter_increase(&stats->arena_count, 1);
      arena->id = palloc_arena_id_create(i);
      palloc_atomic_store_ptr_release(palloc_arena_t, &palloc_arenas[i], arena);
      if (arena_id != NULL) { *arena_id = arena->id; }
      return true;
    }
  }

  return false;
}

static bool palloc_manage_os_memory_ex2(void* start, size_t size, bool is_large, int numa_node, bool exclusive, palloc_memid_t memid, palloc_arena_id_t* arena_id) palloc_attr_noexcept
{
  if (arena_id != NULL) *arena_id = _palloc_arena_id_none();
  if (size < PALLOC_ARENA_BLOCK_SIZE) {
    _palloc_warning_message("the arena size is too small (memory at %p with size %zu)\n", start, size);
    return false;
  }
  if (is_large) {
    palloc_assert_internal(memid.initially_committed && memid.is_pinned);
  }
  if (!_palloc_is_aligned(start, PALLOC_SEGMENT_ALIGN)) {
    void* const aligned_start = palloc_align_up_ptr(start, PALLOC_SEGMENT_ALIGN);
    const size_t diff = (uint8_t*)aligned_start - (uint8_t*)start;
    if (diff >= size || (size - diff) < PALLOC_ARENA_BLOCK_SIZE) {
      _palloc_warning_message("after alignment, the size of the arena becomes too small (memory at %p with size %zu)\n", start, size);
      return false;
    }
    start = aligned_start;
    size = size - diff;
  }

  const size_t bcount = size / PALLOC_ARENA_BLOCK_SIZE;
  const size_t fields = _palloc_divide_up(bcount, PALLOC_BITMAP_FIELD_BITS);
  const size_t bitmaps = (memid.is_pinned ? 3 : 5);
  const size_t asize  = sizeof(palloc_arena_t) + (bitmaps*fields*sizeof(palloc_bitmap_field_t));
  palloc_memid_t meta_memid;
  palloc_arena_t* arena   = (palloc_arena_t*)_palloc_arena_meta_zalloc(asize, &meta_memid);
  if (arena == NULL) return false;

  // already zero'd due to zalloc
  // _palloc_memzero(arena, asize);
  arena->id = _palloc_arena_id_none();
  arena->memid = memid;
  arena->exclusive = exclusive;
  arena->meta_size = asize;
  arena->meta_memid = meta_memid;
  arena->block_count = bcount;
  arena->field_count = fields;
  arena->start = (uint8_t*)start;
  arena->numa_node    = numa_node; // TODO: or get the current numa node if -1? (now it allows anyone to allocate on -1)
  arena->is_large     = is_large;
  arena->purge_expire = 0;
  arena->search_idx   = 0;
  palloc_lock_init(&arena->abandoned_visit_lock);
  // consecutive bitmaps
  arena->blocks_dirty     = &arena->blocks_inuse[fields];     // just after inuse bitmap
  arena->blocks_abandoned = &arena->blocks_inuse[2 * fields]; // just after dirty bitmap
  arena->blocks_committed = (arena->memid.is_pinned ? NULL : &arena->blocks_inuse[3*fields]); // just after abandoned bitmap
  arena->blocks_purge     = (arena->memid.is_pinned ? NULL : &arena->blocks_inuse[4*fields]); // just after committed bitmap
  // initialize committed bitmap?
  if (arena->blocks_committed != NULL && arena->memid.initially_committed) {
    memset((void*)arena->blocks_committed, 0xFF, fields*sizeof(palloc_bitmap_field_t)); // cast to void* to avoid atomic warning
  }

  // and claim leftover blocks if needed (so we never allocate there)
  ptrdiff_t post = (fields * PALLOC_BITMAP_FIELD_BITS) - bcount;
  palloc_assert_internal(post >= 0);
  if (post > 0) {
    // don't use leftover bits at the end
    palloc_bitmap_index_t postidx = palloc_bitmap_index_create(fields - 1, PALLOC_BITMAP_FIELD_BITS - post);
    _palloc_bitmap_claim(arena->blocks_inuse, fields, post, postidx, NULL);
  }
  return palloc_arena_add(arena, arena_id, &_palloc_stats_main);

}

bool palloc_manage_os_memory_ex(void* start, size_t size, bool is_committed, bool is_large, bool is_zero, int numa_node, bool exclusive, palloc_arena_id_t* arena_id) palloc_attr_noexcept {
  palloc_memid_t memid = _palloc_memid_create(PALLOC_MEM_EXTERNAL);
  memid.initially_committed = is_committed;
  memid.initially_zero = is_zero;
  memid.is_pinned = is_large;
  return palloc_manage_os_memory_ex2(start,size,is_large,numa_node,exclusive,memid, arena_id);
}

// Reserve a range of regular OS memory
int palloc_reserve_os_memory_ex(size_t size, bool commit, bool allow_large, bool exclusive, palloc_arena_id_t* arena_id) palloc_attr_noexcept {
  if (arena_id != NULL) *arena_id = _palloc_arena_id_none();
  size = _palloc_align_up(size, PALLOC_ARENA_BLOCK_SIZE); // at least one block
  palloc_memid_t memid;
  void* start = _palloc_os_alloc_aligned(size, PALLOC_SEGMENT_ALIGN, commit, allow_large, &memid);
  if (start == NULL) return ENOMEM;
  const bool is_large = memid.is_pinned; // todo: use separate is_large field?
  if (!palloc_manage_os_memory_ex2(start, size, is_large, -1 /* numa node */, exclusive, memid, arena_id)) {
    _palloc_os_free_ex(start, size, commit, memid);
    _palloc_verbose_message("failed to reserve %zu KiB memory\n", _palloc_divide_up(size, 1024));
    return ENOMEM;
  }
  _palloc_verbose_message("reserved %zu KiB memory%s\n", _palloc_divide_up(size, 1024), is_large ? " (in large os pages)" : "");
  return 0;
}


// Manage a range of regular OS memory
bool palloc_manage_os_memory(void* start, size_t size, bool is_committed, bool is_large, bool is_zero, int numa_node) palloc_attr_noexcept {
  return palloc_manage_os_memory_ex(start, size, is_committed, is_large, is_zero, numa_node, false /* exclusive? */, NULL);
}

// Reserve a range of regular OS memory
int palloc_reserve_os_memory(size_t size, bool commit, bool allow_large) palloc_attr_noexcept {
  return palloc_reserve_os_memory_ex(size, commit, allow_large, false, NULL);
}


/* -----------------------------------------------------------
  Debugging
----------------------------------------------------------- */

static size_t palloc_debug_show_bitmap(const char* prefix, const char* header, size_t block_count, palloc_bitmap_field_t* fields, size_t field_count ) {
  _palloc_message("%s%s:\n", prefix, header);
  size_t bcount = 0;
  size_t inuse_count = 0;
  for (size_t i = 0; i < field_count; i++) {
    char buf[PALLOC_BITMAP_FIELD_BITS + 1];
    uintptr_t field = palloc_atomic_load_relaxed(&fields[i]);
    for (size_t bit = 0; bit < PALLOC_BITMAP_FIELD_BITS; bit++, bcount++) {
      if (bcount < block_count) {
        bool inuse = ((((uintptr_t)1 << bit) & field) != 0);
        if (inuse) inuse_count++;
        buf[bit] = (inuse ? 'x' : '.');
      }
      else {
        buf[bit] = ' ';
      }
    }
    buf[PALLOC_BITMAP_FIELD_BITS] = 0;
    _palloc_message("%s  %s\n", prefix, buf);
  }
  _palloc_message("%s  total ('x'): %zu\n", prefix, inuse_count);
  return inuse_count;
}

void palloc_debug_show_arenas(void) palloc_attr_noexcept {
  const bool show_inuse = true;
  size_t max_arenas = palloc_atomic_load_relaxed(&palloc_arena_count);
  size_t inuse_total = 0;
  //size_t abandoned_total = 0;
  //size_t purge_total = 0;
  for (size_t i = 0; i < max_arenas; i++) {
    palloc_arena_t* arena = palloc_atomic_load_ptr_relaxed(palloc_arena_t, &palloc_arenas[i]);
    if (arena == NULL) break;
    _palloc_message("arena %zu: %zu blocks of size %zuMiB (in %zu fields) %s\n", i, arena->block_count, (size_t)(PALLOC_ARENA_BLOCK_SIZE / PALLOC_MiB), arena->field_count, (arena->memid.is_pinned ? ", pinned" : ""));
    if (show_inuse) {
      inuse_total += palloc_debug_show_bitmap("  ", "inuse blocks", arena->block_count, arena->blocks_inuse, arena->field_count);
    }
    if (arena->blocks_committed != NULL) {
      palloc_debug_show_bitmap("  ", "committed blocks", arena->block_count, arena->blocks_committed, arena->field_count);
    }
    //if (show_abandoned) {
    //  abandoned_total += palloc_debug_show_bitmap("  ", "abandoned blocks", arena->block_count, arena->blocks_abandoned, arena->field_count);
    //}
    //if (show_purge && arena->blocks_purge != NULL) {
    //  purge_total += palloc_debug_show_bitmap("  ", "purgeable blocks", arena->block_count, arena->blocks_purge, arena->field_count);
    //}
  }
  if (show_inuse)     _palloc_message("total inuse blocks    : %zu\n", inuse_total);
  //if (show_abandoned) _palloc_message("total abandoned blocks: %zu\n", abandoned_total);
  //if (show_purge)     _palloc_message("total purgeable blocks: %zu\n", purge_total);
}


void palloc_arenas_print(void) palloc_attr_noexcept {
  palloc_debug_show_arenas();
}


/* -----------------------------------------------------------
  Reserve a huge page arena.
----------------------------------------------------------- */
// reserve at a specific numa node
int palloc_reserve_huge_os_pages_at_ex(size_t pages, int numa_node, size_t timeout_msecs, bool exclusive, palloc_arena_id_t* arena_id) palloc_attr_noexcept {
  if (arena_id != NULL) *arena_id = -1;
  if (pages==0) return 0;
  if (numa_node < -1) numa_node = -1;
  if (numa_node >= 0) numa_node = numa_node % _palloc_os_numa_node_count();
  size_t hsize = 0;
  size_t pages_reserved = 0;
  palloc_memid_t memid;
  void* p = _palloc_os_alloc_huge_os_pages(pages, numa_node, timeout_msecs, &pages_reserved, &hsize, &memid);
  if (p==NULL || pages_reserved==0) {
    _palloc_warning_message("failed to reserve %zu GiB huge pages\n", pages);
    return ENOMEM;
  }
  _palloc_verbose_message("numa node %i: reserved %zu GiB huge pages (of the %zu GiB requested)\n", numa_node, pages_reserved, pages);

  if (!palloc_manage_os_memory_ex2(p, hsize, true, numa_node, exclusive, memid, arena_id)) {
    _palloc_os_free(p, hsize, memid);
    return ENOMEM;
  }
  return 0;
}

int palloc_reserve_huge_os_pages_at(size_t pages, int numa_node, size_t timeout_msecs) palloc_attr_noexcept {
  return palloc_reserve_huge_os_pages_at_ex(pages, numa_node, timeout_msecs, false, NULL);
}

// reserve huge pages evenly among the given number of numa nodes (or use the available ones as detected)
int palloc_reserve_huge_os_pages_interleave(size_t pages, size_t numa_nodes, size_t timeout_msecs) palloc_attr_noexcept {
  if (pages == 0) return 0;

  // pages per numa node
  int numa_count = (numa_nodes > 0 && numa_nodes <= INT_MAX ? (int)numa_nodes : _palloc_os_numa_node_count());
  if (numa_count == 0) numa_count = 1;
  const size_t pages_per = pages / numa_count;
  const size_t pages_mod = pages % numa_count;
  const size_t timeout_per = (timeout_msecs==0 ? 0 : (timeout_msecs / numa_count) + 50);

  // reserve evenly among numa nodes
  for (int numa_node = 0; numa_node < numa_count && pages > 0; numa_node++) {
    size_t node_pages = pages_per;  // can be 0
    if ((size_t)numa_node < pages_mod) node_pages++;
    int err = palloc_reserve_huge_os_pages_at(node_pages, numa_node, timeout_per);
    if (err) return err;
    if (pages < node_pages) {
      pages = 0;
    }
    else {
      pages -= node_pages;
    }
  }

  return 0;
}

int palloc_reserve_huge_os_pages(size_t pages, double max_secs, size_t* pages_reserved) palloc_attr_noexcept {
  PALLOC_UNUSED(max_secs);
  _palloc_warning_message("palloc_reserve_huge_os_pages is deprecated: use palloc_reserve_huge_os_pages_interleave/at instead\n");
  if (pages_reserved != NULL) *pages_reserved = 0;
  int err = palloc_reserve_huge_os_pages_interleave(pages, 0, (size_t)(max_secs * 1000.0));
  if (err==0 && pages_reserved!=NULL) *pages_reserved = pages;
  return err;
}
