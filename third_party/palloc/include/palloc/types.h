/* ----------------------------------------------------------------------------
Copyright (c) 2018-2024, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_TYPES_H
#define PALLOC_TYPES_H

// --------------------------------------------------------------------------
// This file contains the main type definitions for palloc:
// palloc_heap_t      : all data for a thread-local heap, contains
//                  lists of all managed heap pages.
// palloc_segment_t   : a larger chunk of memory (32MiB on 64-bit) from where pages
//                  are allocated. A segment is divided in slices (64KiB) from
//                  which pages are allocated.
// palloc_page_t      : a "palloc" page (usually 64KiB or 512KiB) from
//                  where objects are allocated.
//                  Note: we write "OS page" for OS memory pages while
//                  using plain "page" for palloc pages (`palloc_page_t`).
// --------------------------------------------------------------------------


#include <palloc-stats.h>
#include <stddef.h>   // ptrdiff_t
#include <stdint.h>   // uintptr_t, uint16_t, etc
#include <stdbool.h>  // bool
#include "atomic.h"   // _Atomic

#ifdef _MSC_VER
#pragma warning(disable:4214) // bitfield is not int
#endif

// Minimal alignment necessary. On most platforms 16 bytes are needed
// due to SSE registers for example. This must be at least `sizeof(void*)`
#ifndef PALLOC_MAX_ALIGN_SIZE
#define PALLOC_MAX_ALIGN_SIZE  16   // sizeof(max_align_t)
#endif

// ------------------------------------------------------
// Variants
// ------------------------------------------------------

// Define NDEBUG in the release version to disable assertions.
// #define NDEBUG

// Define PALLOC_TRACK_<tool> to enable tracking support
// #define PALLOC_TRACK_VALGRIND 1
// #define PALLOC_TRACK_ASAN     1
// #define PALLOC_TRACK_ETW      1

// Define PALLOC_STAT as 1 to maintain statistics; set it to 2 to have detailed statistics (but costs some performance).
// #define PALLOC_STAT 1

// Define PALLOC_SECURE to enable security mitigations
// #define PALLOC_SECURE 1  // guard page around metadata
// #define PALLOC_SECURE 2  // guard page around each palloc page
// #define PALLOC_SECURE 3  // encode free lists (detect corrupted free list (buffer overflow), and invalid pointer free)
// #define PALLOC_SECURE 4  // checks for double free. (may be more expensive)

#if !defined(PALLOC_SECURE)
#define PALLOC_SECURE 0
#endif

// Define PALLOC_DEBUG for assertion and invariant checking
// #define PALLOC_DEBUG 1  // basic assertion checks and statistics, check double free, corrupted free list, and invalid pointer free. (cmake -DMI_DEBUG=ON)
// #define PALLOC_DEBUG 2  // + internal assertion checks (cmake -DMI_DEBUG_INTERNAL=ON)
// #define PALLOC_DEBUG 3  // + extensive internal invariant checking (cmake -DMI_DEBUG_FULL=ON)
#if !defined(PALLOC_DEBUG)
#if defined(PALLOC_BUILD_RELEASE) || defined(NDEBUG)
#define PALLOC_DEBUG 0
#else
#define PALLOC_DEBUG 2
#endif
#endif

// Use guard pages behind objects of a certain size (set by the PALLOC_DEBUG_GUARDED_MIN/MAX options)
// Padding should be disabled when using guard pages
// #define PALLOC_GUARDED 1
#if defined(PALLOC_GUARDED)
#define PALLOC_PADDING  0
#endif

// Reserve extra padding at the end of each block to be more resilient against heap block overflows.
// The padding can detect buffer overflow on free.
#if !defined(PALLOC_PADDING) && (PALLOC_SECURE>=3 || PALLOC_DEBUG>=1 || (PALLOC_TRACK_VALGRIND || PALLOC_TRACK_ASAN || PALLOC_TRACK_ETW))
#define PALLOC_PADDING  1
#endif

// Check padding bytes; allows byte-precise buffer overflow detection
#if !defined(PALLOC_PADDING_CHECK) && PALLOC_PADDING && (PALLOC_SECURE>=3 || PALLOC_DEBUG>=1)
#define PALLOC_PADDING_CHECK 1
#endif


// Encoded free lists allow detection of corrupted free lists
// and can detect buffer overflows, modify after free, and double `free`s.
#if (PALLOC_SECURE>=3 || PALLOC_DEBUG>=1)
#define PALLOC_ENCODE_FREELIST  1
#endif


// We used to abandon huge pages in order to eagerly deallocate it if freed from another thread.
// Unfortunately, that makes it not possible to visit them during a heap walk or include them in a
// `palloc_heap_destroy`. We therefore instead reset/decommit the huge blocks nowadays if freed from
// another thread so the memory becomes "virtually" available (and eventually gets properly freed by
// the owning thread).
// #define PALLOC_HUGE_PAGE_ABANDON 1


// ------------------------------------------------------
// Platform specific values
// ------------------------------------------------------

// ------------------------------------------------------
// Size of a pointer.
// We assume that `sizeof(void*)==sizeof(intptr_t)`
// and it holds for all platforms we know of.
//
// However, the C standard only requires that:
//  p == (void*)((intptr_t)p))
// but we also need:
//  i == (intptr_t)((void*)i)
// or otherwise one might define an intptr_t type that is larger than a pointer...
// ------------------------------------------------------

#if INTPTR_MAX > INT64_MAX
# define PALLOC_INTPTR_SHIFT (4)  // assume 128-bit  (as on arm CHERI for example)
#elif INTPTR_MAX == INT64_MAX
# define PALLOC_INTPTR_SHIFT (3)
#elif INTPTR_MAX == INT32_MAX
# define PALLOC_INTPTR_SHIFT (2)
#else
#error platform pointers must be 32, 64, or 128 bits
#endif

#if SIZE_MAX == UINT64_MAX
# define PALLOC_SIZE_SHIFT (3)
typedef int64_t  palloc_ssize_t;
#elif SIZE_MAX == UINT32_MAX
# define PALLOC_SIZE_SHIFT (2)
typedef int32_t  palloc_ssize_t;
#else
#error platform objects must be 32 or 64 bits
#endif

#if (SIZE_MAX/2) > LONG_MAX
# define PALLOC_ZU(x)  x##ULL
# define PALLOC_ZI(x)  x##LL
#else
# define PALLOC_ZU(x)  x##UL
# define PALLOC_ZI(x)  x##L
#endif

#define PALLOC_INTPTR_SIZE  (1<<PALLOC_INTPTR_SHIFT)
#define PALLOC_INTPTR_BITS  (PALLOC_INTPTR_SIZE*8)

#define PALLOC_SIZE_SIZE  (1<<PALLOC_SIZE_SHIFT)
#define PALLOC_SIZE_BITS  (PALLOC_SIZE_SIZE*8)

#define PALLOC_KiB     (PALLOC_ZU(1024))
#define PALLOC_MiB     (PALLOC_KiB*PALLOC_KiB)
#define PALLOC_GiB     (PALLOC_MiB*PALLOC_KiB)


// ------------------------------------------------------
// Main internal data-structures
// ------------------------------------------------------

// Main tuning parameters for segment and page sizes
// Sizes for 64-bit (usually divide by two for 32-bit)
#ifndef PALLOC_SEGMENT_SLICE_SHIFT
#define PALLOC_SEGMENT_SLICE_SHIFT            (13 + PALLOC_INTPTR_SHIFT)         // 64KiB  (32KiB on 32-bit)
#endif

#ifndef PALLOC_SEGMENT_SHIFT
#if PALLOC_INTPTR_SIZE > 4
#define PALLOC_SEGMENT_SHIFT                  ( 9 + PALLOC_SEGMENT_SLICE_SHIFT)  // 32MiB
#else
#define PALLOC_SEGMENT_SHIFT                  ( 7 + PALLOC_SEGMENT_SLICE_SHIFT)  // 4MiB on 32-bit
#endif
#endif

#ifndef PALLOC_SMALL_PAGE_SHIFT
#define PALLOC_SMALL_PAGE_SHIFT               (PALLOC_SEGMENT_SLICE_SHIFT)       // 64KiB
#endif
#ifndef PALLOC_MEDIUM_PAGE_SHIFT
#define PALLOC_MEDIUM_PAGE_SHIFT              ( 3 + PALLOC_SMALL_PAGE_SHIFT)     // 512KiB
#endif

// Derived constants
#define PALLOC_SEGMENT_SIZE                   (PALLOC_ZU(1)<<PALLOC_SEGMENT_SHIFT)
#define PALLOC_SEGMENT_ALIGN                  PALLOC_SEGMENT_SIZE
#define PALLOC_SEGMENT_MASK                   ((uintptr_t)(PALLOC_SEGMENT_ALIGN - 1))
#define PALLOC_SEGMENT_SLICE_SIZE             (PALLOC_ZU(1)<< PALLOC_SEGMENT_SLICE_SHIFT)
#define PALLOC_SLICES_PER_SEGMENT             (PALLOC_SEGMENT_SIZE / PALLOC_SEGMENT_SLICE_SIZE) // 512 (128 on 32-bit)

#define PALLOC_SMALL_PAGE_SIZE                (PALLOC_ZU(1)<<PALLOC_SMALL_PAGE_SHIFT)
#define PALLOC_MEDIUM_PAGE_SIZE               (PALLOC_ZU(1)<<PALLOC_MEDIUM_PAGE_SHIFT)

#define PALLOC_SMALL_OBJ_SIZE_MAX             (PALLOC_SMALL_PAGE_SIZE/8)   // 8 KiB on 64-bit
#define PALLOC_MEDIUM_OBJ_SIZE_MAX            (PALLOC_MEDIUM_PAGE_SIZE/8)  // 64 KiB on 64-bit
#define PALLOC_MEDIUM_OBJ_WSIZE_MAX           (PALLOC_MEDIUM_OBJ_SIZE_MAX/PALLOC_INTPTR_SIZE)
#define PALLOC_LARGE_OBJ_SIZE_MAX             (PALLOC_SEGMENT_SIZE/2)      // 16 MiB on 64-bit
#define PALLOC_LARGE_OBJ_WSIZE_MAX            (PALLOC_LARGE_OBJ_SIZE_MAX/PALLOC_INTPTR_SIZE)

// Maximum number of size classes. (spaced exponentially in 12.5% increments)
#if PALLOC_BIN_HUGE != 73U
#error "palloc internal: expecting 73 bins"
#endif

#if (PALLOC_MEDIUM_OBJ_WSIZE_MAX >= 655360)
#error "palloc internal: define more bins"
#endif

// Maximum block size for which blocks are guaranteed to be block size aligned. (see `segment.c:_palloc_segment_page_start`)
#define PALLOC_MAX_ALIGN_GUARANTEE            (PALLOC_MEDIUM_OBJ_SIZE_MAX)

// Alignments over PALLOC_BLOCK_ALIGNMENT_MAX are allocated in dedicated huge page segments
#define PALLOC_BLOCK_ALIGNMENT_MAX            (PALLOC_SEGMENT_SIZE >> 1)

// Maximum slice count (255) for which we can find the page for interior pointers
#define PALLOC_MAX_SLICE_OFFSET_COUNT         ((PALLOC_BLOCK_ALIGNMENT_MAX / PALLOC_SEGMENT_SLICE_SIZE) - 1)

// we never allocate more than PTRDIFF_MAX (see also <https://sourceware.org/ml/libc-announce/2019/msg00001.html>)
// on 64-bit+ systems we also limit the maximum allocation size such that the slice count fits in 32-bits. (issue #877)
#if (PTRDIFF_MAX > INT32_MAX) && (PTRDIFF_MAX >= (PALLOC_SEGMENT_SLIZE_SIZE * UINT32_MAX))
#define PALLOC_MAX_ALLOC_SIZE   (PALLOC_SEGMENT_SLICE_SIZE * (UINT32_MAX-1))
#else
#define PALLOC_MAX_ALLOC_SIZE   PTRDIFF_MAX
#endif


// ------------------------------------------------------
// Mimalloc pages contain allocated blocks
// ------------------------------------------------------

// The free lists use encoded next fields
// (Only actually encodes when PALLOC_ENCODED_FREELIST is defined.)
typedef uintptr_t  palloc_encoded_t;

// thread id's
typedef size_t     palloc_threadid_t;

// free lists contain blocks
typedef struct palloc_block_s {
  palloc_encoded_t next;
} palloc_block_t;

#if PALLOC_GUARDED
// we always align guarded pointers in a block at an offset
// the block `next` field is then used as a tag to distinguish regular offset aligned blocks from guarded ones
#define PALLOC_BLOCK_TAG_ALIGNED   ((palloc_encoded_t)(0))
#define PALLOC_BLOCK_TAG_GUARDED   (~PALLOC_BLOCK_TAG_ALIGNED)
#endif


// The delayed flags are used for efficient multi-threaded free-ing
typedef enum palloc_delayed_e {
  PALLOC_USE_DELAYED_FREE   = 0, // push on the owning heap thread delayed list
  PALLOC_DELAYED_FREEING    = 1, // temporary: another thread is accessing the owning heap
  PALLOC_NO_DELAYED_FREE    = 2, // optimize: push on page local thread free queue if another block is already in the heap thread delayed free list
  PALLOC_NEVER_DELAYED_FREE = 3  // sticky: used for abandoned pages without a owning heap; this only resets on page reclaim
} palloc_delayed_t;


// The `in_full` and `has_aligned` page flags are put in a union to efficiently
// test if both are false (`full_aligned == 0`) in the `palloc_free` routine.
#if !PALLOC_TSAN
typedef union palloc_page_flags_s {
  uint8_t full_aligned;
  struct {
    uint8_t in_full : 1;
    uint8_t has_aligned : 1;
  } x;
} palloc_page_flags_t;
#else
// under thread sanitizer, use a byte for each flag to suppress warning, issue #130
typedef union palloc_page_flags_s {
  uint32_t full_aligned;
  struct {
    uint8_t in_full;
    uint8_t has_aligned;
  } x;
} palloc_page_flags_t;
#endif

// Thread free list.
// We use the bottom 2 bits of the pointer for palloc_delayed_t flags
typedef uintptr_t palloc_thread_free_t;

// A page contains blocks of one specific size (`block_size`).
// Each page has three list of free blocks:
// `free` for blocks that can be allocated,
// `local_free` for freed blocks that are not yet available to `palloc_malloc`
// `thread_free` for freed blocks by other threads
// The `local_free` and `thread_free` lists are migrated to the `free` list
// when it is exhausted. The separate `local_free` list is necessary to
// implement a monotonic heartbeat. The `thread_free` list is needed for
// avoiding atomic operations in the common case.
//
// `used - |thread_free|` == actual blocks that are in use (alive)
// `used - |thread_free| + |free| + |local_free| == capacity`
//
// We don't count `freed` (as |free|) but use `used` to reduce
// the number of memory accesses in the `palloc_page_all_free` function(s).
//
// Notes:
// - Access is optimized for `free.c:palloc_free` and `alloc.c:palloc_page_alloc`
// - Using `uint16_t` does not seem to slow things down
// - The size is 12 words on 64-bit which helps the page index calculations
//   (and 14 words on 32-bit, and encoded free lists add 2 words)
// - `xthread_free` uses the bottom bits as a delayed-free flags to optimize
//   concurrent frees where only the first concurrent free adds to the owning
//   heap `thread_delayed_free` list (see `free.c:palloc_free_block_mt`).
//   The invariant is that no-delayed-free is only set if there is
//   at least one block that will be added, or as already been added, to
//   the owning heap `thread_delayed_free` list. This guarantees that pages
//   will be freed correctly even if only other threads free blocks.
typedef struct palloc_page_s {
  // "owned" by the segment
  uint32_t              slice_count;       // slices in this page (0 if not a page)
  uint32_t              slice_offset;      // distance from the actual page data slice (0 if a page)
  uint8_t               is_committed:1;    // `true` if the page virtual memory is committed
  uint8_t               is_zero_init:1;    // `true` if the page was initially zero initialized
  uint8_t               is_huge:1;         // `true` if the page is in a huge segment (`segment->kind == PALLOC_SEGMENT_HUGE`)
                                           // padding
  // layout like this to optimize access in `palloc_malloc` and `palloc_free`
  uint16_t              capacity;          // number of blocks committed, must be the first field, see `segment.c:page_clear`
  uint16_t              reserved;          // number of blocks reserved in memory
  palloc_page_flags_t       flags;             // `in_full` and `has_aligned` flags (8 bits)
  uint8_t               free_is_zero:1;    // `true` if the blocks in the free list are zero initialized
  uint8_t               retire_expire:7;   // expiration count for retired blocks

  palloc_block_t*           free;              // list of available free blocks (`malloc` allocates from this list)
  palloc_block_t*           local_free;        // list of deferred free blocks by this thread (migrates to `free`)
  uint16_t              used;              // number of blocks in use (including blocks in `thread_free`)
  uint8_t               block_size_shift;  // if not zero, then `(1 << block_size_shift) == block_size` (only used for fast path in `free.c:_palloc_page_ptr_unalign`)
  uint8_t               heap_tag;          // tag of the owning heap, used to separate heaps by object type
                                           // padding
  size_t                block_size;        // size available in each block (always `>0`)
  uint8_t*              page_start;        // start of the page area containing the blocks

  #if (PALLOC_ENCODE_FREELIST || PALLOC_PADDING)
  uintptr_t             keys[2];           // two random keys to encode the free lists (see `_palloc_block_next`) or padding canary
  #endif

  _Atomic(palloc_thread_free_t) xthread_free;  // list of deferred free blocks freed by other threads
  _Atomic(uintptr_t)        xheap;

  struct palloc_page_s*     next;              // next page owned by this thread with the same `block_size`
  struct palloc_page_s*     prev;              // previous page owned by this thread with the same `block_size`

  // 64-bit 11 words, 32-bit 13 words, (+2 for secure)
  void* padding[1];
} palloc_page_t;



// ------------------------------------------------------
// Mimalloc segments contain palloc pages
// ------------------------------------------------------

typedef enum palloc_page_kind_e {
  PALLOC_PAGE_SMALL,    // small blocks go into 64KiB pages inside a segment
  PALLOC_PAGE_MEDIUM,   // medium blocks go into 512KiB pages inside a segment
  PALLOC_PAGE_LARGE,    // larger blocks go into a single page spanning a whole segment
  PALLOC_PAGE_HUGE      // a huge page is a single page in a segment of variable size
                    // used for blocks `> PALLOC_LARGE_OBJ_SIZE_MAX` or an aligment `> PALLOC_BLOCK_ALIGNMENT_MAX`.
} palloc_page_kind_t;

typedef enum palloc_segment_kind_e {
  PALLOC_SEGMENT_NORMAL, // PALLOC_SEGMENT_SIZE size with pages inside.
  PALLOC_SEGMENT_HUGE,   // segment with just one huge page inside.
} palloc_segment_kind_t;

// ------------------------------------------------------
// A segment holds a commit mask where a bit is set if
// the corresponding PALLOC_COMMIT_SIZE area is committed.
// The PALLOC_COMMIT_SIZE must be a multiple of the slice
// size. If it is equal we have the most fine grained
// decommit (but setting it higher can be more efficient).
// The PALLOC_MINIMAL_COMMIT_SIZE is the minimal amount that will
// be committed in one go which can be set higher than
// PALLOC_COMMIT_SIZE for efficiency (while the decommit mask
// is still tracked in fine-grained PALLOC_COMMIT_SIZE chunks)
// ------------------------------------------------------

#define PALLOC_MINIMAL_COMMIT_SIZE      (1*PALLOC_SEGMENT_SLICE_SIZE)
#define PALLOC_COMMIT_SIZE              (PALLOC_SEGMENT_SLICE_SIZE)              // 64KiB
#define PALLOC_COMMIT_MASK_BITS         (PALLOC_SEGMENT_SIZE / PALLOC_COMMIT_SIZE)
#define PALLOC_COMMIT_MASK_FIELD_BITS    PALLOC_SIZE_BITS
#define PALLOC_COMMIT_MASK_FIELD_COUNT  (PALLOC_COMMIT_MASK_BITS / PALLOC_COMMIT_MASK_FIELD_BITS)

#if (PALLOC_COMMIT_MASK_BITS != (PALLOC_COMMIT_MASK_FIELD_COUNT * PALLOC_COMMIT_MASK_FIELD_BITS))
#error "the segment size must be exactly divisible by the (commit size * size_t bits)"
#endif

typedef struct palloc_commit_mask_s {
  size_t mask[PALLOC_COMMIT_MASK_FIELD_COUNT];
} palloc_commit_mask_t;

typedef palloc_page_t  palloc_slice_t;
typedef int64_t    palloc_msecs_t;


// ---------------------------------------------------------------
// a memory id tracks the provenance of arena/OS allocated memory
// ---------------------------------------------------------------

// Memory can reside in arena's, direct OS allocated, or statically allocated. The memid keeps track of this.
typedef enum palloc_memkind_e {
  PALLOC_MEM_NONE,      // not allocated
  PALLOC_MEM_EXTERNAL,  // not owned by palloc but provided externally (via `palloc_manage_os_memory` for example)
  PALLOC_MEM_STATIC,    // allocated in a static area and should not be freed (for arena meta data for example)
  PALLOC_MEM_OS,        // allocated from the OS
  PALLOC_MEM_OS_HUGE,   // allocated as huge OS pages (usually 1GiB, pinned to physical memory)
  PALLOC_MEM_OS_REMAP,  // allocated in a remapable area (i.e. using `mremap`)
  PALLOC_MEM_ARENA      // allocated from an arena (the usual case)
} palloc_memkind_t;

static inline bool palloc_memkind_is_os(palloc_memkind_t memkind) {
  return (memkind >= PALLOC_MEM_OS && memkind <= PALLOC_MEM_OS_REMAP);
}

typedef struct palloc_memid_os_info {
  void*         base;               // actual base address of the block (used for offset aligned allocations)
  size_t        size;               // full allocation size
} palloc_memid_os_info_t;

typedef struct palloc_memid_arena_info {
  size_t        block_index;        // index in the arena
  palloc_arena_id_t id;                 // arena id (>= 1)
  bool          is_exclusive;       // this arena can only be used for specific arena allocations
} palloc_memid_arena_info_t;

typedef struct palloc_memid_s {
  union {
    palloc_memid_os_info_t    os;       // only used for PALLOC_MEM_OS
    palloc_memid_arena_info_t arena;    // only used for PALLOC_MEM_ARENA
  } mem;
  bool          is_pinned;          // `true` if we cannot decommit/reset/protect in this memory (e.g. when allocated using large (2Mib) or huge (1GiB) OS pages)
  bool          initially_committed;// `true` if the memory was originally allocated as committed
  bool          initially_zero;     // `true` if the memory was originally zero initialized
  palloc_memkind_t  memkind;
} palloc_memid_t;


// -----------------------------------------------------------------------------------------
// Segments are large allocated memory blocks (32mb on 64 bit) from arenas or the OS.
//
// Inside segments we allocated fixed size palloc pages (`palloc_page_t`) that contain blocks.
// The start of a segment is this structure with a fixed number of slice entries (`slices`)
// usually followed by a guard OS page and the actual allocation area with pages.
// While a page is not allocated, we view it's data as a `palloc_slice_t` (instead of a `palloc_page_t`).
// Of any free area, the first slice has the info and `slice_offset == 0`; for any subsequent
// slices part of the area, the `slice_offset` is the byte offset back to the first slice
// (so we can quickly find the page info on a free, `internal.h:_palloc_segment_page_of`).
// For slices, the `block_size` field is repurposed to signify if a slice is used (`1`) or not (`0`).
// Small and medium pages use a fixed amount of slices to reduce slice fragmentation, while
// large and huge pages span a variable amount of slices.

typedef struct palloc_subproc_s palloc_subproc_t;

typedef struct palloc_segment_s {
  // constant fields
  palloc_memid_t        memid;              // memory id for arena/OS allocation
  bool              allow_decommit;     // can we decommmit the memory
  bool              allow_purge;        // can we purge the memory (reset or decommit)
  size_t            segment_size;
  palloc_subproc_t*     subproc;            // segment belongs to sub process

  // segment fields
  palloc_msecs_t        purge_expire;       // purge slices in the `purge_mask` after this time
  palloc_commit_mask_t  purge_mask;         // slices that can be purged
  palloc_commit_mask_t  commit_mask;        // slices that are currently committed

  // from here is zero initialized
  struct palloc_segment_s* next;            // the list of freed segments in the cache (must be first field, see `segment.c:palloc_segment_init`)
  bool              was_reclaimed;      // true if it was reclaimed (used to limit on-free reclamation)
  bool              dont_free;          // can be temporarily true to ensure the segment is not freed
  bool              free_is_zero;       // if free spans are zero

  size_t            abandoned;          // abandoned pages (i.e. the original owning thread stopped) (`abandoned <= used`)
  size_t            abandoned_visits;   // count how often this segment is visited during abondoned reclamation (to force reclaim if it takes too long)
  size_t            used;               // count of pages in use
  uintptr_t         cookie;             // verify addresses in debug mode: `palloc_ptr_cookie(segment) == segment->cookie`

  struct palloc_segment_s* abandoned_os_next; // only used for abandoned segments outside arena's, and only if `palloc_option_visit_abandoned` is enabled
  struct palloc_segment_s* abandoned_os_prev;

  size_t            segment_slices;      // for huge segments this may be different from `PALLOC_SLICES_PER_SEGMENT`
  size_t            segment_info_slices; // initial count of slices that we are using for segment info and possible guard pages.

  // layout like this to optimize access in `palloc_free`
  palloc_segment_kind_t kind;
  size_t            slice_entries;       // entries in the `slices` array, at most `PALLOC_SLICES_PER_SEGMENT`
  _Atomic(palloc_threadid_t) thread_id;      // unique id of the thread owning this segment

  palloc_slice_t        slices[PALLOC_SLICES_PER_SEGMENT+1];  // one extra final entry for huge blocks with large alignment
} palloc_segment_t;


// ------------------------------------------------------
// Heaps
// Provide first-class heaps to allocate from.
// A heap just owns a set of pages for allocation and
// can only be allocate/reallocate from the thread that created it.
// Freeing blocks can be done from any thread though.
// Per thread, the segments are shared among its heaps.
// Per thread, there is always a default heap that is
// used for allocation; it is initialized to statically
// point to an empty heap to avoid initialization checks
// in the fast path.
// ------------------------------------------------------

// Thread local data
typedef struct palloc_tld_s palloc_tld_t;

// Pages of a certain block size are held in a queue.
typedef struct palloc_page_queue_s {
  palloc_page_t* first;
  palloc_page_t* last;
  size_t     block_size;
} palloc_page_queue_t;

#define PALLOC_BIN_FULL  (PALLOC_BIN_HUGE+1)

// Random context
typedef struct palloc_random_cxt_s {
  uint32_t input[16];
  uint32_t output[16];
  int      output_available;
  bool     weak;
} palloc_random_ctx_t;


// In debug mode there is a padding structure at the end of the blocks to check for buffer overflows
#if (PALLOC_PADDING)
typedef struct palloc_padding_s {
  uint32_t canary; // encoded block value to check validity of the padding (in case of overflow)
  uint32_t delta;  // padding bytes before the block. (palloc_usable_size(p) - delta == exact allocated bytes)
} palloc_padding_t;
#define PALLOC_PADDING_SIZE   (sizeof(palloc_padding_t))
#define PALLOC_PADDING_WSIZE  ((PALLOC_PADDING_SIZE + PALLOC_INTPTR_SIZE - 1) / PALLOC_INTPTR_SIZE)
#else
#define PALLOC_PADDING_SIZE   0
#define PALLOC_PADDING_WSIZE  0
#endif

#define PALLOC_PAGES_DIRECT   (PALLOC_SMALL_WSIZE_MAX + PALLOC_PADDING_WSIZE + 1)


// A heap owns a set of pages.
struct palloc_heap_s {
  palloc_tld_t*             tld;
  _Atomic(palloc_block_t*)  thread_delayed_free;
  palloc_threadid_t         thread_id;                           // thread this heap belongs too
  palloc_arena_id_t         arena_id;                            // arena id if the heap belongs to a specific arena (or 0)
  uintptr_t             cookie;                              // random cookie to verify pointers (see `_palloc_ptr_cookie`)
  uintptr_t             keys[2];                             // two random keys used to encode the `thread_delayed_free` list
  palloc_random_ctx_t       random;                              // random number context used for secure allocation
  size_t                page_count;                          // total number of pages in the `pages` queues.
  size_t                page_retired_min;                    // smallest retired index (retired pages are fully free, but still in the page queues)
  size_t                page_retired_max;                    // largest retired index into the `pages` array.
  long                  generic_count;                       // how often is `_palloc_malloc_generic` called?
  long                  generic_collect_count;               // how often is `_palloc_malloc_generic` called without collecting?
  palloc_heap_t*            next;                                // list of heaps per thread
  bool                  no_reclaim;                          // `true` if this heap should not reclaim abandoned pages
  uint8_t               tag;                                 // custom tag, can be used for separating heaps based on the object types
  #if PALLOC_GUARDED
  size_t                guarded_size_min;                    // minimal size for guarded objects
  size_t                guarded_size_max;                    // maximal size for guarded objects
  size_t                guarded_sample_rate;                 // sample rate (set to 0 to disable guarded pages)
  size_t                guarded_sample_count;                // current sample count (counting down to 0)
  #endif
  palloc_page_t*            pages_free_direct[PALLOC_PAGES_DIRECT];  // optimize: array where every entry points a page with possibly free blocks in the corresponding queue for that size.
  palloc_page_queue_t       pages[PALLOC_BIN_FULL + 1];              // queue of pages for each size class (or "bin")
};


// ------------------------------------------------------
// Sub processes do not reclaim or visit segments
// from other sub processes. These are essentially the
// static variables of a process.
// ------------------------------------------------------

struct palloc_subproc_s {
  _Atomic(size_t)    abandoned_count;         // count of abandoned segments for this sub-process
  _Atomic(size_t)    abandoned_os_list_count; // count of abandoned segments in the os-list
  palloc_lock_t          abandoned_os_lock;       // lock for the abandoned os segment list (outside of arena's) (this lock protect list operations)
  palloc_lock_t          abandoned_os_visit_lock; // ensure only one thread per subproc visits the abandoned os list
  palloc_segment_t*      abandoned_os_list;       // doubly-linked list of abandoned segments outside of arena's (in OS allocated memory)
  palloc_segment_t*      abandoned_os_list_tail;  // the tail-end of the list
  palloc_memid_t         memid;                   // provenance of this memory block
};


// ------------------------------------------------------
// Thread Local data
// ------------------------------------------------------

// A "span" is is an available range of slices. The span queues keep
// track of slice spans of at most the given `slice_count` (but more than the previous size class).
typedef struct palloc_span_queue_s {
  palloc_slice_t* first;
  palloc_slice_t* last;
  size_t      slice_count;
} palloc_span_queue_t;

#define PALLOC_SEGMENT_BIN_MAX (35)     // 35 == palloc_segment_bin(PALLOC_SLICES_PER_SEGMENT)

// Segments thread local data
typedef struct palloc_segments_tld_s {
  palloc_span_queue_t     spans[PALLOC_SEGMENT_BIN_MAX+1];  // free slice spans inside segments
  size_t              count;        // current number of segments;
  size_t              peak_count;   // peak number of segments
  size_t              current_size; // current size of all segments
  size_t              peak_size;    // peak size of all segments
  size_t              reclaim_count;// number of reclaimed (abandoned) segments
  palloc_subproc_t*       subproc;      // sub-process this thread belongs to.
  palloc_stats_t*         stats;        // points to tld stats
} palloc_segments_tld_t;

// Thread local data
struct palloc_tld_s {
  unsigned long long  heartbeat;     // monotonic heartbeat count
  bool                recurse;       // true if deferred was called; used to prevent infinite recursion.
  palloc_heap_t*          heap_backing;  // backing heap of this thread (cannot be deleted)
  palloc_heap_t*          heaps;         // list of heaps in this thread (so we can abandon all when the thread terminates)
  palloc_segments_tld_t   segments;      // segment tld
  palloc_stats_t          stats;         // statistics
};


// ------------------------------------------------------
// Debug
// ------------------------------------------------------

#if !defined(PALLOC_DEBUG_UNINIT)
#define PALLOC_DEBUG_UNINIT     (0xD0)
#endif
#if !defined(PALLOC_DEBUG_FREED)
#define PALLOC_DEBUG_FREED      (0xDF)
#endif
#if !defined(PALLOC_DEBUG_PADDING)
#define PALLOC_DEBUG_PADDING    (0xDE)
#endif


// ------------------------------------------------------
// Statistics
// ------------------------------------------------------
#ifndef PALLOC_STAT
#if (PALLOC_DEBUG>0)
#define PALLOC_STAT 2
#else
#define PALLOC_STAT 0
#endif
#endif

// add to stat keeping track of the peak
void _palloc_stat_increase(palloc_stat_count_t* stat, size_t amount);
void _palloc_stat_decrease(palloc_stat_count_t* stat, size_t amount);
void _palloc_stat_adjust_decrease(palloc_stat_count_t* stat, size_t amount);
// counters can just be increased
void _palloc_stat_counter_increase(palloc_stat_counter_t* stat, size_t amount);

#if (PALLOC_STAT)
#define palloc_stat_increase(stat,amount)         _palloc_stat_increase( &(stat), amount)
#define palloc_stat_decrease(stat,amount)         _palloc_stat_decrease( &(stat), amount)
#define palloc_stat_adjust_decrease(stat,amount)  _palloc_stat_adjust_decrease( &(stat), amount)
#define palloc_stat_counter_increase(stat,amount) _palloc_stat_counter_increase( &(stat), amount)
#else
#define palloc_stat_increase(stat,amount)         ((void)0)
#define palloc_stat_decrease(stat,amount)         ((void)0)
#define palloc_stat_adjust_decrease(stat,amount)  ((void)0)
#define palloc_stat_counter_increase(stat,amount) ((void)0)
#endif

#define palloc_heap_stat_counter_increase(heap,stat,amount)  palloc_stat_counter_increase( (heap)->tld->stats.stat, amount)
#define palloc_heap_stat_increase(heap,stat,amount)  palloc_stat_increase( (heap)->tld->stats.stat, amount)
#define palloc_heap_stat_decrease(heap,stat,amount)  palloc_stat_decrease( (heap)->tld->stats.stat, amount)
#define palloc_heap_stat_adjust_decrease(heap,stat,amount)  palloc_stat_adjust_decrease( (heap)->tld->stats.stat, amount)

#endif
