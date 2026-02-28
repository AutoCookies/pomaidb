/* ----------------------------------------------------------------------------
Copyright (c) 2018-2025, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_STATS_H
#define PALLOC_STATS_H

#include <palloc.h>
#include <stdint.h>

#define PALLOC_STAT_VERSION   4  // increased on every backward incompatible change

// count allocation over time
typedef struct palloc_stat_count_s {
  int64_t total;                              // total allocated
  int64_t peak;                               // peak allocation
  int64_t current;                            // current allocation
} palloc_stat_count_t;

// counters only increase
typedef struct palloc_stat_counter_s {
  int64_t total;                              // total count
} palloc_stat_counter_t;

#define PALLOC_STAT_FIELDS() \
  PALLOC_STAT_COUNT(pages)                      /* count of palloc pages */ \
  PALLOC_STAT_COUNT(reserved)                   /* reserved memory bytes */ \
  PALLOC_STAT_COUNT(committed)                  /* committed bytes */ \
  PALLOC_STAT_COUNTER(reset)                    /* reset bytes */ \
  PALLOC_STAT_COUNTER(purged)                   /* purged bytes */ \
  PALLOC_STAT_COUNT(page_committed)             /* committed memory inside pages */ \
  PALLOC_STAT_COUNT(pages_abandoned)            /* abandonded pages count */ \
  PALLOC_STAT_COUNT(threads)                    /* number of threads */ \
  PALLOC_STAT_COUNT(malloc_normal)              /* allocated bytes <= PALLOC_LARGE_OBJ_SIZE_MAX */ \
  PALLOC_STAT_COUNT(malloc_huge)                /* allocated bytes in huge pages */ \
  PALLOC_STAT_COUNT(malloc_requested)           /* malloc requested bytes */ \
  \
  PALLOC_STAT_COUNTER(mmap_calls) \
  PALLOC_STAT_COUNTER(commit_calls) \
  PALLOC_STAT_COUNTER(reset_calls) \
  PALLOC_STAT_COUNTER(purge_calls) \
  PALLOC_STAT_COUNTER(arena_count)              /* number of memory arena's */ \
  PALLOC_STAT_COUNTER(malloc_normal_count)      /* number of blocks <= PALLOC_LARGE_OBJ_SIZE_MAX */ \
  PALLOC_STAT_COUNTER(malloc_huge_count)        /* number of huge bloks */ \
  PALLOC_STAT_COUNTER(malloc_guarded_count)     /* number of allocations with guard pages */ \
  \
  /* internal statistics */ \
  PALLOC_STAT_COUNTER(arena_rollback_count) \
  PALLOC_STAT_COUNTER(arena_purges) \
  PALLOC_STAT_COUNTER(pages_extended)           /* number of page extensions */ \
  PALLOC_STAT_COUNTER(pages_retire)             /* number of pages that are retired */ \
  PALLOC_STAT_COUNTER(page_searches)            /* total pages searched for a fresh page */ \
  PALLOC_STAT_COUNTER(page_searches_count)      /* searched count for a fresh page */ \
  /* only on v1 and v2 */ \
  PALLOC_STAT_COUNT(segments) \
  PALLOC_STAT_COUNT(segments_abandoned) \
  PALLOC_STAT_COUNT(segments_cache) \
  PALLOC_STAT_COUNT(_segments_reserved) \
  /* only on v3 */ \
  PALLOC_STAT_COUNTER(pages_reclaim_on_alloc) \
  PALLOC_STAT_COUNTER(pages_reclaim_on_free) \
  PALLOC_STAT_COUNTER(pages_reabandon_full) \
  PALLOC_STAT_COUNTER(pages_unabandon_busy_wait) \


// Define the statistics structure
#define PALLOC_BIN_HUGE             (73U)   // see types.h
#define PALLOC_STAT_COUNT(stat)     palloc_stat_count_t stat;
#define PALLOC_STAT_COUNTER(stat)   palloc_stat_counter_t stat;

typedef struct palloc_stats_s
{
  size_t size;          // size of the palloc_stats_t structure 
  size_t version;       

  PALLOC_STAT_FIELDS()

  // future extension
  palloc_stat_count_t   _stat_reserved[4];
  palloc_stat_counter_t _stat_counter_reserved[4];

  // size segregated statistics
  palloc_stat_count_t   malloc_bins[PALLOC_BIN_HUGE+1];   // allocation per size bin
  palloc_stat_count_t   page_bins[PALLOC_BIN_HUGE+1];     // pages allocated per size bin
} palloc_stats_t;

#undef PALLOC_STAT_COUNT
#undef PALLOC_STAT_COUNTER

// helper
#define palloc_stats_t_decl(name)  palloc_stats_t name = { 0 }; name.size = sizeof(palloc_stats_t); name.version = PALLOC_STAT_VERSION;

// Exported definitions
#ifdef __cplusplus
extern "C" {
#endif

palloc_decl_export bool  palloc_stats_get( palloc_stats_t* stats ) palloc_attr_noexcept;
palloc_decl_export char* palloc_stats_get_json( size_t buf_size, char* buf ) palloc_attr_noexcept;    // use palloc_free to free the result if the input buf == NULL

#ifdef __cplusplus
}
#endif

#endif // PALLOC_STATS_H
