/* ----------------------------------------------------------------------------
Copyright (c) 2018-2021, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"
#include "palloc/prim.h"

#include <string.h> // memset

#if defined(_MSC_VER) && (_MSC_VER < 1920)
#pragma warning(disable:4204)  // non-constant aggregate initializer
#endif

/* -----------------------------------------------------------
  Statistics operations
----------------------------------------------------------- */

static bool palloc_is_in_main(void* stat) {
  return ((uint8_t*)stat >= (uint8_t*)&_palloc_stats_main
         && (uint8_t*)stat < ((uint8_t*)&_palloc_stats_main + sizeof(palloc_stats_t)));
}

static void palloc_stat_update(palloc_stat_count_t* stat, int64_t amount) {
  if (amount == 0) return;
  if palloc_unlikely(palloc_is_in_main(stat))
  {
    // add atomically (for abandoned pages)
    int64_t current = palloc_atomic_addi64_relaxed(&stat->current, amount);
    // if (stat == &_palloc_stats_main.committed) { palloc_assert_internal(current + amount >= 0); };
    palloc_atomic_maxi64_relaxed(&stat->peak, current + amount);
    if (amount > 0) {
      palloc_atomic_addi64_relaxed(&stat->total,amount);
    }
  }
  else {
    // add thread local
    stat->current += amount;
    if (stat->current > stat->peak) { stat->peak = stat->current; }
    if (amount > 0) { stat->total += amount; }
  }
}

void _palloc_stat_counter_increase(palloc_stat_counter_t* stat, size_t amount) {
  if (palloc_is_in_main(stat)) {
    palloc_atomic_addi64_relaxed( &stat->total, (int64_t)amount );
  }
  else {
    stat->total += amount;
  }
}

void _palloc_stat_increase(palloc_stat_count_t* stat, size_t amount) {
  palloc_stat_update(stat, (int64_t)amount);
}

void _palloc_stat_decrease(palloc_stat_count_t* stat, size_t amount) {
  palloc_stat_update(stat, -((int64_t)amount));
}


static void palloc_stat_adjust(palloc_stat_count_t* stat, int64_t amount) {
  if (amount == 0) return;
  if palloc_unlikely(palloc_is_in_main(stat))
  {
    // adjust atomically 
    palloc_atomic_addi64_relaxed(&stat->current, amount);
    palloc_atomic_addi64_relaxed(&stat->total,amount);
  }
  else {
    // adjust local
    stat->current += amount;
    stat->total += amount;
  }
}

void _palloc_stat_adjust_decrease(palloc_stat_count_t* stat, size_t amount) {
  palloc_stat_adjust(stat, -((int64_t)amount));
}


// must be thread safe as it is called from stats_merge
static void palloc_stat_count_add_mt(palloc_stat_count_t* stat, const palloc_stat_count_t* src) {
  if (stat==src) return;
  palloc_atomic_void_addi64_relaxed(&stat->total, &src->total); 
  const int64_t prev_current = palloc_atomic_addi64_relaxed(&stat->current, src->current);

  // Global current plus thread peak approximates new global peak
  // note: peak scores do really not work across threads.
  // we used to just add them together but that often overestimates in practice.
  // similarly, max does not seem to work well. The current approach
  // by Artem Kharytoniuk (@artem-lunarg) seems to work better, see PR#1112 
  // for a longer description.
  palloc_atomic_maxi64_relaxed(&stat->peak, prev_current + src->peak);
}

static void palloc_stat_counter_add_mt(palloc_stat_counter_t* stat, const palloc_stat_counter_t* src) {
  if (stat==src) return;
  palloc_atomic_void_addi64_relaxed(&stat->total, &src->total);
}

#define PALLOC_STAT_COUNT(stat)    palloc_stat_count_add_mt(&stats->stat, &src->stat);
#define PALLOC_STAT_COUNTER(stat)  palloc_stat_counter_add_mt(&stats->stat, &src->stat);

// must be thread safe as it is called from stats_merge
static void palloc_stats_add(palloc_stats_t* stats, const palloc_stats_t* src) {
  if (stats==src) return;

  // copy all fields
  PALLOC_STAT_FIELDS()

  #if PALLOC_STAT>1
  for (size_t i = 0; i <= PALLOC_BIN_HUGE; i++) {
    palloc_stat_count_add_mt(&stats->malloc_bins[i], &src->malloc_bins[i]);
  }
  #endif
  for (size_t i = 0; i <= PALLOC_BIN_HUGE; i++) {
    palloc_stat_count_add_mt(&stats->page_bins[i], &src->page_bins[i]);
  }
}

#undef PALLOC_STAT_COUNT
#undef PALLOC_STAT_COUNTER

/* -----------------------------------------------------------
  Display statistics
----------------------------------------------------------- */

// unit > 0 : size in binary bytes
// unit == 0: count as decimal
// unit < 0 : count in binary
static void palloc_printf_amount(int64_t n, int64_t unit, palloc_output_fun* out, void* arg, const char* fmt) {
  char buf[32]; buf[0] = 0;
  int  len = 32;
  const char* suffix = (unit <= 0 ? " " : "B");
  const int64_t base = (unit == 0 ? 1000 : 1024);
  if (unit>0) n *= unit;

  const int64_t pos = (n < 0 ? -n : n);
  if (pos < base) {
    if (n!=1 || suffix[0] != 'B') {  // skip printing 1 B for the unit column
      _palloc_snprintf(buf, len, "%lld   %-3s", (long long)n, (n==0 ? "" : suffix));
    }
  }
  else {
    int64_t divider = base;
    const char* magnitude = "K";
    if (pos >= divider*base) { divider *= base; magnitude = "M"; }
    if (pos >= divider*base) { divider *= base; magnitude = "G"; }
    const int64_t tens = (n / (divider/10));
    const long whole = (long)(tens/10);
    const long frac1 = (long)(tens%10);
    char unitdesc[8];
    _palloc_snprintf(unitdesc, 8, "%s%s%s", magnitude, (base==1024 ? "i" : ""), suffix);
    _palloc_snprintf(buf, len, "%ld.%ld %-3s", whole, (frac1 < 0 ? -frac1 : frac1), unitdesc);
  }
  _palloc_fprintf(out, arg, (fmt==NULL ? "%12s" : fmt), buf);
}


static void palloc_print_amount(int64_t n, int64_t unit, palloc_output_fun* out, void* arg) {
  palloc_printf_amount(n,unit,out,arg,NULL);
}

static void palloc_print_count(int64_t n, int64_t unit, palloc_output_fun* out, void* arg) {
  if (unit==1) _palloc_fprintf(out, arg, "%12s"," ");
          else palloc_print_amount(n,0,out,arg);
}

static void palloc_stat_print_ex(const palloc_stat_count_t* stat, const char* msg, int64_t unit, palloc_output_fun* out, void* arg, const char* notok ) {
  _palloc_fprintf(out, arg,"%10s:", msg);
  if (unit != 0) {
    if (unit > 0) {
      palloc_print_amount(stat->peak, unit, out, arg);
      palloc_print_amount(stat->total, unit, out, arg);
      // palloc_print_amount(stat->freed, unit, out, arg);
      palloc_print_amount(stat->current, unit, out, arg);
      palloc_print_amount(unit, 1, out, arg);
      palloc_print_count(stat->total, unit, out, arg);
    }
    else {
      palloc_print_amount(stat->peak, -1, out, arg);
      palloc_print_amount(stat->total, -1, out, arg);
      // palloc_print_amount(stat->freed, -1, out, arg);
      palloc_print_amount(stat->current, -1, out, arg);
      if (unit == -1) {
        _palloc_fprintf(out, arg, "%24s", "");
      }
      else {
        palloc_print_amount(-unit, 1, out, arg);
        palloc_print_count((stat->total / -unit), 0, out, arg);
      }
    }
    if (stat->current != 0) {
      _palloc_fprintf(out, arg, "  ");
      _palloc_fprintf(out, arg, (notok == NULL ? "not all freed" : notok));
      _palloc_fprintf(out, arg, "\n");
    }
    else {
      _palloc_fprintf(out, arg, "  ok\n");
    }
  }
  else {
    palloc_print_amount(stat->peak, 1, out, arg);
    palloc_print_amount(stat->total, 1, out, arg);
    _palloc_fprintf(out, arg, "%11s", " ");  // no freed
    palloc_print_amount(stat->current, 1, out, arg);
    _palloc_fprintf(out, arg, "\n");
  }
}

static void palloc_stat_print(const palloc_stat_count_t* stat, const char* msg, int64_t unit, palloc_output_fun* out, void* arg) {
  palloc_stat_print_ex(stat, msg, unit, out, arg, NULL);
}

#if PALLOC_STAT>1
static void palloc_stat_total_print(const palloc_stat_count_t* stat, const char* msg, int64_t unit, palloc_output_fun* out, void* arg) {
  _palloc_fprintf(out, arg, "%10s:", msg);
  _palloc_fprintf(out, arg, "%12s", " ");  // no peak
  palloc_print_amount(stat->total, unit, out, arg);
  _palloc_fprintf(out, arg, "\n");
}
#endif

static void palloc_stat_counter_print(const palloc_stat_counter_t* stat, const char* msg, palloc_output_fun* out, void* arg ) {
  _palloc_fprintf(out, arg, "%10s:", msg);
  palloc_print_amount(stat->total, -1, out, arg);
  _palloc_fprintf(out, arg, "\n");
}


static void palloc_stat_average_print(size_t count, size_t total, const char* msg, palloc_output_fun* out, void* arg) {
  const int64_t avg_tens = (count == 0 ? 0 : (total*10 / count));
  const long avg_whole = (long)(avg_tens/10);
  const long avg_frac1 = (long)(avg_tens%10);
  _palloc_fprintf(out, arg, "%10s: %5ld.%ld avg\n", msg, avg_whole, avg_frac1);
}


static void palloc_print_header(palloc_output_fun* out, void* arg ) {
  _palloc_fprintf(out, arg, "%10s: %11s %11s %11s %11s %11s\n", "heap stats", "peak   ", "total   ", "current   ", "block   ", "total#   ");
}

#if PALLOC_STAT>1
static void palloc_stats_print_bins(const palloc_stat_count_t* bins, size_t max, const char* fmt, palloc_output_fun* out, void* arg) {
  bool found = false;
  char buf[64];
  for (size_t i = 0; i <= max; i++) {
    if (bins[i].total > 0) {
      found = true;
      int64_t unit = _palloc_bin_size((uint8_t)i);
      _palloc_snprintf(buf, 64, "%s %3lu", fmt, (long)i);
      palloc_stat_print(&bins[i], buf, unit, out, arg);
    }
  }
  if (found) {
    _palloc_fprintf(out, arg, "\n");
    palloc_print_header(out, arg);
  }
}
#endif



//------------------------------------------------------------
// Use an output wrapper for line-buffered output
// (which is nice when using loggers etc.)
//------------------------------------------------------------
typedef struct buffered_s {
  palloc_output_fun* out;   // original output function
  void*          arg;   // and state
  char*          buf;   // local buffer of at least size `count+1`
  size_t         used;  // currently used chars `used <= count`
  size_t         count; // total chars available for output
} buffered_t;

static void palloc_buffered_flush(buffered_t* buf) {
  buf->buf[buf->used] = 0;
  _palloc_fputs(buf->out, buf->arg, NULL, buf->buf);
  buf->used = 0;
}

static void palloc_cdecl palloc_buffered_out(const char* msg, void* arg) {
  buffered_t* buf = (buffered_t*)arg;
  if (msg==NULL || buf==NULL) return;
  for (const char* src = msg; *src != 0; src++) {
    char c = *src;
    if (buf->used >= buf->count) palloc_buffered_flush(buf);
    palloc_assert_internal(buf->used < buf->count);
    buf->buf[buf->used++] = c;
    if (c == '\n') palloc_buffered_flush(buf);
  }
}

//------------------------------------------------------------
// Print statistics
//------------------------------------------------------------

static void _palloc_stats_print(palloc_stats_t* stats, palloc_output_fun* out0, void* arg0) palloc_attr_noexcept {
  // wrap the output function to be line buffered
  char buf[256];
  buffered_t buffer = { out0, arg0, NULL, 0, 255 };
  buffer.buf = buf;
  palloc_output_fun* out = &palloc_buffered_out;
  void* arg = &buffer;

  // and print using that
  palloc_print_header(out,arg);
  #if PALLOC_STAT>1
  palloc_stats_print_bins(stats->malloc_bins, PALLOC_BIN_HUGE, "bin",out,arg);
  #endif
  #if PALLOC_STAT
  palloc_stat_print(&stats->malloc_normal, "binned", (stats->malloc_normal_count.total == 0 ? 1 : -1), out, arg);
  // palloc_stat_print(&stats->malloc_large, "large", (stats->malloc_large_count.total == 0 ? 1 : -1), out, arg);
  palloc_stat_print(&stats->malloc_huge, "huge", (stats->malloc_huge_count.total == 0 ? 1 : -1), out, arg);
  palloc_stat_count_t total = { 0,0,0 };
  palloc_stat_count_add_mt(&total, &stats->malloc_normal);
  // palloc_stat_count_add(&total, &stats->malloc_large);
  palloc_stat_count_add_mt(&total, &stats->malloc_huge);
  palloc_stat_print_ex(&total, "total", 1, out, arg, "");
  #endif
  #if PALLOC_STAT>1
  palloc_stat_total_print(&stats->malloc_requested, "malloc req", 1, out, arg);
  _palloc_fprintf(out, arg, "\n");
  #endif
  palloc_stat_print_ex(&stats->reserved, "reserved", 1, out, arg, "");
  palloc_stat_print_ex(&stats->committed, "committed", 1, out, arg, "");
  palloc_stat_counter_print(&stats->reset, "reset", out, arg );
  palloc_stat_counter_print(&stats->purged, "purged", out, arg );
  palloc_stat_print_ex(&stats->page_committed, "touched", 1, out, arg, "");
  palloc_stat_print(&stats->segments, "segments", -1, out, arg);
  palloc_stat_print(&stats->segments_abandoned, "-abandoned", -1, out, arg);
  palloc_stat_print(&stats->segments_cache, "-cached", -1, out, arg);
  palloc_stat_print(&stats->pages, "pages", -1, out, arg);
  palloc_stat_print(&stats->pages_abandoned, "-abandoned", -1, out, arg);
  palloc_stat_counter_print(&stats->pages_extended, "-extended", out, arg);
  palloc_stat_counter_print(&stats->pages_retire, "-retire", out, arg);
  palloc_stat_counter_print(&stats->arena_count, "arenas", out, arg);
  // palloc_stat_counter_print(&stats->arena_crossover_count, "-crossover", out, arg);
  palloc_stat_counter_print(&stats->arena_rollback_count, "-rollback", out, arg);
  palloc_stat_counter_print(&stats->mmap_calls, "mmaps", out, arg);
  palloc_stat_counter_print(&stats->commit_calls, "commits", out, arg);
  palloc_stat_counter_print(&stats->reset_calls, "resets", out, arg);
  palloc_stat_counter_print(&stats->purge_calls, "purges", out, arg);
  palloc_stat_counter_print(&stats->malloc_guarded_count, "guarded", out, arg);
  palloc_stat_print(&stats->threads, "threads", -1, out, arg);
  palloc_stat_average_print(stats->page_searches_count.total, stats->page_searches.total, "searches", out, arg);
  _palloc_fprintf(out, arg, "%10s: %5i\n", "numa nodes", _palloc_os_numa_node_count());

  size_t elapsed;
  size_t user_time;
  size_t sys_time;
  size_t current_rss;
  size_t peak_rss;
  size_t current_commit;
  size_t peak_commit;
  size_t page_faults;
  palloc_process_info(&elapsed, &user_time, &sys_time, &current_rss, &peak_rss, &current_commit, &peak_commit, &page_faults);
  _palloc_fprintf(out, arg, "%10s: %5zu.%03zu s\n", "elapsed", elapsed/1000, elapsed%1000);
  _palloc_fprintf(out, arg, "%10s: user: %zu.%03zu s, system: %zu.%03zu s, faults: %zu, peak rss: ", "process",
              user_time/1000, user_time%1000, sys_time/1000, sys_time%1000, page_faults );
  palloc_printf_amount((int64_t)peak_rss, 1, out, arg, "%s");
  if (peak_commit > 0) {
    _palloc_fprintf(out, arg, ", peak commit: ");
    palloc_printf_amount((int64_t)peak_commit, 1, out, arg, "%s");
  }
  _palloc_fprintf(out, arg, "\n");
}

static palloc_msecs_t palloc_process_start; // = 0

static palloc_stats_t* palloc_stats_get_default(void) {
  palloc_heap_t* heap = palloc_heap_get_default();
  return &heap->tld->stats;
}

static void palloc_stats_merge_from(palloc_stats_t* stats) {
  if (stats != &_palloc_stats_main) {
    palloc_stats_add(&_palloc_stats_main, stats);
    memset(stats, 0, sizeof(palloc_stats_t));
  }
}

void palloc_stats_reset(void) palloc_attr_noexcept {
  palloc_stats_t* stats = palloc_stats_get_default();
  if (stats != &_palloc_stats_main) { memset(stats, 0, sizeof(palloc_stats_t)); }
  memset(&_palloc_stats_main, 0, sizeof(palloc_stats_t));
  if (palloc_process_start == 0) { palloc_process_start = _palloc_clock_start(); };
}

void palloc_stats_merge(void) palloc_attr_noexcept {
  palloc_stats_merge_from( palloc_stats_get_default() );
}

void _palloc_stats_merge_thread(palloc_tld_t* tld) {
  palloc_stats_merge_from( &tld->stats );
}

void _palloc_stats_done(palloc_stats_t* stats) {  // called from `palloc_thread_done`
  palloc_stats_merge_from(stats);
}

void palloc_stats_print_out(palloc_output_fun* out, void* arg) palloc_attr_noexcept {
  palloc_stats_merge_from(palloc_stats_get_default());
  _palloc_stats_print(&_palloc_stats_main, out, arg);
}

void palloc_stats_print(void* out) palloc_attr_noexcept {
  // for compatibility there is an `out` parameter (which can be `stdout` or `stderr`)
  palloc_stats_print_out((palloc_output_fun*)out, NULL);
}

void palloc_thread_stats_print_out(palloc_output_fun* out, void* arg) palloc_attr_noexcept {
  _palloc_stats_print(palloc_stats_get_default(), out, arg);
}


// ----------------------------------------------------------------
// Basic timer for convenience; use milli-seconds to avoid doubles
// ----------------------------------------------------------------

static palloc_msecs_t palloc_clock_diff;

palloc_msecs_t _palloc_clock_now(void) {
  return _palloc_prim_clock_now();
}

palloc_msecs_t _palloc_clock_start(void) {
  if (palloc_clock_diff == 0.0) {
    palloc_msecs_t t0 = _palloc_clock_now();
    palloc_clock_diff = _palloc_clock_now() - t0;
  }
  return _palloc_clock_now();
}

palloc_msecs_t _palloc_clock_end(palloc_msecs_t start) {
  palloc_msecs_t end = _palloc_clock_now();
  return (end - start - palloc_clock_diff);
}


// --------------------------------------------------------
// Basic process statistics
// --------------------------------------------------------

palloc_decl_export void palloc_process_info(size_t* elapsed_msecs, size_t* user_msecs, size_t* system_msecs, size_t* current_rss, size_t* peak_rss, size_t* current_commit, size_t* peak_commit, size_t* page_faults) palloc_attr_noexcept
{
  palloc_process_info_t pinfo;
  _palloc_memzero_var(pinfo);
  pinfo.elapsed        = _palloc_clock_end(palloc_process_start);
  pinfo.current_commit = (size_t)(palloc_atomic_loadi64_relaxed((_Atomic(int64_t)*)&_palloc_stats_main.committed.current));
  pinfo.peak_commit    = (size_t)(palloc_atomic_loadi64_relaxed((_Atomic(int64_t)*)&_palloc_stats_main.committed.peak));
  pinfo.current_rss    = pinfo.current_commit;
  pinfo.peak_rss       = pinfo.peak_commit;
  pinfo.utime          = 0;
  pinfo.stime          = 0;
  pinfo.page_faults    = 0;

  _palloc_prim_process_info(&pinfo);

  if (elapsed_msecs!=NULL)  *elapsed_msecs  = (pinfo.elapsed < 0 ? 0 : (pinfo.elapsed < (palloc_msecs_t)PTRDIFF_MAX ? (size_t)pinfo.elapsed : PTRDIFF_MAX));
  if (user_msecs!=NULL)     *user_msecs     = (pinfo.utime < 0 ? 0 : (pinfo.utime < (palloc_msecs_t)PTRDIFF_MAX ? (size_t)pinfo.utime : PTRDIFF_MAX));
  if (system_msecs!=NULL)   *system_msecs   = (pinfo.stime < 0 ? 0 : (pinfo.stime < (palloc_msecs_t)PTRDIFF_MAX ? (size_t)pinfo.stime : PTRDIFF_MAX));
  if (current_rss!=NULL)    *current_rss    = pinfo.current_rss;
  if (peak_rss!=NULL)       *peak_rss       = pinfo.peak_rss;
  if (current_commit!=NULL) *current_commit = pinfo.current_commit;
  if (peak_commit!=NULL)    *peak_commit    = pinfo.peak_commit;
  if (page_faults!=NULL)    *page_faults    = pinfo.page_faults;
}


// --------------------------------------------------------
// Return statistics
// --------------------------------------------------------

bool palloc_stats_get(palloc_stats_t* stats) palloc_attr_noexcept {
  if (stats == NULL || stats->size != sizeof(palloc_stats_t) || stats->version != PALLOC_STAT_VERSION) return false;
  _palloc_memzero(stats,stats->size);
  _palloc_memcpy(stats, &_palloc_stats_main, sizeof(palloc_stats_t));
  return true;
}


// --------------------------------------------------------
// Statics in json format
// --------------------------------------------------------

typedef struct palloc_heap_buf_s {
  char*   buf;
  size_t  size;
  size_t  used;
  bool    can_realloc;
} palloc_heap_buf_t;

static bool palloc_heap_buf_expand(palloc_heap_buf_t* hbuf) {
  if (hbuf==NULL) return false;
  if (hbuf->buf != NULL && hbuf->size>0) {
    hbuf->buf[hbuf->size-1] = 0;
  }
  if (hbuf->size > SIZE_MAX/2 || !hbuf->can_realloc) return false;
  const size_t newsize = (hbuf->size == 0 ? palloc_good_size(12*PALLOC_KiB) : 2*hbuf->size);
  char* const  newbuf  = (char*)palloc_rezalloc(hbuf->buf, newsize);
  if (newbuf == NULL) return false;
  hbuf->buf = newbuf;
  hbuf->size = newsize;
  return true;
}

static void palloc_heap_buf_print(palloc_heap_buf_t* hbuf, const char* msg) {
  if (msg==NULL || hbuf==NULL) return;
  if (hbuf->used + 1 >= hbuf->size && !hbuf->can_realloc) return;
  for (const char* src = msg; *src != 0; src++) {
    char c = *src;
    if (hbuf->used + 1 >= hbuf->size) {
      if (!palloc_heap_buf_expand(hbuf)) return;
    }
    palloc_assert_internal(hbuf->used < hbuf->size);
    hbuf->buf[hbuf->used++] = c;
  }
  palloc_assert_internal(hbuf->used < hbuf->size);
  hbuf->buf[hbuf->used] = 0;
}

static void palloc_heap_buf_print_count_bin(palloc_heap_buf_t* hbuf, const char* prefix, palloc_stat_count_t* stat, size_t bin, bool add_comma) {
  const size_t binsize = _palloc_bin_size(bin);
  const size_t pagesize = (binsize <= PALLOC_SMALL_OBJ_SIZE_MAX ? PALLOC_SMALL_PAGE_SIZE :
                            (binsize <= PALLOC_MEDIUM_OBJ_SIZE_MAX ? PALLOC_MEDIUM_PAGE_SIZE :
                              #if PALLOC_LARGE_PAGE_SIZE
                              (binsize <= PALLOC_LARGE_OBJ_SIZE_MAX ? PALLOC_LARGE_PAGE_SIZE : 0)
                              #else
                              0
                              #endif
                              ));
  char buf[128];
  _palloc_snprintf(buf, 128, "%s{ \"total\": %lld, \"peak\": %lld, \"current\": %lld, \"block_size\": %zu, \"page_size\": %zu }%s\n", prefix, stat->total, stat->peak, stat->current, binsize, pagesize, (add_comma ? "," : ""));
  buf[127] = 0;
  palloc_heap_buf_print(hbuf, buf);
}

static void palloc_heap_buf_print_count(palloc_heap_buf_t* hbuf, const char* prefix, palloc_stat_count_t* stat, bool add_comma) {
  char buf[128];
  _palloc_snprintf(buf, 128, "%s{ \"total\": %lld, \"peak\": %lld, \"current\": %lld }%s\n", prefix, stat->total, stat->peak, stat->current, (add_comma ? "," : ""));
  buf[127] = 0;
  palloc_heap_buf_print(hbuf, buf);
}

static void palloc_heap_buf_print_count_value(palloc_heap_buf_t* hbuf, const char* name, palloc_stat_count_t* stat) {
  char buf[128];
  _palloc_snprintf(buf, 128, "  \"%s\": ", name);
  buf[127] = 0;
  palloc_heap_buf_print(hbuf, buf);
  palloc_heap_buf_print_count(hbuf, "", stat, true);
}

static void palloc_heap_buf_print_value(palloc_heap_buf_t* hbuf, const char* name, int64_t val) {
  char buf[128];
  _palloc_snprintf(buf, 128, "  \"%s\": %lld,\n", name, val);
  buf[127] = 0;
  palloc_heap_buf_print(hbuf, buf);
}

static void palloc_heap_buf_print_size(palloc_heap_buf_t* hbuf, const char* name, size_t val, bool add_comma) {
  char buf[128];
  _palloc_snprintf(buf, 128, "    \"%s\": %zu%s\n", name, val, (add_comma ? "," : ""));
  buf[127] = 0;
  palloc_heap_buf_print(hbuf, buf);
}

static void palloc_heap_buf_print_counter_value(palloc_heap_buf_t* hbuf, const char* name, palloc_stat_counter_t* stat) {
  palloc_heap_buf_print_value(hbuf, name, stat->total);
}

#define PALLOC_STAT_COUNT(stat)    palloc_heap_buf_print_count_value(&hbuf, #stat, &stats->stat);
#define PALLOC_STAT_COUNTER(stat)  palloc_heap_buf_print_counter_value(&hbuf, #stat, &stats->stat);

char* palloc_stats_get_json(size_t output_size, char* output_buf) palloc_attr_noexcept {
  palloc_heap_buf_t hbuf = { NULL, 0, 0, true };
  if (output_size > 0 && output_buf != NULL) {
    _palloc_memzero(output_buf, output_size);
    hbuf.buf = output_buf;
    hbuf.size = output_size;
    hbuf.can_realloc = false;
  }
  else {
    if (!palloc_heap_buf_expand(&hbuf)) return NULL;
  }
  palloc_heap_buf_print(&hbuf, "{\n");
  palloc_heap_buf_print_value(&hbuf, "stat_version", PALLOC_STAT_VERSION);
  palloc_heap_buf_print_value(&hbuf, "palloc_version", PALLOC_MALLOC_VERSION);

  // process info
  palloc_heap_buf_print(&hbuf, "  \"process\": {\n");
  size_t elapsed;
  size_t user_time;
  size_t sys_time;
  size_t current_rss;
  size_t peak_rss;
  size_t current_commit;
  size_t peak_commit;
  size_t page_faults;
  palloc_process_info(&elapsed, &user_time, &sys_time, &current_rss, &peak_rss, &current_commit, &peak_commit, &page_faults);
  palloc_heap_buf_print_size(&hbuf, "elapsed_msecs", elapsed, true);
  palloc_heap_buf_print_size(&hbuf, "user_msecs", user_time, true);
  palloc_heap_buf_print_size(&hbuf, "system_msecs", sys_time, true);
  palloc_heap_buf_print_size(&hbuf, "page_faults", page_faults, true);
  palloc_heap_buf_print_size(&hbuf, "rss_current", current_rss, true);
  palloc_heap_buf_print_size(&hbuf, "rss_peak", peak_rss, true);
  palloc_heap_buf_print_size(&hbuf, "commit_current", current_commit, true);
  palloc_heap_buf_print_size(&hbuf, "commit_peak", peak_commit, false);
  palloc_heap_buf_print(&hbuf, "  },\n");

  // statistics
  palloc_stats_t* stats = &_palloc_stats_main;
  PALLOC_STAT_FIELDS()

  // size bins
  palloc_heap_buf_print(&hbuf, "  \"malloc_bins\": [\n");
  for (size_t i = 0; i <= PALLOC_BIN_HUGE; i++) {
    palloc_heap_buf_print_count_bin(&hbuf, "    ", &stats->malloc_bins[i], i, i!=PALLOC_BIN_HUGE);
  }
  palloc_heap_buf_print(&hbuf, "  ],\n");
  palloc_heap_buf_print(&hbuf, "  \"page_bins\": [\n");
  for (size_t i = 0; i <= PALLOC_BIN_HUGE; i++) {
    palloc_heap_buf_print_count_bin(&hbuf, "    ", &stats->page_bins[i], i, i!=PALLOC_BIN_HUGE);
  }
  palloc_heap_buf_print(&hbuf, "  ]\n");
  palloc_heap_buf_print(&hbuf, "}\n");
  if (hbuf.used >= hbuf.size) {
    // failed
    if (hbuf.can_realloc) { palloc_free(hbuf.buf); }
    return NULL;
  }
  else {
    return hbuf.buf;
  }
}
