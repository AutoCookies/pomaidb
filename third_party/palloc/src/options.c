/* ----------------------------------------------------------------------------
Copyright (c) 2018-2021, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"
#include "palloc/prim.h"  // palloc_prim_out_stderr

#include <stdio.h>      // stdin/stdout
#include <stdlib.h>     // abort



static long palloc_max_error_count   = 16; // stop outputting errors after this (use < 0 for no limit)
static long palloc_max_warning_count = 16; // stop outputting warnings after this (use < 0 for no limit)

static void palloc_add_stderr_output(void);

int palloc_version(void) palloc_attr_noexcept {
  return PALLOC_MALLOC_VERSION;
}


// --------------------------------------------------------
// Options
// These can be accessed by multiple threads and may be
// concurrently initialized, but an initializing data race
// is ok since they resolve to the same value.
// --------------------------------------------------------
typedef enum palloc_init_e {
  UNINIT,       // not yet initialized
  DEFAULTED,    // not found in the environment, use default value
  INITIALIZED   // found in environment or set explicitly
} palloc_init_t;

typedef struct palloc_option_desc_s {
  long        value;  // the value
  palloc_init_t   init;   // is it initialized yet? (from the environment)
  palloc_option_t option; // for debugging: the option index should match the option
  const char* name;   // option name without `palloc_` prefix
  const char* legacy_name; // potential legacy option name
} palloc_option_desc_t;

#define PALLOC_OPTION(opt)                  palloc_option_##opt, #opt, NULL
#define PALLOC_OPTION_LEGACY(opt,legacy)    palloc_option_##opt, #opt, #legacy

// Some options can be set at build time for statically linked libraries
// (use `-DMI_EXTRA_CPPDEFS="opt1=val1;opt2=val2"`)
//
// This is useful if we cannot pass them as environment variables
// (and setting them programmatically would be too late)

#ifndef PALLOC_DEFAULT_VERBOSE
#define PALLOC_DEFAULT_VERBOSE 0
#endif

#ifndef PALLOC_DEFAULT_EAGER_COMMIT
#define PALLOC_DEFAULT_EAGER_COMMIT 1
#endif

#ifndef PALLOC_DEFAULT_ARENA_EAGER_COMMIT
#define PALLOC_DEFAULT_ARENA_EAGER_COMMIT 2
#endif

// in KiB
#ifndef PALLOC_DEFAULT_ARENA_RESERVE
 #if (PALLOC_INTPTR_SIZE>4)
  #define PALLOC_DEFAULT_ARENA_RESERVE 1024L*1024L
 #else
  #define PALLOC_DEFAULT_ARENA_RESERVE 128L*1024L
 #endif
#endif

#ifndef PALLOC_DEFAULT_DISALLOW_ARENA_ALLOC
#define PALLOC_DEFAULT_DISALLOW_ARENA_ALLOC 0
#endif

#ifndef PALLOC_DEFAULT_ALLOW_LARGE_OS_PAGES
#define PALLOC_DEFAULT_ALLOW_LARGE_OS_PAGES 0
#endif

#ifndef PALLOC_DEFAULT_RESERVE_HUGE_OS_PAGES
#define PALLOC_DEFAULT_RESERVE_HUGE_OS_PAGES 0
#endif

#ifndef PALLOC_DEFAULT_RESERVE_OS_MEMORY
#define PALLOC_DEFAULT_RESERVE_OS_MEMORY 0
#endif

#ifndef PALLOC_DEFAULT_GUARDED_SAMPLE_RATE
#if PALLOC_GUARDED
#define PALLOC_DEFAULT_GUARDED_SAMPLE_RATE 4000
#else
#define PALLOC_DEFAULT_GUARDED_SAMPLE_RATE 0
#endif
#endif


#ifndef PALLOC_DEFAULT_ALLOW_THP
#if defined(__ANDROID__)
#define PALLOC_DEFAULT_ALLOW_THP  0
#else
#define PALLOC_DEFAULT_ALLOW_THP  1
#endif
#endif

// Static options
static palloc_option_desc_t options[_palloc_option_last] =
{
  // stable options
  #if PALLOC_DEBUG || defined(PALLOC_SHOW_ERRORS)
  { 1, UNINIT, PALLOC_OPTION(show_errors) },
  #else
  { 0, UNINIT, PALLOC_OPTION(show_errors) },
  #endif
  { 0, UNINIT, PALLOC_OPTION(show_stats) },
  { PALLOC_DEFAULT_VERBOSE, UNINIT, PALLOC_OPTION(verbose) },

  // some of the following options are experimental and not all combinations are allowed.
  { PALLOC_DEFAULT_EAGER_COMMIT,
       UNINIT, PALLOC_OPTION(eager_commit) },               // commit per segment directly (4MiB)  (but see also `eager_commit_delay`)
  { PALLOC_DEFAULT_ARENA_EAGER_COMMIT,
       UNINIT, PALLOC_OPTION_LEGACY(arena_eager_commit,eager_region_commit) }, // eager commit arena's? 2 is used to enable this only on an OS that has overcommit (i.e. linux)
  { 1, UNINIT, PALLOC_OPTION_LEGACY(purge_decommits,reset_decommits) },        // purge decommits memory (instead of reset) (note: on linux this uses MADV_DONTNEED for decommit)
  { PALLOC_DEFAULT_ALLOW_LARGE_OS_PAGES,
       UNINIT, PALLOC_OPTION_LEGACY(allow_large_os_pages,large_os_pages) },    // use large OS pages, use only with eager commit to prevent fragmentation of VMA's
  { PALLOC_DEFAULT_RESERVE_HUGE_OS_PAGES,
       UNINIT, PALLOC_OPTION(reserve_huge_os_pages) },      // per 1GiB huge pages
  {-1, UNINIT, PALLOC_OPTION(reserve_huge_os_pages_at) },   // reserve huge pages at node N
  { PALLOC_DEFAULT_RESERVE_OS_MEMORY,
       UNINIT, PALLOC_OPTION(reserve_os_memory)     },      // reserve N KiB OS memory in advance (use `option_get_size`)
  { 0, UNINIT, PALLOC_OPTION(deprecated_segment_cache) },   // cache N segments per thread
  { 0, UNINIT, PALLOC_OPTION(deprecated_page_reset) },      // reset page memory on free
  { 0, UNINIT, PALLOC_OPTION_LEGACY(abandoned_page_purge,abandoned_page_reset) },       // reset free page memory when a thread terminates
  { 0, UNINIT, PALLOC_OPTION(deprecated_segment_reset) },   // reset segment memory on free (needs eager commit)
#if defined(__NetBSD__)
  { 0, UNINIT, PALLOC_OPTION(eager_commit_delay) },         // the first N segments per thread are not eagerly committed
#else
  { 1, UNINIT, PALLOC_OPTION(eager_commit_delay) },         // the first N segments per thread are not eagerly committed (but per page in the segment on demand)
#endif
  { 10,  UNINIT, PALLOC_OPTION_LEGACY(purge_delay,reset_delay) },  // purge delay in milli-seconds
  { 0,   UNINIT, PALLOC_OPTION(use_numa_nodes) },           // 0 = use available numa nodes, otherwise use at most N nodes.
  { 0,   UNINIT, PALLOC_OPTION_LEGACY(disallow_os_alloc,limit_os_alloc) },           // 1 = do not use OS memory for allocation (but only reserved arenas)
  { 100, UNINIT, PALLOC_OPTION(os_tag) },                   // only apple specific for now but might serve more or less related purpose
  { 32,  UNINIT, PALLOC_OPTION(max_errors) },               // maximum errors that are output
  { 32,  UNINIT, PALLOC_OPTION(max_warnings) },             // maximum warnings that are output
  { 10,  UNINIT, PALLOC_OPTION(max_segment_reclaim)},       // max. percentage of the abandoned segments to be reclaimed per try.
  { 0,   UNINIT, PALLOC_OPTION(destroy_on_exit)},           // release all OS memory on process exit; careful with dangling pointer or after-exit frees!
  { PALLOC_DEFAULT_ARENA_RESERVE, UNINIT, PALLOC_OPTION(arena_reserve) }, // reserve memory N KiB at a time (=1GiB) (use `option_get_size`)
  { 10,  UNINIT, PALLOC_OPTION(arena_purge_mult) },         // purge delay multiplier for arena's
  { 1,   UNINIT, PALLOC_OPTION_LEGACY(purge_extend_delay, decommit_extend_delay) },
  { 0,   UNINIT, PALLOC_OPTION(abandoned_reclaim_on_free) },// reclaim an abandoned segment on a free
  { PALLOC_DEFAULT_DISALLOW_ARENA_ALLOC,   UNINIT, PALLOC_OPTION(disallow_arena_alloc) }, // 1 = do not use arena's for allocation (except if using specific arena id's)
  { 400, UNINIT, PALLOC_OPTION(retry_on_oom) },             // windows only: retry on out-of-memory for N milli seconds (=400), set to 0 to disable retries.
#if defined(PALLOC_VISIT_ABANDONED)
  { 1,   INITIALIZED, PALLOC_OPTION(visit_abandoned) },     // allow visiting heap blocks in abandoned segments; requires taking locks during reclaim.
#else
  { 0,   UNINIT, PALLOC_OPTION(visit_abandoned) },
#endif
  { 0,   UNINIT, PALLOC_OPTION(guarded_min) },              // only used when building with PALLOC_GUARDED: minimal rounded object size for guarded objects
  { PALLOC_GiB, UNINIT, PALLOC_OPTION(guarded_max) },           // only used when building with PALLOC_GUARDED: maximal rounded object size for guarded objects
  { 0,   UNINIT, PALLOC_OPTION(guarded_precise) },          // disregard minimal alignment requirement to always place guarded blocks exactly in front of a guard page (=0)
  { PALLOC_DEFAULT_GUARDED_SAMPLE_RATE,
         UNINIT, PALLOC_OPTION(guarded_sample_rate)},       // 1 out of N allocations in the min/max range will be guarded (=4000)
  { 0,   UNINIT, PALLOC_OPTION(guarded_sample_seed)},
  { 0,   UNINIT, PALLOC_OPTION(target_segments_per_thread) }, // abandon segments beyond this point, or 0 to disable.
  { 10000, UNINIT, PALLOC_OPTION(generic_collect) },          // collect heaps every N (=10000) generic allocation calls
  { PALLOC_DEFAULT_ALLOW_THP, 
         UNINIT, PALLOC_OPTION(allow_thp) }                 // allow transparent huge pages?
};

static void palloc_option_init(palloc_option_desc_t* desc);

static bool palloc_option_has_size_in_kib(palloc_option_t option) {
  return (option == palloc_option_reserve_os_memory || option == palloc_option_arena_reserve);
}

void _palloc_options_init(void) {
  // called on process load
  palloc_add_stderr_output(); // now it safe to use stderr for output
  for(int i = 0; i < _palloc_option_last; i++ ) {
    palloc_option_t option = (palloc_option_t)i;
    long l = palloc_option_get(option); PALLOC_UNUSED(l); // initialize
  }
  palloc_max_error_count = palloc_option_get(palloc_option_max_errors);
  palloc_max_warning_count = palloc_option_get(palloc_option_max_warnings);
  #if PALLOC_GUARDED
  if (palloc_option_get(palloc_option_guarded_sample_rate) > 0) {
    if (palloc_option_is_enabled(palloc_option_allow_large_os_pages)) {
      palloc_option_disable(palloc_option_allow_large_os_pages);
      _palloc_warning_message("option 'allow_large_os_pages' is disabled to allow for guarded objects\n");
    }
  }
  #endif
  if (palloc_option_is_enabled(palloc_option_verbose)) { palloc_options_print(); }
}

#define palloc_stringifyx(str)  #str                // and stringify
#define palloc_stringify(str)   palloc_stringifyx(str)  // expand

void palloc_options_print(void) palloc_attr_noexcept
{
  // show version
  const int vermajor = PALLOC_MALLOC_VERSION/100;
  const int verminor = (PALLOC_MALLOC_VERSION%100)/10;
  const int verpatch = (PALLOC_MALLOC_VERSION%10);
  _palloc_message("v%i.%i.%i%s%s (built on %s, %s)\n", vermajor, verminor, verpatch,
      #if defined(PALLOC_CMAKE_BUILD_TYPE)
      ", " palloc_stringify(PALLOC_CMAKE_BUILD_TYPE)
      #else
      ""
      #endif
      ,
      #if defined(PALLOC_GIT_DESCRIBE)
      ", git " palloc_stringify(PALLOC_GIT_DESCRIBE)
      #else
      ""
      #endif
      , __DATE__, __TIME__);

  // show options
  for (int i = 0; i < _palloc_option_last; i++) {
    palloc_option_t option = (palloc_option_t)i;
    long l = palloc_option_get(option); PALLOC_UNUSED(l); // possibly initialize
    palloc_option_desc_t* desc = &options[option];
    _palloc_message("option '%s': %ld %s\n", desc->name, desc->value, (palloc_option_has_size_in_kib(option) ? "KiB" : ""));
  }

  // show build configuration
  _palloc_message("debug level : %d\n", PALLOC_DEBUG );
  _palloc_message("secure level: %d\n", PALLOC_SECURE );
  _palloc_message("mem tracking: %s\n", PALLOC_TRACK_TOOL);
  #if PALLOC_GUARDED
  _palloc_message("guarded build: %s\n", palloc_option_get(palloc_option_guarded_sample_rate) != 0 ? "enabled" : "disabled");
  #endif
  #if PALLOC_TSAN
  _palloc_message("thread santizer enabled\n");
  #endif
}

long _palloc_option_get_fast(palloc_option_t option) {
  palloc_assert(option >= 0 && option < _palloc_option_last);
  palloc_option_desc_t* desc = &options[option];
  palloc_assert(desc->option == option);  // index should match the option
  //palloc_assert(desc->init != UNINIT);
  return desc->value;
}


palloc_decl_nodiscard long palloc_option_get(palloc_option_t option) {
  palloc_assert(option >= 0 && option < _palloc_option_last);
  if (option < 0 || option >= _palloc_option_last) return 0;
  palloc_option_desc_t* desc = &options[option];
  palloc_assert(desc->option == option);  // index should match the option
  if palloc_unlikely(desc->init == UNINIT) {
    palloc_option_init(desc);
  }
  return desc->value;
}

palloc_decl_nodiscard long palloc_option_get_clamp(palloc_option_t option, long min, long max) {
  long x = palloc_option_get(option);
  return (x < min ? min : (x > max ? max : x));
}

palloc_decl_nodiscard size_t palloc_option_get_size(palloc_option_t option) {
  const long x = palloc_option_get(option);
  size_t size = (x < 0 ? 0 : (size_t)x);
  if (palloc_option_has_size_in_kib(option)) {
    size *= PALLOC_KiB;
  }
  return size;
}

void palloc_option_set(palloc_option_t option, long value) {
  palloc_assert(option >= 0 && option < _palloc_option_last);
  if (option < 0 || option >= _palloc_option_last) return;
  palloc_option_desc_t* desc = &options[option];
  palloc_assert(desc->option == option);  // index should match the option
  desc->value = value;
  desc->init = INITIALIZED;
  // ensure min/max range; be careful to not recurse.
  if (desc->option == palloc_option_guarded_min && _palloc_option_get_fast(palloc_option_guarded_max) < value) {
    palloc_option_set(palloc_option_guarded_max, value);
  }
  else if (desc->option == palloc_option_guarded_max && _palloc_option_get_fast(palloc_option_guarded_min) > value) {
    palloc_option_set(palloc_option_guarded_min, value);
  }
}

void palloc_option_set_default(palloc_option_t option, long value) {
  palloc_assert(option >= 0 && option < _palloc_option_last);
  if (option < 0 || option >= _palloc_option_last) return;
  palloc_option_desc_t* desc = &options[option];
  if (desc->init != INITIALIZED) {
    desc->value = value;
  }
}

palloc_decl_nodiscard bool palloc_option_is_enabled(palloc_option_t option) {
  return (palloc_option_get(option) != 0);
}

void palloc_option_set_enabled(palloc_option_t option, bool enable) {
  palloc_option_set(option, (enable ? 1 : 0));
}

void palloc_option_set_enabled_default(palloc_option_t option, bool enable) {
  palloc_option_set_default(option, (enable ? 1 : 0));
}

void palloc_option_enable(palloc_option_t option) {
  palloc_option_set_enabled(option,true);
}

void palloc_option_disable(palloc_option_t option) {
  palloc_option_set_enabled(option,false);
}

static void palloc_cdecl palloc_out_stderr(const char* msg, void* arg) {
  PALLOC_UNUSED(arg);
  if (msg != NULL && msg[0] != 0) {
    _palloc_prim_out_stderr(msg);
  }
}

// Since an output function can be registered earliest in the `main`
// function we also buffer output that happens earlier. When
// an output function is registered it is called immediately with
// the output up to that point.
#ifndef PALLOC_MAX_DELAY_OUTPUT
#define PALLOC_MAX_DELAY_OUTPUT ((size_t)(16*1024))
#endif
static char out_buf[PALLOC_MAX_DELAY_OUTPUT+1];
static _Atomic(size_t) out_len;

static void palloc_cdecl palloc_out_buf(const char* msg, void* arg) {
  PALLOC_UNUSED(arg);
  if (msg==NULL) return;
  if (palloc_atomic_load_relaxed(&out_len)>=PALLOC_MAX_DELAY_OUTPUT) return;
  size_t n = _palloc_strlen(msg);
  if (n==0) return;
  // claim space
  size_t start = palloc_atomic_add_acq_rel(&out_len, n);
  if (start >= PALLOC_MAX_DELAY_OUTPUT) return;
  // check bound
  if (start+n >= PALLOC_MAX_DELAY_OUTPUT) {
    n = PALLOC_MAX_DELAY_OUTPUT-start-1;
  }
  _palloc_memcpy(&out_buf[start], msg, n);
}

static void palloc_out_buf_flush(palloc_output_fun* out, bool no_more_buf, void* arg) {
  if (out==NULL) return;
  // claim (if `no_more_buf == true`, no more output will be added after this point)
  size_t count = palloc_atomic_add_acq_rel(&out_len, (no_more_buf ? PALLOC_MAX_DELAY_OUTPUT : 1));
  // and output the current contents
  if (count>PALLOC_MAX_DELAY_OUTPUT) count = PALLOC_MAX_DELAY_OUTPUT;
  out_buf[count] = 0;
  out(out_buf,arg);
  if (!no_more_buf) {
    out_buf[count] = '\n'; // if continue with the buffer, insert a newline
  }
}


// Once this module is loaded, switch to this routine
// which outputs to stderr and the delayed output buffer.
static void palloc_cdecl palloc_out_buf_stderr(const char* msg, void* arg) {
  palloc_out_stderr(msg,arg);
  palloc_out_buf(msg,arg);
}



// --------------------------------------------------------
// Default output handler
// --------------------------------------------------------

// Should be atomic but gives errors on many platforms as generally we cannot cast a function pointer to a uintptr_t.
// For now, don't register output from multiple threads.
static palloc_output_fun* volatile palloc_out_default; // = NULL
static _Atomic(void*) palloc_out_arg; // = NULL

static palloc_output_fun* palloc_out_get_default(void** parg) {
  if (parg != NULL) { *parg = palloc_atomic_load_ptr_acquire(void,&palloc_out_arg); }
  palloc_output_fun* out = palloc_out_default;
  return (out == NULL ? &palloc_out_buf : out);
}

void palloc_register_output(palloc_output_fun* out, void* arg) palloc_attr_noexcept {
  palloc_out_default = (out == NULL ? &palloc_out_stderr : out); // stop using the delayed output buffer
  palloc_atomic_store_ptr_release(void,&palloc_out_arg, arg);
  if (out!=NULL) palloc_out_buf_flush(out,true,arg);         // output all the delayed output now
}

// add stderr to the delayed output after the module is loaded
static void palloc_add_stderr_output(void) {
  palloc_assert_internal(palloc_out_default == NULL);
  palloc_out_buf_flush(&palloc_out_stderr, false, NULL); // flush current contents to stderr
  palloc_out_default = &palloc_out_buf_stderr;           // and add stderr to the delayed output
}

// --------------------------------------------------------
// Messages, all end up calling `_palloc_fputs`.
// --------------------------------------------------------
static _Atomic(size_t) error_count;   // = 0;  // when >= max_error_count stop emitting errors
static _Atomic(size_t) warning_count; // = 0;  // when >= max_warning_count stop emitting warnings

// When overriding malloc, we may recurse into palloc_vfprintf if an allocation
// inside the C runtime causes another message.
// In some cases (like on macOS) the loader already allocates which
// calls into palloc; if we then access thread locals (like `recurse`)
// this may crash as the access may call _tlv_bootstrap that tries to
// (recursively) invoke malloc again to allocate space for the thread local
// variables on demand. This is why we use a _palloc_preloading test on such
// platforms. However, C code generator may move the initial thread local address
// load before the `if` and we therefore split it out in a separate function.
static palloc_decl_thread bool recurse = false;

static palloc_decl_noinline bool palloc_recurse_enter_prim(void) {
  if (recurse) return false;
  recurse = true;
  return true;
}

static palloc_decl_noinline void palloc_recurse_exit_prim(void) {
  recurse = false;
}

static bool palloc_recurse_enter(void) {
  #if defined(__APPLE__) || defined(__ANDROID__) || defined(PALLOC_TLS_RECURSE_GUARD)
  if (_palloc_preloading()) return false;
  #endif
  return palloc_recurse_enter_prim();
}

static void palloc_recurse_exit(void) {
  #if defined(__APPLE__) || defined(__ANDROID__) || defined(PALLOC_TLS_RECURSE_GUARD)
  if (_palloc_preloading()) return;
  #endif
  palloc_recurse_exit_prim();
}

void _palloc_fputs(palloc_output_fun* out, void* arg, const char* prefix, const char* message) {
  if (out==NULL || (void*)out==(void*)stdout || (void*)out==(void*)stderr) { // TODO: use palloc_out_stderr for stderr?
    if (!palloc_recurse_enter()) return;
    out = palloc_out_get_default(&arg);
    if (prefix != NULL) out(prefix, arg);
    out(message, arg);
    palloc_recurse_exit();
  }
  else {
    if (prefix != NULL) out(prefix, arg);
    out(message, arg);
  }
}

// Define our own limited `fprintf` that avoids memory allocation.
// We do this using `_palloc_vsnprintf` with a limited buffer.
static void palloc_vfprintf( palloc_output_fun* out, void* arg, const char* prefix, const char* fmt, va_list args ) {
  char buf[512];
  if (fmt==NULL) return;
  if (!palloc_recurse_enter()) return;
  _palloc_vsnprintf(buf, sizeof(buf)-1, fmt, args);
  palloc_recurse_exit();
  _palloc_fputs(out,arg,prefix,buf);
}

void _palloc_fprintf( palloc_output_fun* out, void* arg, const char* fmt, ... ) {
  va_list args;
  va_start(args,fmt);
  palloc_vfprintf(out,arg,NULL,fmt,args);
  va_end(args);
}

static void palloc_vfprintf_thread(palloc_output_fun* out, void* arg, const char* prefix, const char* fmt, va_list args) {
  if (prefix != NULL && _palloc_strnlen(prefix,33) <= 32 && !_palloc_is_main_thread()) {
    char tprefix[64];
    _palloc_snprintf(tprefix, sizeof(tprefix), "%sthread 0x%tx: ", prefix, (uintptr_t)_palloc_thread_id());
    palloc_vfprintf(out, arg, tprefix, fmt, args);
  }
  else {
    palloc_vfprintf(out, arg, prefix, fmt, args);
  }
}

void _palloc_message(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  palloc_vfprintf_thread(NULL, NULL, "palloc: ", fmt, args);
  va_end(args);
}

void _palloc_trace_message(const char* fmt, ...) {
  if (palloc_option_get(palloc_option_verbose) <= 1) return;  // only with verbose level 2 or higher
  va_list args;
  va_start(args, fmt);
  palloc_vfprintf_thread(NULL, NULL, "palloc: ", fmt, args);
  va_end(args);
}

void _palloc_verbose_message(const char* fmt, ...) {
  if (!palloc_option_is_enabled(palloc_option_verbose)) return;
  va_list args;
  va_start(args,fmt);
  palloc_vfprintf(NULL, NULL, "palloc: ", fmt, args);
  va_end(args);
}

static void palloc_show_error_message(const char* fmt, va_list args) {
  if (!palloc_option_is_enabled(palloc_option_verbose)) {
    if (!palloc_option_is_enabled(palloc_option_show_errors)) return;
    if (palloc_max_error_count >= 0 && (long)palloc_atomic_increment_acq_rel(&error_count) > palloc_max_error_count) return;
  }
  palloc_vfprintf_thread(NULL, NULL, "palloc: error: ", fmt, args);
}

void _palloc_warning_message(const char* fmt, ...) {
  if (!palloc_option_is_enabled(palloc_option_verbose)) {
    if (!palloc_option_is_enabled(palloc_option_show_errors)) return;
    if (palloc_max_warning_count >= 0 && (long)palloc_atomic_increment_acq_rel(&warning_count) > palloc_max_warning_count) return;
  }
  va_list args;
  va_start(args,fmt);
  palloc_vfprintf_thread(NULL, NULL, "palloc: warning: ", fmt, args);
  va_end(args);
}


#if PALLOC_DEBUG
palloc_decl_noreturn palloc_decl_cold void _palloc_assert_fail(const char* assertion, const char* fname, unsigned line, const char* func ) palloc_attr_noexcept {
  _palloc_fprintf(NULL, NULL, "palloc: assertion failed: at \"%s\":%u, %s\n  assertion: \"%s\"\n", fname, line, (func==NULL?"":func), assertion);
  abort();
}
#endif

// --------------------------------------------------------
// Errors
// --------------------------------------------------------

static palloc_error_fun* volatile  palloc_error_handler; // = NULL
static _Atomic(void*) palloc_error_arg;     // = NULL

static void palloc_error_default(int err) {
  PALLOC_UNUSED(err);
#if (PALLOC_DEBUG>0)
  if (err==EFAULT) {
    #ifdef _MSC_VER
    __debugbreak();
    #endif
    abort();
  }
#endif
#if (PALLOC_SECURE>0)
  if (err==EFAULT) {  // abort on serious errors in secure mode (corrupted meta-data)
    abort();
  }
#endif
#if defined(PALLOC_XMALLOC)
  if (err==ENOMEM || err==EOVERFLOW) { // abort on memory allocation fails in xmalloc mode
    abort();
  }
#endif
}

void palloc_register_error(palloc_error_fun* fun, void* arg) {
  palloc_error_handler = fun;  // can be NULL
  palloc_atomic_store_ptr_release(void,&palloc_error_arg, arg);
}

void _palloc_error_message(int err, const char* fmt, ...) {
  // show detailed error message
  va_list args;
  va_start(args, fmt);
  palloc_show_error_message(fmt, args);
  va_end(args);
  // and call the error handler which may abort (or return normally)
  if (palloc_error_handler != NULL) {
    palloc_error_handler(err, palloc_atomic_load_ptr_acquire(void,&palloc_error_arg));
  }
  else {
    palloc_error_default(err);
  }
}

// --------------------------------------------------------
// Initialize options by checking the environment
// --------------------------------------------------------

// TODO: implement ourselves to reduce dependencies on the C runtime
#include <stdlib.h> // strtol
#include <string.h> // strstr


static void palloc_option_init(palloc_option_desc_t* desc) {
  // Read option value from the environment
  char s[64 + 1];
  char buf[64+1];
  _palloc_strlcpy(buf, "palloc_", sizeof(buf));
  _palloc_strlcat(buf, desc->name, sizeof(buf));
  bool found = _palloc_getenv(buf, s, sizeof(s));
  if (!found && desc->legacy_name != NULL) {
    _palloc_strlcpy(buf, "palloc_", sizeof(buf));
    _palloc_strlcat(buf, desc->legacy_name, sizeof(buf));
    found = _palloc_getenv(buf, s, sizeof(s));
    if (found) {
      _palloc_warning_message("environment option \"palloc_%s\" is deprecated -- use \"palloc_%s\" instead.\n", desc->legacy_name, desc->name);
    }
  }

  if (found) {
    size_t len = _palloc_strnlen(s, sizeof(buf) - 1);
    for (size_t i = 0; i < len; i++) {
      buf[i] = _palloc_toupper(s[i]);
    }
    buf[len] = 0;
    if (buf[0] == 0 || strstr("1;TRUE;YES;ON", buf) != NULL) {
      desc->value = 1;
      desc->init = INITIALIZED;
    }
    else if (strstr("0;FALSE;NO;OFF", buf) != NULL) {
      desc->value = 0;
      desc->init = INITIALIZED;
    }
    else {
      char* end = buf;
      long value = strtol(buf, &end, 10);
      if (palloc_option_has_size_in_kib(desc->option)) {
        // this option is interpreted in KiB to prevent overflow of `long` for large allocations
        // (long is 32-bit on 64-bit windows, which allows for 4TiB max.)
        size_t size = (value < 0 ? 0 : (size_t)value);
        bool overflow = false;
        if (*end == 'K') { end++; }
        else if (*end == 'M') { overflow = palloc_mul_overflow(size,PALLOC_KiB,&size); end++; }
        else if (*end == 'G') { overflow = palloc_mul_overflow(size,PALLOC_MiB,&size); end++; }
        else if (*end == 'T') { overflow = palloc_mul_overflow(size,PALLOC_GiB,&size); end++; }
        else { size = (size + PALLOC_KiB - 1) / PALLOC_KiB; }
        if (end[0] == 'I' && end[1] == 'B') { end += 2; } // KiB, MiB, GiB, TiB
        else if (*end == 'B') { end++; }                  // Kb, Mb, Gb, Tb
        if (overflow || size > PALLOC_MAX_ALLOC_SIZE) { size = (PALLOC_MAX_ALLOC_SIZE / PALLOC_KiB); }
        value = (size > LONG_MAX ? LONG_MAX : (long)size);
      }
      if (*end == 0) {
        palloc_option_set(desc->option, value);
      }
      else {
        // set `init` first to avoid recursion through _palloc_warning_message on palloc_verbose.
        desc->init = DEFAULTED;
        if (desc->option == palloc_option_verbose && desc->value == 0) {
          // if the 'palloc_verbose' env var has a bogus value we'd never know
          // (since the value defaults to 'off') so in that case briefly enable verbose
          desc->value = 1;
          _palloc_warning_message("environment option palloc_%s has an invalid value.\n", desc->name);
          desc->value = 0;
        }
        else {
          _palloc_warning_message("environment option palloc_%s has an invalid value.\n", desc->name);
        }
      }
    }
    palloc_assert_internal(desc->init != UNINIT);
  }
  else if (!_palloc_preloading()) {
    desc->init = DEFAULTED;
  }
}
