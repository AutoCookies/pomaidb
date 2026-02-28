/* ----------------------------------------------------------------------------
Copyright (c) 2018-2024, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE   // for realpath() on Linux
#endif

#include "palloc.h"
#include "palloc/internal.h"
#include "palloc/atomic.h"
#include "palloc/prim.h"   // _palloc_prim_thread_id()

#include <string.h>      // memset, strlen (for palloc_strdup)
#include <stdlib.h>      // malloc, abort

#define PALLOC_IN_ALLOC_C
#include "alloc-override.c"
#include "free.c"
#undef PALLOC_IN_ALLOC_C

// ------------------------------------------------------
// Allocation
// ------------------------------------------------------

// Fast allocation in a page: just pop from the free list.
// Fall back to generic allocation only if the list is empty.
// Note: in release mode the (inlined) routine is about 7 instructions with a single test.
extern inline void* _palloc_page_malloc_zero(palloc_heap_t* heap, palloc_page_t* page, size_t size, bool zero, size_t* usable) palloc_attr_noexcept
{
  palloc_assert_internal(size >= PALLOC_PADDING_SIZE);
  palloc_assert_internal(page->block_size == 0 /* empty heap */ || palloc_page_block_size(page) >= size);

  // check the free list
  palloc_block_t* const block = page->free;
  if palloc_unlikely(block == NULL) {
    return _palloc_malloc_generic(heap, size, zero, 0, usable);
  }
  palloc_assert_internal(block != NULL && _palloc_ptr_page(block) == page);
  if (usable != NULL) { *usable = palloc_page_usable_block_size(page); };
  // pop from the free list
  page->free = palloc_block_next(page, block);
  page->used++;
  palloc_assert_internal(page->free == NULL || _palloc_ptr_page(page->free) == page);
  palloc_assert_internal(page->block_size < PALLOC_MAX_ALIGN_SIZE || _palloc_is_aligned(block, PALLOC_MAX_ALIGN_SIZE));

  #if PALLOC_DEBUG>3
  if (page->free_is_zero && size > sizeof(*block)) {
    palloc_assert_expensive(palloc_mem_is_zero(block+1,size - sizeof(*block)));
  }
  #endif

  // allow use of the block internally
  // note: when tracking we need to avoid ever touching the PALLOC_PADDING since
  // that is tracked by valgrind etc. as non-accessible (through the red-zone, see `palloc/track.h`)
  palloc_track_mem_undefined(block, palloc_page_usable_block_size(page));

  // zero the block? note: we need to zero the full block size (issue #63)
  if palloc_unlikely(zero) {
    palloc_assert_internal(page->block_size != 0); // do not call with zero'ing for huge blocks (see _palloc_malloc_generic)
    #if PALLOC_PADDING
    palloc_assert_internal(page->block_size >= PALLOC_PADDING_SIZE);
    #endif
    if (page->free_is_zero) {
      block->next = 0;
      palloc_track_mem_defined(block, page->block_size - PALLOC_PADDING_SIZE);
    }
    else {
      _palloc_memzero_aligned(block, page->block_size - PALLOC_PADDING_SIZE);
    }
  }

  #if (PALLOC_DEBUG>0) && !PALLOC_TRACK_ENABLED && !PALLOC_TSAN
  if (!zero && !palloc_page_is_huge(page)) {
    memset(block, PALLOC_DEBUG_UNINIT, palloc_page_usable_block_size(page));
  }
  #elif (PALLOC_SECURE!=0)
  if (!zero) { block->next = 0; } // don't leak internal data
  #endif

  #if (PALLOC_STAT>0)
  const size_t bsize = palloc_page_usable_block_size(page);
  if (bsize <= PALLOC_MEDIUM_OBJ_SIZE_MAX) {
    palloc_heap_stat_increase(heap, malloc_normal, bsize);
    palloc_heap_stat_counter_increase(heap, malloc_normal_count, 1);
    #if (PALLOC_STAT>1)
    const size_t bin = _palloc_bin(bsize);
    palloc_heap_stat_increase(heap, malloc_bins[bin], 1);
    palloc_heap_stat_increase(heap, malloc_requested, size - PALLOC_PADDING_SIZE);
    #endif
  }
  #endif

  #if PALLOC_PADDING // && !PALLOC_TRACK_ENABLED
    palloc_padding_t* const padding = (palloc_padding_t*)((uint8_t*)block + palloc_page_usable_block_size(page));
    ptrdiff_t delta = ((uint8_t*)padding - (uint8_t*)block - (size - PALLOC_PADDING_SIZE));
    #if (PALLOC_DEBUG>=2)
    palloc_assert_internal(delta >= 0 && palloc_page_usable_block_size(page) >= (size - PALLOC_PADDING_SIZE + delta));
    #endif
    palloc_track_mem_defined(padding,sizeof(palloc_padding_t));  // note: re-enable since palloc_page_usable_block_size may set noaccess
    padding->canary = palloc_ptr_encode_canary(page,block,page->keys);
    padding->delta  = (uint32_t)(delta);
    #if PALLOC_PADDING_CHECK
    if (!palloc_page_is_huge(page)) {
      uint8_t* fill = (uint8_t*)padding - delta;
      const size_t maxpad = (delta > PALLOC_MAX_ALIGN_SIZE ? PALLOC_MAX_ALIGN_SIZE : delta); // set at most N initial padding bytes
      for (size_t i = 0; i < maxpad; i++) { fill[i] = PALLOC_DEBUG_PADDING; }
    }
    #endif
  #endif

  return block;
}

// extra entries for improved efficiency in `alloc-aligned.c`.
extern void* _palloc_page_malloc(palloc_heap_t* heap, palloc_page_t* page, size_t size) palloc_attr_noexcept {
  return _palloc_page_malloc_zero(heap,page,size,false,NULL);
}
extern void* _palloc_page_malloc_zeroed(palloc_heap_t* heap, palloc_page_t* page, size_t size) palloc_attr_noexcept {
  return _palloc_page_malloc_zero(heap,page,size,true,NULL);
}

#if PALLOC_GUARDED
palloc_decl_restrict void* _palloc_heap_malloc_guarded(palloc_heap_t* heap, size_t size, bool zero) palloc_attr_noexcept;
#endif

static inline palloc_decl_restrict void* palloc_heap_malloc_small_zero(palloc_heap_t* heap, size_t size, bool zero, size_t* usable) palloc_attr_noexcept {
  palloc_assert(heap != NULL);
  palloc_assert(size <= PALLOC_SMALL_SIZE_MAX);
  #if PALLOC_DEBUG
  const uintptr_t tid = _palloc_thread_id();
  palloc_assert(heap->thread_id == 0 || heap->thread_id == tid); // heaps are thread local
  #endif
  #if (PALLOC_PADDING || PALLOC_GUARDED)
  if (size == 0) { size = sizeof(void*); }
  #endif
  #if PALLOC_GUARDED
  if (palloc_heap_malloc_use_guarded(heap,size)) {
    return _palloc_heap_malloc_guarded(heap, size, zero);
  }
  #endif

  // get page in constant time, and allocate from it
  palloc_page_t* page = _palloc_heap_get_free_small_page(heap, size + PALLOC_PADDING_SIZE);
  void* const p = _palloc_page_malloc_zero(heap, page, size + PALLOC_PADDING_SIZE, zero, usable);
  palloc_track_malloc(p,size,zero);

  #if PALLOC_DEBUG>3
  if (p != NULL && zero) {
    palloc_assert_expensive(palloc_mem_is_zero(p, size));
  }
  #endif
  return p;
}

// allocate a small block
palloc_decl_nodiscard extern inline palloc_decl_restrict void* palloc_heap_malloc_small(palloc_heap_t* heap, size_t size) palloc_attr_noexcept {
  return palloc_heap_malloc_small_zero(heap, size, false, NULL);
}

palloc_decl_nodiscard extern inline palloc_decl_restrict void* palloc_malloc_small(size_t size) palloc_attr_noexcept {
  return palloc_heap_malloc_small(palloc_prim_get_default_heap(), size);
}

// The main allocation function
extern inline void* _palloc_heap_malloc_zero_ex(palloc_heap_t* heap, size_t size, bool zero, size_t huge_alignment, size_t* usable) palloc_attr_noexcept {
  // fast path for small objects
  if palloc_likely(size <= PALLOC_SMALL_SIZE_MAX) {
    palloc_assert_internal(huge_alignment == 0);
    return palloc_heap_malloc_small_zero(heap, size, zero, usable);
  }
  #if PALLOC_GUARDED
  else if (huge_alignment==0 && palloc_heap_malloc_use_guarded(heap,size)) {
    return _palloc_heap_malloc_guarded(heap, size, zero);
  }
  #endif
  else {
    // regular allocation
    palloc_assert(heap!=NULL);
    palloc_assert(heap->thread_id == 0 || heap->thread_id == _palloc_thread_id());   // heaps are thread local
    void* const p = _palloc_malloc_generic(heap, size + PALLOC_PADDING_SIZE, zero, huge_alignment, usable);  // note: size can overflow but it is detected in malloc_generic
    palloc_track_malloc(p,size,zero);

    #if PALLOC_DEBUG>3
    if (p != NULL && zero) {
      palloc_assert_expensive(palloc_mem_is_zero(p, size));
    }
    #endif
    return p;
  }
}

extern inline void* _palloc_heap_malloc_zero(palloc_heap_t* heap, size_t size, bool zero) palloc_attr_noexcept {
  return _palloc_heap_malloc_zero_ex(heap, size, zero, 0, NULL);
}

palloc_decl_nodiscard extern inline palloc_decl_restrict void* palloc_heap_malloc(palloc_heap_t* heap, size_t size) palloc_attr_noexcept {
  return _palloc_heap_malloc_zero(heap, size, false);
}

palloc_decl_nodiscard extern inline palloc_decl_restrict void* palloc_malloc(size_t size) palloc_attr_noexcept {
  return palloc_heap_malloc(palloc_prim_get_default_heap(), size);
}

// zero initialized small block
palloc_decl_nodiscard palloc_decl_restrict void* palloc_zalloc_small(size_t size) palloc_attr_noexcept {
  return palloc_heap_malloc_small_zero(palloc_prim_get_default_heap(), size, true, NULL);
}

palloc_decl_nodiscard extern inline palloc_decl_restrict void* palloc_heap_zalloc(palloc_heap_t* heap, size_t size) palloc_attr_noexcept {
  return _palloc_heap_malloc_zero(heap, size, true);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_zalloc(size_t size) palloc_attr_noexcept {
  return palloc_heap_zalloc(palloc_prim_get_default_heap(),size);
}


palloc_decl_nodiscard extern inline palloc_decl_restrict void* palloc_heap_calloc(palloc_heap_t* heap, size_t count, size_t size) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(count,size,&total)) return NULL;
  return palloc_heap_zalloc(heap,total);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_calloc(size_t count, size_t size) palloc_attr_noexcept {
  return palloc_heap_calloc(palloc_prim_get_default_heap(),count,size);
}

// Return usable size
palloc_decl_nodiscard palloc_decl_restrict void* palloc_umalloc_small(size_t size, size_t* usable) palloc_attr_noexcept {
  return palloc_heap_malloc_small_zero(palloc_prim_get_default_heap(), size, false, usable);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_umalloc(palloc_heap_t* heap, size_t size, size_t* usable) palloc_attr_noexcept {
  return _palloc_heap_malloc_zero_ex(heap, size, false, 0, usable);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_umalloc(size_t size, size_t* usable) palloc_attr_noexcept {
  return palloc_heap_umalloc(palloc_prim_get_default_heap(), size, usable);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_uzalloc(size_t size, size_t* usable) palloc_attr_noexcept {
  return _palloc_heap_malloc_zero_ex(palloc_prim_get_default_heap(), size, true, 0, usable);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_ucalloc(size_t count, size_t size, size_t* usable) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(count,size,&total)) return NULL;
  return palloc_uzalloc(total, usable);
}

// Uninitialized `calloc`
palloc_decl_nodiscard extern palloc_decl_restrict void* palloc_heap_mallocn(palloc_heap_t* heap, size_t count, size_t size) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(count, size, &total)) return NULL;
  return palloc_heap_malloc(heap, total);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_mallocn(size_t count, size_t size) palloc_attr_noexcept {
  return palloc_heap_mallocn(palloc_prim_get_default_heap(),count,size);
}

// Expand (or shrink) in place (or fail)
void* palloc_expand(void* p, size_t newsize) palloc_attr_noexcept {
  #if PALLOC_PADDING
  // we do not shrink/expand with padding enabled
  PALLOC_UNUSED(p); PALLOC_UNUSED(newsize);
  return NULL;
  #else
  if (p == NULL) return NULL;
  const palloc_page_t* const page = palloc_validate_ptr_page(p,"palloc_expand");  
  const size_t size = _palloc_usable_size(p,page);
  if (newsize > size) return NULL;
  return p; // it fits
  #endif
}

void* _palloc_heap_realloc_zero(palloc_heap_t* heap, void* p, size_t newsize, bool zero, size_t* usable_pre, size_t* usable_post) palloc_attr_noexcept {
  // if p == NULL then behave as malloc.
  // else if size == 0 then reallocate to a zero-sized block (and don't return NULL, just as palloc_malloc(0)).
  // (this means that returning NULL always indicates an error, and `p` will not have been freed in that case.)
  const palloc_page_t* page;
  size_t size;
  if (p==NULL) {
    page = NULL;
    size = 0;
    if (usable_pre!=NULL) { *usable_pre = 0; }
  }
  else {    
    page = palloc_validate_ptr_page(p,"palloc_realloc");  
    size = _palloc_usable_size(p,page);
    if (usable_pre!=NULL) { *usable_pre = palloc_page_usable_block_size(page); }    
  }
  if palloc_unlikely(newsize <= size && newsize >= (size / 2) && newsize > 0) {  // note: newsize must be > 0 or otherwise we return NULL for realloc(NULL,0)
    palloc_assert_internal(p!=NULL);
    // todo: do not track as the usable size is still the same in the free; adjust potential padding?
    // palloc_track_resize(p,size,newsize)
    // if (newsize < size) { palloc_track_mem_noaccess((uint8_t*)p + newsize, size - newsize); }
    if (usable_post!=NULL) { *usable_post = palloc_page_usable_block_size(page); }
    return p;  // reallocation still fits and not more than 50% waste
  }
  void* newp = palloc_heap_umalloc(heap,newsize,usable_post);
  if palloc_likely(newp != NULL) {
    if (zero && newsize > size) {
      // also set last word in the previous allocation to zero to ensure any padding is zero-initialized
      const size_t start = (size >= sizeof(intptr_t) ? size - sizeof(intptr_t) : 0);
      _palloc_memzero((uint8_t*)newp + start, newsize - start);
    }
    else if (newsize == 0) {
      ((uint8_t*)newp)[0] = 0; // work around for applications that expect zero-reallocation to be zero initialized (issue #725)
    }
    if palloc_likely(p != NULL) {
      const size_t copysize = (newsize > size ? size : newsize);
      palloc_track_mem_defined(p,copysize);  // _palloc_useable_size may be too large for byte precise memory tracking..
      _palloc_memcpy(newp, p, copysize);
      palloc_free(p); // only free the original pointer if successful
    }
  }
  return newp;
}

palloc_decl_nodiscard void* palloc_heap_realloc(palloc_heap_t* heap, void* p, size_t newsize) palloc_attr_noexcept {
  return _palloc_heap_realloc_zero(heap, p, newsize, false, NULL, NULL);
}

palloc_decl_nodiscard void* palloc_heap_reallocn(palloc_heap_t* heap, void* p, size_t count, size_t size) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(count, size, &total)) return NULL;
  return palloc_heap_realloc(heap, p, total);
}


// Reallocate but free `p` on errors
palloc_decl_nodiscard void* palloc_heap_reallocf(palloc_heap_t* heap, void* p, size_t newsize) palloc_attr_noexcept {
  void* newp = palloc_heap_realloc(heap, p, newsize);
  if (newp==NULL && p!=NULL) palloc_free(p);
  return newp;
}

palloc_decl_nodiscard void* palloc_heap_rezalloc(palloc_heap_t* heap, void* p, size_t newsize) palloc_attr_noexcept {
  return _palloc_heap_realloc_zero(heap, p, newsize, true, NULL, NULL);
}

palloc_decl_nodiscard void* palloc_heap_recalloc(palloc_heap_t* heap, void* p, size_t count, size_t size) palloc_attr_noexcept {
  size_t total;
  if (palloc_count_size_overflow(count, size, &total)) return NULL;
  return palloc_heap_rezalloc(heap, p, total);
}


palloc_decl_nodiscard void* palloc_realloc(void* p, size_t newsize) palloc_attr_noexcept {
  return palloc_heap_realloc(palloc_prim_get_default_heap(),p,newsize);
}

palloc_decl_nodiscard void* palloc_reallocn(void* p, size_t count, size_t size) palloc_attr_noexcept {
  return palloc_heap_reallocn(palloc_prim_get_default_heap(),p,count,size);
}

palloc_decl_nodiscard void* palloc_urealloc(void* p, size_t newsize, size_t* usable_pre, size_t* usable_post) palloc_attr_noexcept {
  return _palloc_heap_realloc_zero(palloc_prim_get_default_heap(),p,newsize, false, usable_pre, usable_post);
}

// Reallocate but free `p` on errors
palloc_decl_nodiscard void* palloc_reallocf(void* p, size_t newsize) palloc_attr_noexcept {
  return palloc_heap_reallocf(palloc_prim_get_default_heap(),p,newsize);
}

palloc_decl_nodiscard void* palloc_rezalloc(void* p, size_t newsize) palloc_attr_noexcept {
  return palloc_heap_rezalloc(palloc_prim_get_default_heap(), p, newsize);
}

palloc_decl_nodiscard void* palloc_recalloc(void* p, size_t count, size_t size) palloc_attr_noexcept {
  return palloc_heap_recalloc(palloc_prim_get_default_heap(), p, count, size);
}



// ------------------------------------------------------
// strdup, strndup, and realpath
// ------------------------------------------------------

// `strdup` using palloc_malloc
palloc_decl_nodiscard palloc_decl_restrict char* palloc_heap_strdup(palloc_heap_t* heap, const char* s) palloc_attr_noexcept {
  if (s == NULL) return NULL;
  size_t len = _palloc_strlen(s);
  char* t = (char*)palloc_heap_malloc(heap,len+1);
  if (t == NULL) return NULL;
  _palloc_memcpy(t, s, len);
  t[len] = 0;
  return t;
}

palloc_decl_nodiscard palloc_decl_restrict char* palloc_strdup(const char* s) palloc_attr_noexcept {
  return palloc_heap_strdup(palloc_prim_get_default_heap(), s);
}

// `strndup` using palloc_malloc
palloc_decl_nodiscard palloc_decl_restrict char* palloc_heap_strndup(palloc_heap_t* heap, const char* s, size_t n) palloc_attr_noexcept {
  if (s == NULL) return NULL;
  const size_t len = _palloc_strnlen(s,n);  // len <= n
  char* t = (char*)palloc_heap_malloc(heap, len+1);
  if (t == NULL) return NULL;
  _palloc_memcpy(t, s, len);
  t[len] = 0;
  return t;
}

palloc_decl_nodiscard palloc_decl_restrict char* palloc_strndup(const char* s, size_t n) palloc_attr_noexcept {
  return palloc_heap_strndup(palloc_prim_get_default_heap(),s,n);
}

#ifndef __wasi__
// `realpath` using palloc_malloc
#ifdef _WIN32
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif

palloc_decl_nodiscard palloc_decl_restrict char* palloc_heap_realpath(palloc_heap_t* heap, const char* fname, char* resolved_name) palloc_attr_noexcept {
  // todo: use GetFullPathNameW to allow longer file names
  char buf[PATH_MAX];
  DWORD res = GetFullPathNameA(fname, PATH_MAX, (resolved_name == NULL ? buf : resolved_name), NULL);
  if (res == 0) {
    errno = GetLastError(); return NULL;
  }
  else if (res > PATH_MAX) {
    errno = EINVAL; return NULL;
  }
  else if (resolved_name != NULL) {
    return resolved_name;
  }
  else {
    return palloc_heap_strndup(heap, buf, PATH_MAX);
  }
}
#else
/*
#include <unistd.h>  // pathconf
static size_t palloc_path_max(void) {
  static size_t path_max = 0;
  if (path_max <= 0) {
    long m = pathconf("/",_PC_PATH_MAX);
    if (m <= 0) path_max = 4096;      // guess
    else if (m < 256) path_max = 256; // at least 256
    else path_max = m;
  }
  return path_max;
}
*/
char* palloc_heap_realpath(palloc_heap_t* heap, const char* fname, char* resolved_name) palloc_attr_noexcept {
  if (resolved_name != NULL) {
    return realpath(fname,resolved_name);
  }
  else {
    char* rname = realpath(fname, NULL);
    if (rname == NULL) return NULL;
    char* result = palloc_heap_strdup(heap, rname);
    palloc_cfree(rname);  // use checked free (which may be redirected to our free but that's ok)
    // note: with ASAN realpath is intercepted and palloc_cfree may leak the returned pointer :-(
    return result;
  }
  /*
    const size_t n  = palloc_path_max();
    char* buf = (char*)palloc_malloc(n+1);
    if (buf == NULL) {
      errno = ENOMEM;
      return NULL;
    }
    char* rname  = realpath(fname,buf);
    char* result = palloc_heap_strndup(heap,rname,n); // ok if `rname==NULL`
    palloc_free(buf);
    return result;
  }
  */
}
#endif

palloc_decl_nodiscard palloc_decl_restrict char* palloc_realpath(const char* fname, char* resolved_name) palloc_attr_noexcept {
  return palloc_heap_realpath(palloc_prim_get_default_heap(),fname,resolved_name);
}
#endif

/*-------------------------------------------------------
C++ new and new_aligned
The standard requires calling into `get_new_handler` and
throwing the bad_alloc exception on failure. If we compile
with a C++ compiler we can implement this precisely. If we
use a C compiler we cannot throw a `bad_alloc` exception
but we call `exit` instead (i.e. not returning).
-------------------------------------------------------*/

#ifdef __cplusplus
#include <new>
static bool palloc_try_new_handler(bool nothrow) {
  #if defined(_MSC_VER) || (__cplusplus >= 201103L)
    std::new_handler h = std::get_new_handler();
  #else
    std::new_handler h = std::set_new_handler();
    std::set_new_handler(h);
  #endif
  if (h==NULL) {
    _palloc_error_message(ENOMEM, "out of memory in 'new'");
    #if defined(_CPPUNWIND) || defined(__cpp_exceptions)  // exceptions are not always enabled
    if (!nothrow) {
      throw std::bad_alloc();
    }
    #else
    PALLOC_UNUSED(nothrow);
    #endif
    return false;
  }
  else {
    h();
    return true;
  }
}
#else
typedef void (*std_new_handler_t)(void);

#if (defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER)))  // exclude clang-cl, see issue #631
std_new_handler_t __attribute__((weak)) _ZSt15get_new_handlerv(void) {
  return NULL;
}
static std_new_handler_t palloc_get_new_handler(void) {
  return _ZSt15get_new_handlerv();
}
#else
// note: on windows we could dynamically link to `?get_new_handler@std@@YAP6AXXZXZ`.
static std_new_handler_t palloc_get_new_handler(void) {
  return NULL;
}
#endif

static bool palloc_try_new_handler(bool nothrow) {
  std_new_handler_t h = palloc_get_new_handler();
  if (h==NULL) {
    _palloc_error_message(ENOMEM, "out of memory in 'new'");
    if (!nothrow) {
      abort();  // cannot throw in plain C, use abort
    }
    return false;
  }
  else {
    h();
    return true;
  }
}
#endif

palloc_decl_export palloc_decl_noinline void* palloc_heap_try_new(palloc_heap_t* heap, size_t size, bool nothrow ) {
  void* p = NULL;
  while(p == NULL && palloc_try_new_handler(nothrow)) {
    p = palloc_heap_malloc(heap,size);
  }
  return p;
}

static palloc_decl_noinline void* palloc_try_new(size_t size, bool nothrow) {
  return palloc_heap_try_new(palloc_prim_get_default_heap(), size, nothrow);
}


palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_alloc_new(palloc_heap_t* heap, size_t size) {
  void* p = palloc_heap_malloc(heap,size);
  if palloc_unlikely(p == NULL) return palloc_heap_try_new(heap, size, false);
  return p;
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_new(size_t size) {
  return palloc_heap_alloc_new(palloc_prim_get_default_heap(), size);
}


palloc_decl_nodiscard palloc_decl_restrict void* palloc_heap_alloc_new_n(palloc_heap_t* heap, size_t count, size_t size) {
  size_t total;
  if palloc_unlikely(palloc_count_size_overflow(count, size, &total)) {
    palloc_try_new_handler(false);  // on overflow we invoke the try_new_handler once to potentially throw std::bad_alloc
    return NULL;
  }
  else {
    return palloc_heap_alloc_new(heap,total);
  }
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_new_n(size_t count, size_t size) {
  return palloc_heap_alloc_new_n(palloc_prim_get_default_heap(), count, size);
}


palloc_decl_nodiscard palloc_decl_restrict void* palloc_new_nothrow(size_t size) palloc_attr_noexcept {
  void* p = palloc_malloc(size);
  if palloc_unlikely(p == NULL) return palloc_try_new(size, true);
  return p;
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_new_aligned(size_t size, size_t alignment) {
  void* p;
  do {
    p = palloc_malloc_aligned(size, alignment);
  }
  while(p == NULL && palloc_try_new_handler(false));
  return p;
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_new_aligned_nothrow(size_t size, size_t alignment) palloc_attr_noexcept {
  void* p;
  do {
    p = palloc_malloc_aligned(size, alignment);
  }
  while(p == NULL && palloc_try_new_handler(true));
  return p;
}

palloc_decl_nodiscard void* palloc_new_realloc(void* p, size_t newsize) {
  void* q;
  do {
    q = palloc_realloc(p, newsize);
  } while (q == NULL && palloc_try_new_handler(false));
  return q;
}

palloc_decl_nodiscard void* palloc_new_reallocn(void* p, size_t newcount, size_t size) {
  size_t total;
  if palloc_unlikely(palloc_count_size_overflow(newcount, size, &total)) {
    palloc_try_new_handler(false);  // on overflow we invoke the try_new_handler once to potentially throw std::bad_alloc
    return NULL;
  }
  else {
    return palloc_new_realloc(p, total);
  }
}

#if PALLOC_GUARDED
// We always allocate a guarded allocation at an offset (`palloc_page_has_aligned` will be true).
// We then set the first word of the block to `0` for regular offset aligned allocations (in `alloc-aligned.c`)
// and the first word to `~0` for guarded allocations to have a correct `palloc_usable_size`

static void* palloc_block_ptr_set_guarded(palloc_block_t* block, size_t obj_size) {
  // TODO: we can still make padding work by moving it out of the guard page area
  palloc_page_t* const page = _palloc_ptr_page(block);
  palloc_page_set_has_aligned(page, true);
  block->next = PALLOC_BLOCK_TAG_GUARDED;

  // set guard page at the end of the block
  palloc_segment_t* const segment = _palloc_page_segment(page);
  const size_t block_size = palloc_page_block_size(page);  // must use `block_size` to match `palloc_free_local`
  const size_t os_page_size = _palloc_os_page_size();
  palloc_assert_internal(block_size >= obj_size + os_page_size + sizeof(palloc_block_t));
  if (block_size < obj_size + os_page_size + sizeof(palloc_block_t)) {
    // should never happen
    palloc_free(block);
    return NULL;
  }
  uint8_t* guard_page = (uint8_t*)block + block_size - os_page_size;
  palloc_assert_internal(_palloc_is_aligned(guard_page, os_page_size));
  if palloc_likely(segment->allow_decommit && _palloc_is_aligned(guard_page, os_page_size)) {
    const bool ok = _palloc_os_protect(guard_page, os_page_size);
    if palloc_unlikely(!ok) {
      _palloc_warning_message("failed to set a guard page behind an object (object %p of size %zu)\n", block, block_size);
    }
  }
  else {
    _palloc_warning_message("unable to set a guard page behind an object due to pinned memory (large OS pages?) (object %p of size %zu)\n", block, block_size);
  }

  // align pointer just in front of the guard page
  size_t offset = block_size - os_page_size - obj_size;
  palloc_assert_internal(offset > sizeof(palloc_block_t));
  if (offset > PALLOC_BLOCK_ALIGNMENT_MAX) {
    // give up to place it right in front of the guard page if the offset is too large for unalignment
    offset = PALLOC_BLOCK_ALIGNMENT_MAX;
  }
  void* p = (uint8_t*)block + offset;
  palloc_track_align(block, p, offset, obj_size);
  palloc_track_mem_defined(block, sizeof(palloc_block_t));
  return p;
}

palloc_decl_restrict void* _palloc_heap_malloc_guarded(palloc_heap_t* heap, size_t size, bool zero) palloc_attr_noexcept
{
  #if defined(PALLOC_PADDING_SIZE)
  palloc_assert(PALLOC_PADDING_SIZE==0);
  #endif
  // allocate multiple of page size ending in a guard page
  // ensure minimal alignment requirement?
  const size_t os_page_size = _palloc_os_page_size();
  const size_t obj_size = (palloc_option_is_enabled(palloc_option_guarded_precise) ? size : _palloc_align_up(size, PALLOC_MAX_ALIGN_SIZE));
  const size_t bsize    = _palloc_align_up(_palloc_align_up(obj_size, PALLOC_MAX_ALIGN_SIZE) + sizeof(palloc_block_t), PALLOC_MAX_ALIGN_SIZE);
  const size_t req_size = _palloc_align_up(bsize + os_page_size, os_page_size);
  palloc_block_t* const block = (palloc_block_t*)_palloc_malloc_generic(heap, req_size, zero, 0 /* huge_alignment */, NULL);
  if (block==NULL) return NULL;
  void* const p   = palloc_block_ptr_set_guarded(block, obj_size);

  // stats
  palloc_track_malloc(p, size, zero);
  if (p != NULL) {
    if (!palloc_heap_is_initialized(heap)) { heap = palloc_prim_get_default_heap(); }
    #if PALLOC_STAT>1
    palloc_heap_stat_adjust_decrease(heap, malloc_requested, req_size);
    palloc_heap_stat_increase(heap, malloc_requested, size);
    #endif
    _palloc_stat_counter_increase(&heap->tld->stats.malloc_guarded_count, 1);
  }
  #if PALLOC_DEBUG>3
  if (p != NULL && zero) {
    palloc_assert_expensive(palloc_mem_is_zero(p, size));
  }
  #endif
  return p;
}
#endif

// ------------------------------------------------------
// ensure explicit external inline definitions are emitted!
// ------------------------------------------------------

#ifdef __cplusplus
void* _palloc_externs[] = {
  (void*)&_palloc_page_malloc,
  (void*)&_palloc_page_malloc_zero,
  (void*)&_palloc_heap_malloc_zero,
  (void*)&_palloc_heap_malloc_zero_ex,
  (void*)&palloc_malloc,
  (void*)&palloc_malloc_small,
  (void*)&palloc_zalloc_small,
  (void*)&palloc_heap_malloc,
  (void*)&palloc_heap_zalloc,
  (void*)&palloc_heap_malloc_small,
  // (void*)&palloc_heap_alloc_new,
  // (void*)&palloc_heap_alloc_new_n
};
#endif
