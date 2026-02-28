/* ----------------------------------------------------------------------------
Copyright (c) 2018-2021, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

// ------------------------------------------------------------------------
// mi prefixed publi definitions of various Posix, Unix, and C++ functions
// for convenience and used when overriding these functions.
// ------------------------------------------------------------------------
#include "palloc.h"
#include "palloc/internal.h"

// ------------------------------------------------------
// Posix & Unix functions definitions
// ------------------------------------------------------

#include <errno.h>
#include <string.h>  // memset
#include <stdlib.h>  // getenv

#ifdef _MSC_VER
#pragma warning(disable:4996)  // getenv _wgetenv
#endif

#ifndef EINVAL
#define EINVAL 22
#endif
#ifndef ENOMEM
#define ENOMEM 12
#endif


palloc_decl_nodiscard size_t palloc_malloc_size(const void* p) palloc_attr_noexcept {
  // if (!palloc_is_in_heap_region(p)) return 0;
  return palloc_usable_size(p);
}

palloc_decl_nodiscard size_t palloc_malloc_usable_size(const void *p) palloc_attr_noexcept {
  // if (!palloc_is_in_heap_region(p)) return 0;
  return palloc_usable_size(p);
}

palloc_decl_nodiscard size_t palloc_malloc_good_size(size_t size) palloc_attr_noexcept {
  return palloc_good_size(size);
}

void palloc_cfree(void* p) palloc_attr_noexcept {
  if (palloc_is_in_heap_region(p)) {
    palloc_free(p);
  }
}

int palloc_posix_memalign(void** p, size_t alignment, size_t size) palloc_attr_noexcept {
  // Note: The spec dictates we should not modify `*p` on an error. (issue#27)
  // <http://man7.org/linux/man-pages/man3/posix_memalign.3.html>
  if (p == NULL) return EINVAL;
  if ((alignment % sizeof(void*)) != 0) return EINVAL;                 // natural alignment
  // it is also required that alignment is a power of 2 and > 0; this is checked in `palloc_malloc_aligned`
  if (alignment==0 || !_palloc_is_power_of_two(alignment)) return EINVAL;  // not a power of 2
  void* q = palloc_malloc_aligned(size, alignment);
  if (q==NULL && size != 0) return ENOMEM;
  palloc_assert_internal(((uintptr_t)q % alignment) == 0);
  *p = q;
  return 0;
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_memalign(size_t alignment, size_t size) palloc_attr_noexcept {
  void* p = palloc_malloc_aligned(size, alignment);
  palloc_assert_internal(((uintptr_t)p % alignment) == 0);
  return p;
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_valloc(size_t size) palloc_attr_noexcept {
  return palloc_memalign( _palloc_os_page_size(), size );
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_pvalloc(size_t size) palloc_attr_noexcept {
  size_t psize = _palloc_os_page_size();
  if (size >= SIZE_MAX - psize) return NULL; // overflow
  size_t asize = _palloc_align_up(size, psize);
  return palloc_malloc_aligned(asize, psize);
}

palloc_decl_nodiscard palloc_decl_restrict void* palloc_aligned_alloc(size_t alignment, size_t size) palloc_attr_noexcept {
  // C11 requires the size to be an integral multiple of the alignment, see <https://en.cppreference.com/w/c/memory/aligned_alloc>.
  // unfortunately, it turns out quite some programs pass a size that is not an integral multiple so skip this check..
  /* if palloc_unlikely((size & (alignment - 1)) != 0) { // C11 requires alignment>0 && integral multiple, see <https://en.cppreference.com/w/c/memory/aligned_alloc>
      #if PALLOC_DEBUG > 0
      _palloc_error_message(EOVERFLOW, "(palloc_)aligned_alloc requires the size to be an integral multiple of the alignment (size %zu, alignment %zu)\n", size, alignment);
      #endif
      return NULL;
    }
  */
  // C11 also requires alignment to be a power-of-two (and > 0) which is checked in palloc_malloc_aligned
  void* p = palloc_malloc_aligned(size, alignment);
  palloc_assert_internal(((uintptr_t)p % alignment) == 0);
  return p;
}

palloc_decl_nodiscard void* palloc_reallocarray( void* p, size_t count, size_t size ) palloc_attr_noexcept {  // BSD
  void* newp = palloc_reallocn(p,count,size);
  if (newp==NULL) { errno = ENOMEM; }
  return newp;
}

palloc_decl_nodiscard int palloc_reallocarr( void* p, size_t count, size_t size ) palloc_attr_noexcept { // NetBSD
  palloc_assert(p != NULL);
  if (p == NULL) {
    errno = EINVAL;
    return EINVAL;
  }
  void** op = (void**)p;
  void* newp = palloc_reallocarray(*op, count, size);
  if palloc_unlikely(newp == NULL) { return errno; }
  *op = newp;
  return 0;
}

void* palloc__expand(void* p, size_t newsize) palloc_attr_noexcept {  // Microsoft
  void* res = palloc_expand(p, newsize);
  if (res == NULL) { errno = ENOMEM; }
  return res;
}

palloc_decl_nodiscard palloc_decl_restrict unsigned short* palloc_wcsdup(const unsigned short* s) palloc_attr_noexcept {
  if (s==NULL) return NULL;
  size_t len;
  for(len = 0; s[len] != 0; len++) { }
  size_t size = (len+1)*sizeof(unsigned short);
  unsigned short* p = (unsigned short*)palloc_malloc(size);
  if (p != NULL) {
    _palloc_memcpy(p,s,size);
  }
  return p;
}

palloc_decl_nodiscard palloc_decl_restrict unsigned char* palloc_mbsdup(const unsigned char* s)  palloc_attr_noexcept {
  return (unsigned char*)palloc_strdup((const char*)s);
}

int palloc_dupenv_s(char** buf, size_t* size, const char* name) palloc_attr_noexcept {
  if (buf==NULL || name==NULL) return EINVAL;
  if (size != NULL) *size = 0;
  char* p = getenv(name);        // mscver warning 4996
  if (p==NULL) {
    *buf = NULL;
  }
  else {
    *buf = palloc_strdup(p);
    if (*buf==NULL) return ENOMEM;
    if (size != NULL) *size = _palloc_strlen(p);
  }
  return 0;
}

int palloc_wdupenv_s(unsigned short** buf, size_t* size, const unsigned short* name) palloc_attr_noexcept {
  if (buf==NULL || name==NULL) return EINVAL;
  if (size != NULL) *size = 0;
#if !defined(_WIN32) || (defined(WINAPI_FAMILY) && (WINAPI_FAMILY != WINAPI_FAMILY_DESKTOP_APP))
  // not supported
  *buf = NULL;
  return EINVAL;
#else
  unsigned short* p = (unsigned short*)_wgetenv((const wchar_t*)name);  // msvc warning 4996
  if (p==NULL) {
    *buf = NULL;
  }
  else {
    *buf = palloc_wcsdup(p);
    if (*buf==NULL) return ENOMEM;
    if (size != NULL) *size = wcslen((const wchar_t*)p);
  }
  return 0;
#endif
}

palloc_decl_nodiscard void* palloc_aligned_offset_recalloc(void* p, size_t newcount, size_t size, size_t alignment, size_t offset) palloc_attr_noexcept { // Microsoft
  return palloc_recalloc_aligned_at(p, newcount, size, alignment, offset);
}

palloc_decl_nodiscard void* palloc_aligned_recalloc(void* p, size_t newcount, size_t size, size_t alignment) palloc_attr_noexcept { // Microsoft
  return palloc_recalloc_aligned(p, newcount, size, alignment);
}
