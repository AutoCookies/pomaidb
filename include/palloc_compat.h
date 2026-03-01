// Compatibility header: map PomaiDB's palloc_* names to the pa_* API from third_party/palloc (submodule).
// When POMAI_USE_PALLOC=0, provides stubs using the system allocator (for CI sanitizer builds etc.).
#pragma once

#if defined(POMAI_USE_PALLOC) && POMAI_USE_PALLOC
#include "palloc.h"

#ifdef __cplusplus
#define palloc_heap_t         pa_heap_t
#define palloc_free           pa_free
#define palloc_malloc_aligned pa_malloc_aligned
#define palloc_heap_new       pa_heap_new
#define palloc_heap_delete    pa_heap_delete
#define palloc_heap_malloc_aligned pa_heap_malloc_aligned
#define palloc_option_set     pa_option_set
#define palloc_option_reserve_huge_os_pages pa_option_reserve_huge_os_pages
#endif

#else
// Stub: system allocator when palloc is disabled (no link to third_party/palloc).
#include <cstddef>
#include <cstdlib>
#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef __cplusplus
typedef void* palloc_heap_t;
static inline void* palloc_malloc_aligned(std::size_t size, std::size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
#if defined(_WIN32) || defined(_WIN64)
  return _aligned_malloc(size, alignment);
#else
  void* p = nullptr;
  return (posix_memalign(&p, alignment, size) == 0) ? p : nullptr;
#endif
}
static inline void palloc_free(void* p) { std::free(p); }
static inline palloc_heap_t* palloc_heap_new(void) { return nullptr; }
static inline void palloc_heap_delete(palloc_heap_t*) {}
static inline void* palloc_heap_malloc_aligned(palloc_heap_t*, std::size_t size, std::size_t alignment) {
  return palloc_malloc_aligned(size, alignment);
}
static inline void palloc_option_set(long, long) {}
static constexpr long palloc_option_reserve_huge_os_pages = 0;
#endif
#endif
