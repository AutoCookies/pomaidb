// PomaiDB memory shim: routes aligned heap allocations through vendored palloc
// (pa_malloc_aligned / pa_free / pa_heap_*). Requires linking palloc-static.
#pragma once

#include <palloc.h>

#ifdef __cplusplus
#include <cstddef>

typedef pa_heap_t palloc_heap_t;

inline void* palloc_malloc_aligned(std::size_t size, std::size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return pa_malloc_aligned(size, alignment);
}

inline void palloc_free(void* p) { pa_free(p); }

inline palloc_heap_t* palloc_heap_new(void) { return pa_heap_new(); }

inline void palloc_heap_delete(palloc_heap_t* heap) {
  if (heap) {
    pa_heap_delete(heap);
  }
}

inline void* palloc_heap_malloc_aligned(palloc_heap_t* heap, std::size_t size,
                                        std::size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return heap ? pa_heap_malloc_aligned(heap, size, alignment)
              : pa_malloc_aligned(size, alignment);
}

inline void palloc_option_set(long option, long value) {
  pa_option_set(static_cast<pa_option_t>(option), value);
}

static constexpr pa_option_t palloc_option_reserve_huge_os_pages =
    pa_option_reserve_huge_os_pages;

#else

#include <stddef.h>

typedef pa_heap_t palloc_heap_t;

static inline void* palloc_malloc_aligned(size_t size, size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return pa_malloc_aligned(size, alignment);
}

static inline void palloc_free(void* p) { pa_free(p); }

static inline palloc_heap_t* palloc_heap_new(void) { return pa_heap_new(); }

static inline void palloc_heap_delete(palloc_heap_t* heap) {
  if (heap) {
    pa_heap_delete(heap);
  }
}

static inline void* palloc_heap_malloc_aligned(palloc_heap_t* heap, size_t size,
                                               size_t alignment) {
  if (alignment < sizeof(void*)) alignment = sizeof(void*);
  return heap ? pa_heap_malloc_aligned(heap, size, alignment)
              : pa_malloc_aligned(size, alignment);
}

static inline void palloc_option_set(long option, long value) {
  pa_option_set((pa_option_t)option, value);
}

#define palloc_option_reserve_huge_os_pages pa_option_reserve_huge_os_pages

#endif
