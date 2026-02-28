/* ----------------------------------------------------------------------------
Copyright (c) 2018-2023, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_TRACK_H
#define PALLOC_TRACK_H

/* ------------------------------------------------------------------------------------------------------
Track memory ranges with macros for tools like Valgrind address sanitizer, or other memory checkers.
These can be defined for tracking allocation:

  #define palloc_track_malloc_size(p,reqsize,size,zero)
  #define palloc_track_free_size(p,_size)

The macros are set up such that the size passed to `palloc_track_free_size`
always matches the size of `palloc_track_malloc_size`. (currently, `size == palloc_usable_size(p)`).
The `reqsize` is what the user requested, and `size >= reqsize`.
The `size` is either byte precise (and `size==reqsize`) if `PALLOC_PADDING` is enabled,
or otherwise it is the usable block size which may be larger than the original request.
Use `_palloc_block_size_of(void* p)` to get the full block size that was allocated (including padding etc).
The `zero` parameter is `true` if the allocated block is zero initialized.

Optional:

  #define palloc_track_align(p,alignedp,offset,size)
  #define palloc_track_resize(p,oldsize,newsize)
  #define palloc_track_init()

The `palloc_track_align` is called right after a `palloc_track_malloc` for aligned pointers in a block.
The corresponding `palloc_track_free` still uses the block start pointer and original size (corresponding to the `palloc_track_malloc`).
The `palloc_track_resize` is currently unused but could be called on reallocations within a block.
`palloc_track_init` is called at program start.

The following macros are for tools like asan and valgrind to track whether memory is
defined, undefined, or not accessible at all:

  #define palloc_track_mem_defined(p,size)
  #define palloc_track_mem_undefined(p,size)
  #define palloc_track_mem_noaccess(p,size)

-------------------------------------------------------------------------------------------------------*/

#if PALLOC_TRACK_VALGRIND
// valgrind tool

#define PALLOC_TRACK_ENABLED      1
#define PALLOC_TRACK_HEAP_DESTROY 1           // track free of individual blocks on heap_destroy
#define PALLOC_TRACK_TOOL         "valgrind"

#include <valgrind/valgrind.h>
#include <valgrind/memcheck.h>

#define palloc_track_malloc_size(p,reqsize,size,zero) VALGRIND_MALLOCLIKE_BLOCK(p,size,PALLOC_PADDING_SIZE /*red zone*/,zero)
#define palloc_track_free_size(p,_size)               VALGRIND_FREELIKE_BLOCK(p,PALLOC_PADDING_SIZE /*red zone*/)
#define palloc_track_resize(p,oldsize,newsize)        VALGRIND_RESIZEINPLACE_BLOCK(p,oldsize,newsize,PALLOC_PADDING_SIZE /*red zone*/)
#define palloc_track_mem_defined(p,size)              VALGRIND_MAKE_MEM_DEFINED(p,size)
#define palloc_track_mem_undefined(p,size)            VALGRIND_MAKE_MEM_UNDEFINED(p,size)
#define palloc_track_mem_noaccess(p,size)             VALGRIND_MAKE_MEM_NOACCESS(p,size)

#elif PALLOC_TRACK_ASAN
// address sanitizer

#define PALLOC_TRACK_ENABLED      1
#define PALLOC_TRACK_HEAP_DESTROY 0
#define PALLOC_TRACK_TOOL         "asan"

#include <sanitizer/asan_interface.h>

#define palloc_track_malloc_size(p,reqsize,size,zero) ASAN_UNPOISON_MEMORY_REGION(p,size)
#define palloc_track_free_size(p,size)                ASAN_POISON_MEMORY_REGION(p,size)
#define palloc_track_mem_defined(p,size)              ASAN_UNPOISON_MEMORY_REGION(p,size)
#define palloc_track_mem_undefined(p,size)            ASAN_UNPOISON_MEMORY_REGION(p,size)
#define palloc_track_mem_noaccess(p,size)             ASAN_POISON_MEMORY_REGION(p,size)

#elif PALLOC_TRACK_ETW
// windows event tracing

#define PALLOC_TRACK_ENABLED      1
#define PALLOC_TRACK_HEAP_DESTROY 1
#define PALLOC_TRACK_TOOL         "ETW"

#include "../src/prim/windows/etw.h"

#define palloc_track_init()                           EventRegistermicrosoft_windows_palloc();
#define palloc_track_malloc_size(p,reqsize,size,zero) EventWriteETW_PALLOC_ALLOC((UINT64)(p), size)
#define palloc_track_free_size(p,size)                EventWriteETW_PALLOC_FREE((UINT64)(p), size)

#else
// no tracking

#define PALLOC_TRACK_ENABLED      0
#define PALLOC_TRACK_HEAP_DESTROY 0
#define PALLOC_TRACK_TOOL         "none"

#define palloc_track_malloc_size(p,reqsize,size,zero)
#define palloc_track_free_size(p,_size)

#endif

// -------------------
// Utility definitions

#ifndef palloc_track_resize
#define palloc_track_resize(p,oldsize,newsize)      palloc_track_free_size(p,oldsize); palloc_track_malloc(p,newsize,false)
#endif

#ifndef palloc_track_align
#define palloc_track_align(p,alignedp,offset,size)  palloc_track_mem_noaccess(p,offset)
#endif

#ifndef palloc_track_init
#define palloc_track_init()
#endif

#ifndef palloc_track_mem_defined
#define palloc_track_mem_defined(p,size)
#endif

#ifndef palloc_track_mem_undefined
#define palloc_track_mem_undefined(p,size)
#endif

#ifndef palloc_track_mem_noaccess
#define palloc_track_mem_noaccess(p,size)
#endif


#if PALLOC_PADDING
#define palloc_track_malloc(p,reqsize,zero) \
  if ((p)!=NULL) { \
    palloc_assert_internal(palloc_usable_size(p)==(reqsize)); \
    palloc_track_malloc_size(p,reqsize,reqsize,zero); \
  }
#else
#define palloc_track_malloc(p,reqsize,zero) \
  if ((p)!=NULL) { \
    palloc_assert_internal(palloc_usable_size(p)>=(reqsize)); \
    palloc_track_malloc_size(p,reqsize,palloc_usable_size(p),zero); \
  }
#endif

#endif
