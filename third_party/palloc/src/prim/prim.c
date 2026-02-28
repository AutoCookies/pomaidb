/* ----------------------------------------------------------------------------
Copyright (c) 2018-2023, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

// Select the implementation of the primitives
// depending on the OS.

#if defined(_WIN32)
#include "windows/prim.c"  // VirtualAlloc (Windows)

#elif defined(__APPLE__)
#include "osx/prim.c"      // macOSX (actually defers to mmap in unix/prim.c)

#elif defined(__wasi__)
#define PALLOC_USE_SBRK
#include "wasi/prim.c"     // memory-grow or sbrk (Wasm)

#elif defined(__EMSCRIPTEN__)
#include "emscripten/prim.c" // emmalloc_*, + pthread support

#else
#include "unix/prim.c"     // mmap() (Linux, macOSX, BSD, Illumnos, Haiku, DragonFly, etc.)

#endif

// Generic process initialization
#ifndef PALLOC_PRIM_HAS_PROCESS_ATTACH
#if defined(__GNUC__) || defined(__clang__)
  // gcc,clang: use the constructor/destructor attribute
  // which for both seem to run before regular constructors/destructors
  #if defined(__clang__)
    #define palloc_attr_constructor __attribute__((constructor(101)))
    #define palloc_attr_destructor  __attribute__((destructor(101)))
  #else
    #define palloc_attr_constructor __attribute__((constructor))
    #define palloc_attr_destructor  __attribute__((destructor))
  #endif
  static void palloc_attr_constructor palloc_process_attach(void) {
    _palloc_auto_process_init();
  }
  static void palloc_attr_destructor palloc_process_detach(void) {
    _palloc_auto_process_done();
  }
#elif defined(__cplusplus)
  // C++: use static initialization to detect process start/end
  // This is not guaranteed to be first/last but the best we can generally do?
  struct palloc_init_done_t {
    palloc_init_done_t() {
      _palloc_auto_process_init();
    }
    ~palloc_init_done_t() {
      _palloc_auto_process_done();
    }
  };
  static palloc_init_done_t palloc_init_done;
 #else
  #pragma message("define a way to call _palloc_auto_process_init/done on your platform")
#endif
#endif

// Generic allocator init/done callback
#ifndef PALLOC_PRIM_HAS_ALLOCATOR_INIT
bool _palloc_is_redirected(void) {
  return false;
}
bool _palloc_allocator_init(const char** message) {
  if (message != NULL) { *message = NULL; }
  return true;
}
void _palloc_allocator_done(void) {
  // nothing to do
}
#endif
