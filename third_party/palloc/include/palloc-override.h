/* ----------------------------------------------------------------------------
Copyright (c) 2018-2020 Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_OVERRIDE_H
#define PALLOC_OVERRIDE_H

/* ----------------------------------------------------------------------------
This header can be used to statically redirect malloc/free and new/delete
to the palloc variants. This can be useful if one can include this file on
each source file in a project (but be careful when using external code to
not accidentally mix pointers from different allocators).
-----------------------------------------------------------------------------*/

#include <palloc.h>

// Standard C allocation
#define malloc(n)               palloc_malloc(n)
#define calloc(n,c)             palloc_calloc(n,c)
#define realloc(p,n)            palloc_realloc(p,n)
#define free(p)                 palloc_free(p)

#define strdup(s)               palloc_strdup(s)
#define strndup(s,n)            palloc_strndup(s,n)
#define realpath(f,n)           palloc_realpath(f,n)

// Microsoft extensions
#define _expand(p,n)            palloc_expand(p,n)
#define _msize(p)               palloc_usable_size(p)
#define _recalloc(p,n,c)        palloc_recalloc(p,n,c)

#define _strdup(s)              palloc_strdup(s)
#define _strndup(s,n)           palloc_strndup(s,n)
#define _wcsdup(s)              (wchar_t*)palloc_wcsdup((const unsigned short*)(s))
#define _mbsdup(s)              palloc_mbsdup(s)
#define _dupenv_s(b,n,v)        palloc_dupenv_s(b,n,v)
#define _wdupenv_s(b,n,v)       palloc_wdupenv_s((unsigned short*)(b),n,(const unsigned short*)(v))

// Various Posix and Unix variants
#define reallocf(p,n)           palloc_reallocf(p,n)
#define malloc_size(p)          palloc_usable_size(p)
#define malloc_usable_size(p)   palloc_usable_size(p)
#define malloc_good_size(sz)    palloc_malloc_good_size(sz)
#define cfree(p)                palloc_free(p)

#define valloc(n)               palloc_valloc(n)
#define pvalloc(n)              palloc_pvalloc(n)
#define reallocarray(p,s,n)     palloc_reallocarray(p,s,n)
#define reallocarr(p,s,n)       palloc_reallocarr(p,s,n)
#define memalign(a,n)           palloc_memalign(a,n)
#define aligned_alloc(a,n)      palloc_aligned_alloc(a,n)
#define posix_memalign(p,a,n)   palloc_posix_memalign(p,a,n)
#define _posix_memalign(p,a,n)  palloc_posix_memalign(p,a,n)

// Microsoft aligned variants
#define _aligned_malloc(n,a)                  palloc_malloc_aligned(n,a)
#define _aligned_realloc(p,n,a)               palloc_realloc_aligned(p,n,a)
#define _aligned_recalloc(p,s,n,a)            palloc_aligned_recalloc(p,s,n,a)
#define _aligned_msize(p,a,o)                 palloc_usable_size(p)
#define _aligned_free(p)                      palloc_free(p)
#define _aligned_offset_malloc(n,a,o)         palloc_malloc_aligned_at(n,a,o)
#define _aligned_offset_realloc(p,n,a,o)      palloc_realloc_aligned_at(p,n,a,o)
#define _aligned_offset_recalloc(p,s,n,a,o)   palloc_recalloc_aligned_at(p,s,n,a,o)

#endif // PALLOC_OVERRIDE_H
