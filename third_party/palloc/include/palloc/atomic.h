/* ----------------------------------------------------------------------------
Copyright (c) 2018-2024 Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/
#pragma once
#ifndef PALLOC_ATOMIC_H
#define PALLOC_ATOMIC_H

// include windows.h or pthreads.h
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif !defined(__wasi__) && (!defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__))
#define  PALLOC_USE_PTHREADS
#include <pthread.h>
#endif

// --------------------------------------------------------------------------------------------
// Atomics
// We need to be portable between C, C++, and MSVC.
// We base the primitives on the C/C++ atomics and create a minimal wrapper for MSVC in C compilation mode.
// This is why we try to use only `uintptr_t` and `<type>*` as atomic types.
// To gain better insight in the range of used atomics, we use explicitly named memory order operations
// instead of passing the memory order as a parameter.
// -----------------------------------------------------------------------------------------------

#if defined(__cplusplus)
// Use C++ atomics
#include <atomic>
#define  _Atomic(tp)              std::atomic<tp>
#define  palloc_atomic(name)          std::atomic_##name
#define  palloc_memory_order(name)    std::memory_order_##name
#if (__cplusplus >= 202002L)      // c++20, see issue #571
 #define PALLOC_ATOMIC_VAR_INIT(x)    x
#elif !defined(ATOMIC_VAR_INIT)
 #define PALLOC_ATOMIC_VAR_INIT(x)    x
#else
 #define PALLOC_ATOMIC_VAR_INIT(x)    ATOMIC_VAR_INIT(x)
#endif
#elif defined(_MSC_VER)
// Use MSVC C wrapper for C11 atomics
#define  _Atomic(tp)              tp
#define  PALLOC_ATOMIC_VAR_INIT(x)    x
#define  palloc_atomic(name)          palloc_atomic_##name
#define  palloc_memory_order(name)    palloc_memory_order_##name
#else
// Use C11 atomics
#include <stdatomic.h>
#define  palloc_atomic(name)          atomic_##name
#define  palloc_memory_order(name)    memory_order_##name
#if (__STDC_VERSION__ >= 201710L) // c17, see issue #735
 #define PALLOC_ATOMIC_VAR_INIT(x)    x
#elif !defined(ATOMIC_VAR_INIT)
 #define PALLOC_ATOMIC_VAR_INIT(x)    x
#else
 #define PALLOC_ATOMIC_VAR_INIT(x)    ATOMIC_VAR_INIT(x)
#endif
#endif

// Various defines for all used memory orders in palloc
#define palloc_atomic_cas_weak(p,expected,desired,mem_success,mem_fail)  \
  palloc_atomic(compare_exchange_weak_explicit)(p,expected,desired,mem_success,mem_fail)

#define palloc_atomic_cas_strong(p,expected,desired,mem_success,mem_fail)  \
  palloc_atomic(compare_exchange_strong_explicit)(p,expected,desired,mem_success,mem_fail)

#define palloc_atomic_load_acquire(p)                palloc_atomic(load_explicit)(p,palloc_memory_order(acquire))
#define palloc_atomic_load_relaxed(p)                palloc_atomic(load_explicit)(p,palloc_memory_order(relaxed))
#define palloc_atomic_store_release(p,x)             palloc_atomic(store_explicit)(p,x,palloc_memory_order(release))
#define palloc_atomic_store_relaxed(p,x)             palloc_atomic(store_explicit)(p,x,palloc_memory_order(relaxed))
#define palloc_atomic_exchange_relaxed(p,x)          palloc_atomic(exchange_explicit)(p,x,palloc_memory_order(relaxed))
#define palloc_atomic_exchange_release(p,x)          palloc_atomic(exchange_explicit)(p,x,palloc_memory_order(release))
#define palloc_atomic_exchange_acq_rel(p,x)          palloc_atomic(exchange_explicit)(p,x,palloc_memory_order(acq_rel))
#define palloc_atomic_cas_weak_release(p,exp,des)    palloc_atomic_cas_weak(p,exp,des,palloc_memory_order(release),palloc_memory_order(relaxed))
#define palloc_atomic_cas_weak_acq_rel(p,exp,des)    palloc_atomic_cas_weak(p,exp,des,palloc_memory_order(acq_rel),palloc_memory_order(acquire))
#define palloc_atomic_cas_strong_release(p,exp,des)  palloc_atomic_cas_strong(p,exp,des,palloc_memory_order(release),palloc_memory_order(relaxed))
#define palloc_atomic_cas_strong_acq_rel(p,exp,des)  palloc_atomic_cas_strong(p,exp,des,palloc_memory_order(acq_rel),palloc_memory_order(acquire))

#define palloc_atomic_add_relaxed(p,x)               palloc_atomic(fetch_add_explicit)(p,x,palloc_memory_order(relaxed))
#define palloc_atomic_sub_relaxed(p,x)               palloc_atomic(fetch_sub_explicit)(p,x,palloc_memory_order(relaxed))
#define palloc_atomic_add_acq_rel(p,x)               palloc_atomic(fetch_add_explicit)(p,x,palloc_memory_order(acq_rel))
#define palloc_atomic_sub_acq_rel(p,x)               palloc_atomic(fetch_sub_explicit)(p,x,palloc_memory_order(acq_rel))
#define palloc_atomic_and_acq_rel(p,x)               palloc_atomic(fetch_and_explicit)(p,x,palloc_memory_order(acq_rel))
#define palloc_atomic_or_acq_rel(p,x)                palloc_atomic(fetch_or_explicit)(p,x,palloc_memory_order(acq_rel))

#define palloc_atomic_increment_relaxed(p)           palloc_atomic_add_relaxed(p,(uintptr_t)1)
#define palloc_atomic_decrement_relaxed(p)           palloc_atomic_sub_relaxed(p,(uintptr_t)1)
#define palloc_atomic_increment_acq_rel(p)           palloc_atomic_add_acq_rel(p,(uintptr_t)1)
#define palloc_atomic_decrement_acq_rel(p)           palloc_atomic_sub_acq_rel(p,(uintptr_t)1)

static inline void palloc_atomic_yield(void);
static inline intptr_t palloc_atomic_addi(_Atomic(intptr_t)*p, intptr_t add);
static inline intptr_t palloc_atomic_subi(_Atomic(intptr_t)*p, intptr_t sub);


#if defined(__cplusplus) || !defined(_MSC_VER)

// In C++/C11 atomics we have polymorphic atomics so can use the typed `ptr` variants (where `tp` is the type of atomic value)
// We use these macros so we can provide a typed wrapper in MSVC in C compilation mode as well
#define palloc_atomic_load_ptr_acquire(tp,p)                palloc_atomic_load_acquire(p)
#define palloc_atomic_load_ptr_relaxed(tp,p)                palloc_atomic_load_relaxed(p)

// In C++ we need to add casts to help resolve templates if NULL is passed
#if defined(__cplusplus)
#define palloc_atomic_store_ptr_release(tp,p,x)             palloc_atomic_store_release(p,(tp*)x)
#define palloc_atomic_store_ptr_relaxed(tp,p,x)             palloc_atomic_store_relaxed(p,(tp*)x)
#define palloc_atomic_cas_ptr_weak_release(tp,p,exp,des)    palloc_atomic_cas_weak_release(p,exp,(tp*)des)
#define palloc_atomic_cas_ptr_weak_acq_rel(tp,p,exp,des)    palloc_atomic_cas_weak_acq_rel(p,exp,(tp*)des)
#define palloc_atomic_cas_ptr_strong_release(tp,p,exp,des)  palloc_atomic_cas_strong_release(p,exp,(tp*)des)
#define palloc_atomic_cas_ptr_strong_acq_rel(tp,p,exp,des)  palloc_atomic_cas_strong_acq_rel(p,exp,(tp*)des)
#define palloc_atomic_exchange_ptr_relaxed(tp,p,x)          palloc_atomic_exchange_relaxed(p,(tp*)x)
#define palloc_atomic_exchange_ptr_release(tp,p,x)          palloc_atomic_exchange_release(p,(tp*)x)
#define palloc_atomic_exchange_ptr_acq_rel(tp,p,x)          palloc_atomic_exchange_acq_rel(p,(tp*)x)
#else
#define palloc_atomic_store_ptr_release(tp,p,x)             palloc_atomic_store_release(p,x)
#define palloc_atomic_store_ptr_relaxed(tp,p,x)             palloc_atomic_store_relaxed(p,x)
#define palloc_atomic_cas_ptr_weak_release(tp,p,exp,des)    palloc_atomic_cas_weak_release(p,exp,des)
#define palloc_atomic_cas_ptr_weak_acq_rel(tp,p,exp,des)    palloc_atomic_cas_weak_acq_rel(p,exp,des)
#define palloc_atomic_cas_ptr_strong_release(tp,p,exp,des)  palloc_atomic_cas_strong_release(p,exp,des)
#define palloc_atomic_cas_ptr_strong_acq_rel(tp,p,exp,des)  palloc_atomic_cas_strong_acq_rel(p,exp,des)
#define palloc_atomic_exchange_ptr_relaxed(tp,p,x)          palloc_atomic_exchange_relaxed(p,x)
#define palloc_atomic_exchange_ptr_release(tp,p,x)          palloc_atomic_exchange_release(p,x)
#define palloc_atomic_exchange_ptr_acq_rel(tp,p,x)          palloc_atomic_exchange_acq_rel(p,x)
#endif

// These are used by the statistics
static inline int64_t palloc_atomic_addi64_relaxed(volatile int64_t* p, int64_t add) {
  return palloc_atomic(fetch_add_explicit)((_Atomic(int64_t)*)p, add, palloc_memory_order(relaxed));
}
static inline void palloc_atomic_void_addi64_relaxed(volatile int64_t* p, const volatile int64_t* padd) {
  const int64_t add = palloc_atomic_load_relaxed((_Atomic(int64_t)*)padd);
  if (add != 0) {
    palloc_atomic(fetch_add_explicit)((_Atomic(int64_t)*)p, add, palloc_memory_order(relaxed));
  }
}
static inline void palloc_atomic_maxi64_relaxed(volatile int64_t* p, int64_t x) {
  int64_t current = palloc_atomic_load_relaxed((_Atomic(int64_t)*)p);
  while (current < x && !palloc_atomic_cas_weak_release((_Atomic(int64_t)*)p, &current, x)) { /* nothing */ };
}

// Used by timers
#define palloc_atomic_loadi64_acquire(p)            palloc_atomic(load_explicit)(p,palloc_memory_order(acquire))
#define palloc_atomic_loadi64_relaxed(p)            palloc_atomic(load_explicit)(p,palloc_memory_order(relaxed))
#define palloc_atomic_storei64_release(p,x)         palloc_atomic(store_explicit)(p,x,palloc_memory_order(release))
#define palloc_atomic_storei64_relaxed(p,x)         palloc_atomic(store_explicit)(p,x,palloc_memory_order(relaxed))

#define palloc_atomic_casi64_strong_acq_rel(p,e,d)  palloc_atomic_cas_strong_acq_rel(p,e,d)
#define palloc_atomic_addi64_acq_rel(p,i)           palloc_atomic_add_acq_rel(p,i)


#elif defined(_MSC_VER)

// Legacy MSVC plain C compilation wrapper that uses Interlocked operations to model C11 atomics.
#include <intrin.h>
#ifdef _WIN64
typedef LONG64   msc_intptr_t;
#define PALLOC_64(f) f##64
#else
typedef LONG     msc_intptr_t;
#define PALLOC_64(f) f
#endif

typedef enum palloc_memory_order_e {
  palloc_memory_order_relaxed,
  palloc_memory_order_consume,
  palloc_memory_order_acquire,
  palloc_memory_order_release,
  palloc_memory_order_acq_rel,
  palloc_memory_order_seq_cst
} palloc_memory_order;

static inline uintptr_t palloc_atomic_fetch_add_explicit(_Atomic(uintptr_t)*p, uintptr_t add, palloc_memory_order mo) {
  (void)(mo);
  return (uintptr_t)PALLOC_64(_InterlockedExchangeAdd)((volatile msc_intptr_t*)p, (msc_intptr_t)add);
}
static inline uintptr_t palloc_atomic_fetch_sub_explicit(_Atomic(uintptr_t)*p, uintptr_t sub, palloc_memory_order mo) {
  (void)(mo);
  return (uintptr_t)PALLOC_64(_InterlockedExchangeAdd)((volatile msc_intptr_t*)p, -((msc_intptr_t)sub));
}
static inline uintptr_t palloc_atomic_fetch_and_explicit(_Atomic(uintptr_t)*p, uintptr_t x, palloc_memory_order mo) {
  (void)(mo);
  return (uintptr_t)PALLOC_64(_InterlockedAnd)((volatile msc_intptr_t*)p, (msc_intptr_t)x);
}
static inline uintptr_t palloc_atomic_fetch_or_explicit(_Atomic(uintptr_t)*p, uintptr_t x, palloc_memory_order mo) {
  (void)(mo);
  return (uintptr_t)PALLOC_64(_InterlockedOr)((volatile msc_intptr_t*)p, (msc_intptr_t)x);
}
static inline bool palloc_atomic_compare_exchange_strong_explicit(_Atomic(uintptr_t)*p, uintptr_t* expected, uintptr_t desired, palloc_memory_order mo1, palloc_memory_order mo2) {
  (void)(mo1); (void)(mo2);
  uintptr_t read = (uintptr_t)PALLOC_64(_InterlockedCompareExchange)((volatile msc_intptr_t*)p, (msc_intptr_t)desired, (msc_intptr_t)(*expected));
  if (read == *expected) {
    return true;
  }
  else {
    *expected = read;
    return false;
  }
}
static inline bool palloc_atomic_compare_exchange_weak_explicit(_Atomic(uintptr_t)*p, uintptr_t* expected, uintptr_t desired, palloc_memory_order mo1, palloc_memory_order mo2) {
  return palloc_atomic_compare_exchange_strong_explicit(p, expected, desired, mo1, mo2);
}
static inline uintptr_t palloc_atomic_exchange_explicit(_Atomic(uintptr_t)*p, uintptr_t exchange, palloc_memory_order mo) {
  (void)(mo);
  return (uintptr_t)PALLOC_64(_InterlockedExchange)((volatile msc_intptr_t*)p, (msc_intptr_t)exchange);
}
static inline void palloc_atomic_thread_fence(palloc_memory_order mo) {
  (void)(mo);
  _Atomic(uintptr_t) x = 0;
  palloc_atomic_exchange_explicit(&x, 1, mo);
}
static inline uintptr_t palloc_atomic_load_explicit(_Atomic(uintptr_t) const* p, palloc_memory_order mo) {
  (void)(mo);
#if defined(_M_IX86) || defined(_M_X64)
  return *p;
#else
  uintptr_t x = *p;
  if (mo > palloc_memory_order_relaxed) {
    while (!palloc_atomic_compare_exchange_weak_explicit((_Atomic(uintptr_t)*)p, &x, x, mo, palloc_memory_order_relaxed)) { /* nothing */ };
  }
  return x;
#endif
}
static inline void palloc_atomic_store_explicit(_Atomic(uintptr_t)*p, uintptr_t x, palloc_memory_order mo) {
  (void)(mo);
#if defined(_M_IX86) || defined(_M_X64)
  *p = x;
#else
  palloc_atomic_exchange_explicit(p, x, mo);
#endif
}
static inline int64_t palloc_atomic_loadi64_explicit(_Atomic(int64_t)*p, palloc_memory_order mo) {
  (void)(mo);
#if defined(_M_X64)
  return *p;
#else
  int64_t old = *p;
  int64_t x = old;
  while ((old = InterlockedCompareExchange64(p, x, old)) != x) {
    x = old;
  }
  return x;
#endif
}
static inline void palloc_atomic_storei64_explicit(_Atomic(int64_t)*p, int64_t x, palloc_memory_order mo) {
  (void)(mo);
#if defined(x_M_IX86) || defined(_M_X64)
  *p = x;
#else
  InterlockedExchange64(p, x);
#endif
}

// These are used by the statistics
static inline int64_t palloc_atomic_addi64_relaxed(volatile _Atomic(int64_t)*p, int64_t add) {
#ifdef _WIN64
  return (int64_t)palloc_atomic_addi((int64_t*)p, add);
#else
  int64_t current;
  int64_t sum;
  do {
    current = *p;
    sum = current + add;
  } while (_InterlockedCompareExchange64(p, sum, current) != current);
  return current;
#endif
}
static inline void palloc_atomic_void_addi64_relaxed(volatile int64_t* p, const volatile int64_t* padd) {
  const int64_t add = *padd;
  if (add != 0) {
    palloc_atomic_addi64_relaxed((volatile _Atomic(int64_t)*)p, add);
  }
}

static inline void palloc_atomic_maxi64_relaxed(volatile _Atomic(int64_t)*p, int64_t x) {
  int64_t current;
  do {
    current = *p;
  } while (current < x && _InterlockedCompareExchange64(p, x, current) != current);
}

static inline void palloc_atomic_addi64_acq_rel(volatile _Atomic(int64_t*)p, int64_t i) {
  palloc_atomic_addi64_relaxed(p, i);
}

static inline bool palloc_atomic_casi64_strong_acq_rel(volatile _Atomic(int64_t*)p, int64_t* exp, int64_t des) {
  int64_t read = _InterlockedCompareExchange64(p, des, *exp);
  if (read == *exp) {
    return true;
  }
  else {
    *exp = read;
    return false;
  }
}

// The pointer macros cast to `uintptr_t`.
#define palloc_atomic_load_ptr_acquire(tp,p)                (tp*)palloc_atomic_load_acquire((_Atomic(uintptr_t)*)(p))
#define palloc_atomic_load_ptr_relaxed(tp,p)                (tp*)palloc_atomic_load_relaxed((_Atomic(uintptr_t)*)(p))
#define palloc_atomic_store_ptr_release(tp,p,x)             palloc_atomic_store_release((_Atomic(uintptr_t)*)(p),(uintptr_t)(x))
#define palloc_atomic_store_ptr_relaxed(tp,p,x)             palloc_atomic_store_relaxed((_Atomic(uintptr_t)*)(p),(uintptr_t)(x))
#define palloc_atomic_cas_ptr_weak_release(tp,p,exp,des)    palloc_atomic_cas_weak_release((_Atomic(uintptr_t)*)(p),(uintptr_t*)exp,(uintptr_t)des)
#define palloc_atomic_cas_ptr_weak_acq_rel(tp,p,exp,des)    palloc_atomic_cas_weak_acq_rel((_Atomic(uintptr_t)*)(p),(uintptr_t*)exp,(uintptr_t)des)
#define palloc_atomic_cas_ptr_strong_release(tp,p,exp,des)  palloc_atomic_cas_strong_release((_Atomic(uintptr_t)*)(p),(uintptr_t*)exp,(uintptr_t)des)
#define palloc_atomic_cas_ptr_strong_acq_rel(tp,p,exp,des)  palloc_atomic_cas_strong_acq_rel((_Atomic(uintptr_t)*)(p),(uintptr_t*)exp,(uintptr_t)des)
#define palloc_atomic_exchange_ptr_relaxed(tp,p,x)          (tp*)palloc_atomic_exchange_relaxed((_Atomic(uintptr_t)*)(p),(uintptr_t)x)
#define palloc_atomic_exchange_ptr_release(tp,p,x)          (tp*)palloc_atomic_exchange_release((_Atomic(uintptr_t)*)(p),(uintptr_t)x)
#define palloc_atomic_exchange_ptr_acq_rel(tp,p,x)          (tp*)palloc_atomic_exchange_acq_rel((_Atomic(uintptr_t)*)(p),(uintptr_t)x)

#define palloc_atomic_loadi64_acquire(p)    palloc_atomic(loadi64_explicit)(p,palloc_memory_order(acquire))
#define palloc_atomic_loadi64_relaxed(p)    palloc_atomic(loadi64_explicit)(p,palloc_memory_order(relaxed))
#define palloc_atomic_storei64_release(p,x) palloc_atomic(storei64_explicit)(p,x,palloc_memory_order(release))
#define palloc_atomic_storei64_relaxed(p,x) palloc_atomic(storei64_explicit)(p,x,palloc_memory_order(relaxed))


#endif


// Atomically add a signed value; returns the previous value.
static inline intptr_t palloc_atomic_addi(_Atomic(intptr_t)*p, intptr_t add) {
  return (intptr_t)palloc_atomic_add_acq_rel((_Atomic(uintptr_t)*)p, (uintptr_t)add);
}

// Atomically subtract a signed value; returns the previous value.
static inline intptr_t palloc_atomic_subi(_Atomic(intptr_t)*p, intptr_t sub) {
  return (intptr_t)palloc_atomic_addi(p, -sub);
}


// ----------------------------------------------------------------------
// Once and Guard
// ----------------------------------------------------------------------

typedef _Atomic(uintptr_t) palloc_atomic_once_t;

// Returns true only on the first invocation
static inline bool palloc_atomic_once( palloc_atomic_once_t* once ) {
  if (palloc_atomic_load_relaxed(once) != 0) return false;     // quick test
  uintptr_t expected = 0;
  return palloc_atomic_cas_strong_acq_rel(once, &expected, (uintptr_t)1); // try to set to 1
}

typedef _Atomic(uintptr_t) palloc_atomic_guard_t;

// Allows only one thread to execute at a time
#define palloc_atomic_guard(guard) \
  uintptr_t _palloc_guard_expected = 0; \
  for(bool _palloc_guard_once = true; \
      _palloc_guard_once && palloc_atomic_cas_strong_acq_rel(guard,&_palloc_guard_expected,(uintptr_t)1); \
      (palloc_atomic_store_release(guard,(uintptr_t)0), _palloc_guard_once = false) )



// ----------------------------------------------------------------------
// Yield
// ----------------------------------------------------------------------

#if defined(__cplusplus)
#include <thread>
static inline void palloc_atomic_yield(void) {
  std::this_thread::yield();
}
#elif defined(_WIN32)
static inline void palloc_atomic_yield(void) {
  YieldProcessor();
}
#elif defined(__SSE2__)
#include <emmintrin.h>
static inline void palloc_atomic_yield(void) {
  _mm_pause();
}
#elif (defined(__GNUC__) || defined(__clang__)) && \
      (defined(__x86_64__) || defined(__i386__) || \
       defined(__aarch64__) || defined(__arm__) || \
       defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__POWERPC__))
#if defined(__x86_64__) || defined(__i386__)
static inline void palloc_atomic_yield(void) {
  __asm__ volatile ("pause" ::: "memory");
}
#elif defined(__aarch64__)
static inline void palloc_atomic_yield(void) {
  __asm__ volatile("wfe");
}
#elif defined(__arm__)
#if __ARM_ARCH >= 7
static inline void palloc_atomic_yield(void) {
  __asm__ volatile("yield" ::: "memory");
}
#else
static inline void palloc_atomic_yield(void) {
  __asm__ volatile ("nop" ::: "memory");
}
#endif
#elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__POWERPC__)
#ifdef __APPLE__
static inline void palloc_atomic_yield(void) {
  __asm__ volatile ("or r27,r27,r27" ::: "memory");
}
#else
static inline void palloc_atomic_yield(void) {
  __asm__ __volatile__ ("or 27,27,27" ::: "memory");
}
#endif
#endif
#elif defined(__sun)
// Fallback for other archs
#include <synch.h>
static inline void palloc_atomic_yield(void) {
  smt_pause();
}
#elif defined(__wasi__)
#include <sched.h>
static inline void palloc_atomic_yield(void) {
  sched_yield();
}
#else
#include <unistd.h>
static inline void palloc_atomic_yield(void) {
  sleep(0);
}
#endif


// ----------------------------------------------------------------------
// Locks 
// These do not have to be recursive and should be light-weight 
// in-process only locks. Only used for reserving arena's and to 
// maintain the abandoned list.
// ----------------------------------------------------------------------
#if _MSC_VER
#pragma warning(disable:26110)  // unlock with holding lock
#endif

#define palloc_lock(lock)    for(bool _go = (palloc_lock_acquire(lock),true); _go; (palloc_lock_release(lock), _go=false) )

#if defined(_WIN32)

#if 1
#define palloc_lock_t  SRWLOCK   // slim reader-writer lock

static inline bool palloc_lock_try_acquire(palloc_lock_t* lock) {
  return TryAcquireSRWLockExclusive(lock);
}
static inline void palloc_lock_acquire(palloc_lock_t* lock) {
  AcquireSRWLockExclusive(lock);
}
static inline void palloc_lock_release(palloc_lock_t* lock) {
  ReleaseSRWLockExclusive(lock);
}
static inline void palloc_lock_init(palloc_lock_t* lock) {
  InitializeSRWLock(lock);
}
static inline void palloc_lock_done(palloc_lock_t* lock) {
  (void)(lock);
}

#else
#define palloc_lock_t  CRITICAL_SECTION

static inline bool palloc_lock_try_acquire(palloc_lock_t* lock) {
  return TryEnterCriticalSection(lock);
}
static inline void palloc_lock_acquire(palloc_lock_t* lock) {
  EnterCriticalSection(lock);
}
static inline void palloc_lock_release(palloc_lock_t* lock) {
  LeaveCriticalSection(lock);
}
static inline void palloc_lock_init(palloc_lock_t* lock) {
  InitializeCriticalSection(lock);
}
static inline void palloc_lock_done(palloc_lock_t* lock) {
  DeleteCriticalSection(lock);
}

#endif

#elif defined(PALLOC_USE_PTHREADS)

void _palloc_error_message(int err, const char* fmt, ...);

#define palloc_lock_t  pthread_mutex_t

static inline bool palloc_lock_try_acquire(palloc_lock_t* lock) {
  return (pthread_mutex_trylock(lock) == 0);
}
static inline void palloc_lock_acquire(palloc_lock_t* lock) {
  const int err = pthread_mutex_lock(lock);
  if (err != 0) {
    _palloc_error_message(err, "internal error: lock cannot be acquired\n");
  }
}
static inline void palloc_lock_release(palloc_lock_t* lock) {
  pthread_mutex_unlock(lock);
}
static inline void palloc_lock_init(palloc_lock_t* lock) {
  pthread_mutex_init(lock, NULL);
}
static inline void palloc_lock_done(palloc_lock_t* lock) {
  pthread_mutex_destroy(lock);
}

#elif defined(__cplusplus)

#include <mutex>
#define palloc_lock_t  std::mutex

static inline bool palloc_lock_try_acquire(palloc_lock_t* lock) {
  return lock->try_lock();
}
static inline void palloc_lock_acquire(palloc_lock_t* lock) {
  lock->lock();
}
static inline void palloc_lock_release(palloc_lock_t* lock) {
  lock->unlock();
}
static inline void palloc_lock_init(palloc_lock_t* lock) {
  (void)(lock);
}
static inline void palloc_lock_done(palloc_lock_t* lock) {
  (void)(lock);
}

#else

// fall back to poor man's locks.
// this should only be the case in a single-threaded environment (like __wasi__)

#define palloc_lock_t  _Atomic(uintptr_t)

static inline bool palloc_lock_try_acquire(palloc_lock_t* lock) {
  uintptr_t expected = 0;
  return palloc_atomic_cas_strong_acq_rel(lock, &expected, (uintptr_t)1);
}
static inline void palloc_lock_acquire(palloc_lock_t* lock) {
  for (int i = 0; i < 1000; i++) {  // for at most 1000 tries?
    if (palloc_lock_try_acquire(lock)) return;
    palloc_atomic_yield();
  }
}
static inline void palloc_lock_release(palloc_lock_t* lock) {
  palloc_atomic_store_release(lock, (uintptr_t)0);
}
static inline void palloc_lock_init(palloc_lock_t* lock) {
  palloc_lock_release(lock);
}
static inline void palloc_lock_done(palloc_lock_t* lock) {
  (void)(lock);
}

#endif


#endif // __PALLOC_ATOMIC_H
