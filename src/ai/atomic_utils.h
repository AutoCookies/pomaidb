/*
 * src/ai/atomic_utils.h
 *
 * Portable atomic helpers for values that may live in mmap'd memory.
 *
 * Exposes:
 *  - uint64_t atomic_load_u64(const uint64_t*)
 *  - void     atomic_store_u64(uint64_t*, uint64_t)
 *  - uint32_t atomic_load_u32(const uint32_t*)
 *  - void     atomic_store_u32(uint32_t*, uint32_t)
 *  - uint32_t atomic_fetch_or_u32(uint32_t*, uint32_t)
 *  - uint32_t atomic_fetch_and_u32(uint32_t*, uint32_t)
 *
 * Implementation:
 *  - Prefer std::atomic_ref (C++20) when available.
 *  - Fallback to GCC/Clang __atomic builtins otherwise.
 *
 * Ordering:
 *  - loads use Acquire, stores use Release (sufficient for publish pattern:
 *      store(payload, Release); store(flag, Release);
 *      load(flag, Acquire); load(payload, Acquire);
 *    )
 *  - fetch_or/fetch_and use SEQ_CST to be conservative for read-modify operations.
 */
#pragma once

#include <cstdint>
#include <atomic>

namespace pomai::ai::atomic_utils
{

#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201811L)

    // C++20 path using atomic_ref ------------------------------------------------

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        std::atomic_ref<const uint64_t> aref(*ptr);
        return aref.load(std::memory_order_acquire);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        std::atomic_ref<uint64_t> aref(*ptr);
        aref.store(v, std::memory_order_release);
    }

    inline uint32_t atomic_load_u32(const uint32_t *ptr) noexcept
    {
        std::atomic_ref<const uint32_t> aref(*ptr);
        return aref.load(std::memory_order_acquire);
    }

    inline void atomic_store_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        std::atomic_ref<uint32_t> aref(*ptr);
        aref.store(v, std::memory_order_release);
    }

    inline uint32_t atomic_fetch_or_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        std::atomic_ref<uint32_t> aref(*ptr);
        return aref.fetch_or(v, std::memory_order_seq_cst);
    }

    inline uint32_t atomic_fetch_and_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        std::atomic_ref<uint32_t> aref(*ptr);
        return aref.fetch_and(v, std::memory_order_seq_cst);
    }

#else

    // Fallback: GCC/Clang __atomic builtins -------------------------------------

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    inline uint32_t atomic_load_u32(const uint32_t *ptr) noexcept
    {
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    inline uint32_t atomic_fetch_or_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        return __atomic_fetch_or(ptr, v, __ATOMIC_SEQ_CST);
    }

    inline uint32_t atomic_fetch_and_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        return __atomic_fetch_and(ptr, v, __ATOMIC_SEQ_CST);
    }

#endif

} // namespace pomai::ai::atomic_utils