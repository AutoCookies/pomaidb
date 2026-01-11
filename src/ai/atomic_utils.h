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
 *  - Prefer std::atomic_ref (C++20) when available and *properly aligned*.
 *  - If the target address is not aligned for atomic_ref, fall back to
 *    GCC/Clang __atomic builtins which handle unaligned cases on common CPUs.
 *
 * Ordering:
 *  - Loads use Acquire, stores use Release (suitable for publish/unpublish).
 *  - Fetch-or / fetch-and use SEQ_CST (conservative for read-modify).
 *
 * This header avoids asserting on alignment (some mmap'd layouts are packed).
 * When std::atomic_ref would be UB due to alignment we transparently use
 * compiler builtins instead.
 */

#pragma once

#include <cstdint>
#include <atomic>
#include <cstddef>

namespace pomai::ai::atomic_utils
{

    inline bool is_aligned(const void *ptr, size_t align) noexcept
    {
        // align is power-of-two in our uses; safe to compute mask this way
        return (reinterpret_cast<uintptr_t>(ptr) & (align - 1)) == 0;
    }

#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201811L)

    // Use atomic_ref when the pointer is aligned for the type; otherwise fall back
    // to __atomic builtins below.

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        if (is_aligned(ptr, alignof(uint64_t)))
        {
            std::atomic_ref<const uint64_t> aref(*ptr);
            return aref.load(std::memory_order_acquire);
        }
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        if (is_aligned(ptr, alignof(uint64_t)))
        {
            std::atomic_ref<uint64_t> aref(*ptr);
            aref.store(v, std::memory_order_release);
            return;
        }
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    inline uint32_t atomic_load_u32(const uint32_t *ptr) noexcept
    {
        if (is_aligned(ptr, alignof(uint32_t)))
        {
            std::atomic_ref<const uint32_t> aref(*ptr);
            return aref.load(std::memory_order_acquire);
        }
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        if (is_aligned(ptr, alignof(uint32_t)))
        {
            std::atomic_ref<uint32_t> aref(*ptr);
            aref.store(v, std::memory_order_release);
            return;
        }
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    inline uint32_t atomic_fetch_or_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        if (is_aligned(ptr, alignof(uint32_t)))
        {
            std::atomic_ref<uint32_t> aref(*ptr);
            return aref.fetch_or(v, std::memory_order_seq_cst);
        }
        return __atomic_fetch_or(ptr, v, __ATOMIC_SEQ_CST);
    }

    inline uint32_t atomic_fetch_and_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        if (is_aligned(ptr, alignof(uint32_t)))
        {
            std::atomic_ref<uint32_t> aref(*ptr);
            return aref.fetch_and(v, std::memory_order_seq_cst);
        }
        return __atomic_fetch_and(ptr, v, __ATOMIC_SEQ_CST);
    }

#else

    // No atomic_ref available; use compiler builtins (they support unaligned memory on most targets).
    // Use Acquire/Release for loads/stores and SeqCst for RMW.

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