/*
 * src/ai/atomic_utils.h
 *
 * Portable atomic helpers for values that may live in mmap'd memory.
 *
 * Exposes:
 * - uint64_t atomic_load_u64(const uint64_t*)
 * - void     atomic_store_u64(uint64_t*, uint64_t)
 * - bool     atomic_compare_exchange_u64(uint64_t*, uint64_t&, uint64_t)  <-- ADDED
 * - uint32_t atomic_load_u32(const uint32_t*)
 * - void     atomic_store_u32(uint32_t*, uint32_t)
 * - uint32_t atomic_fetch_or_u32(uint32_t*, uint32_t)
 * - uint32_t atomic_fetch_and_u32(uint32_t*, uint32_t)
 *
 * Implementation note:
 * - Use compiler __atomic builtins for correctness on shared mmap'd memory.
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace pomai::ai::atomic_utils
{

    // Always use GCC/Clang __atomic builtins which work for mmap'd memory.

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    // [ADDED] Missing CAS helper required by ids_block.cc
    // Returns true if successful (ptr == expected), false otherwise (expected updated).
    inline bool atomic_compare_exchange_u64(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept
    {
        // __atomic_compare_exchange_n(ptr, expected, desired, weak, success_memorder, failure_memorder)
        // weak=false (strong CAS), SEQ_CST for safety.
        return __atomic_compare_exchange_n(ptr, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
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

} // namespace pomai::ai::atomic_utils