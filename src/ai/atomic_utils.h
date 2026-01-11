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
 * Implementation note:
 *  - Use compiler __atomic builtins for correctness on shared mmap'd memory
 *    and when alignment may be unusual. These builtins are robust on GCC/Clang
 *    targets and handle unaligned accesses where necessary.
 *
 * Ordering:
 *  - Loads use Acquire, stores use Release (suitable for publish/unpublish).
 *  - Fetch-or / fetch-and use SeqCst (conservative for read-modify).
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace pomai::ai::atomic_utils
{

    // Always use GCC/Clang __atomic builtins which work for mmap'd memory and
    // for unaligned addresses on common targets. These builtins give the
    // needed memory ordering and atomicity guarantees.
    //
    // We purposely avoid std::atomic_ref here because some std::atomic_ref
    // implementations in combination with mmap'd/externally-managed memory
    // can produce subtle platform-specific behaviors that lead to rare
    // transient inconsistencies in stress tests (observed in atomic_mmap_test).

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

} // namespace pomai::ai::atomic_utils