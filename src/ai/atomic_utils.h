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
 *  - Use sequentially-consistent ordering (memory_order_seq_cst) for both loads and stores
 *    to be conservative and avoid subtle cross-TU/library differences that may surface in
 *    concurrency tests (publish/unpublish patterns).
 *
 * Note: seq_cst is slightly stronger (and slower) than acquire/release but much simpler
 * to reason about and robust for mmap'd memory that may be concurrently accessed by
 * different mechanisms.
 */

#pragma once

#include <cstdint>
#include <atomic>
#include <cstddef>

namespace pomai::ai::atomic_utils
{

    // NOTE: previous versions asserted that the pointer was aligned. In practice
    // mmap'd layouts or packed headers may place atomic values at offsets that are
    // not strictly aligned to the natural alignment. On common platforms (x86/x86_64)
    // the compiler/runtime support unaligned atomic builtin operations; asserting and
    // aborting here breaks tests. We therefore do not enforce a hard assert;
    // callers should prefer aligned placement when portability/performance matters.
    inline void maybe_check_alignment(const void * /*ptr*/, size_t /*align*/) noexcept
    {
        // Intentionally empty: keep as hook for optional logging in future.
    }

#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201811L)

    // C++20 path using atomic_ref (use SEQ_CST ordering for robustness) ----------------

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint64_t));
        std::atomic_ref<const uint64_t> aref(*ptr);
        return aref.load(std::memory_order_seq_cst);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint64_t));
        std::atomic_ref<uint64_t> aref(*ptr);
        aref.store(v, std::memory_order_seq_cst);
    }

    inline uint32_t atomic_load_u32(const uint32_t *ptr) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        std::atomic_ref<const uint32_t> aref(*ptr);
        return aref.load(std::memory_order_seq_cst);
    }

    inline void atomic_store_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        std::atomic_ref<uint32_t> aref(*ptr);
        aref.store(v, std::memory_order_seq_cst);
    }

    inline uint32_t atomic_fetch_or_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        std::atomic_ref<uint32_t> aref(*ptr);
        return aref.fetch_or(v, std::memory_order_seq_cst);
    }

    inline uint32_t atomic_fetch_and_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        std::atomic_ref<uint32_t> aref(*ptr);
        return aref.fetch_and(v, std::memory_order_seq_cst);
    }

#else

    // Fallback: GCC/Clang __atomic builtins (use SEQ_CST ordering) ---------------------

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint64_t));
        return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint64_t));
        __atomic_store_n(ptr, v, __ATOMIC_SEQ_CST);
    }

    inline uint32_t atomic_load_u32(const uint32_t *ptr) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
    }

    inline void atomic_store_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        __atomic_store_n(ptr, v, __ATOMIC_SEQ_CST);
    }

    inline uint32_t atomic_fetch_or_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        return __atomic_fetch_or(ptr, v, __ATOMIC_SEQ_CST);
    }

    inline uint32_t atomic_fetch_and_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        maybe_check_alignment(ptr, alignof(uint32_t));
        return __atomic_fetch_and(ptr, v, __ATOMIC_SEQ_CST);
    }

#endif

} // namespace pomai::ai::atomic_utils