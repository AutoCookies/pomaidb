/*
 * src/ai/atomic_utils.h
 *
 * Small portability helpers to perform atomic load/store on integer values
 * that may live in mmap'd memory.
 *
 * Updates:
 * - Added uint32_t support (atomic_load_u32, atomic_store_u32) needed for PPE flags.
 * - Uses std::atomic_ref (C++20) if available for cleanest semantics.
 * - Falls back to GCC/Clang __atomic builtins to ensure correct Acquire/Release ordering.
 */

#pragma once

#include <cstdint>
#include <atomic>

namespace pomai::ai::atomic_utils
{

#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201811L)
    // -------------------------------------------------------------------------
    // C++20 Path: Use std::atomic_ref
    // -------------------------------------------------------------------------

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

#else
    // -------------------------------------------------------------------------
    // Fallback Path: Use GCC/Clang __atomic builtins
    // We strictly prefer these over volatile to guarantee ordering barriers.
    // -------------------------------------------------------------------------

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

#endif

} // namespace pomai::ai::atomic_utils