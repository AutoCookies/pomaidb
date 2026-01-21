#pragma once

#include <cstdint>
#include <cstddef>

namespace pomai::ai::atomic_utils
{

    // --- UINT64 Operations ---

    inline uint64_t atomic_load_u64(const uint64_t *ptr) noexcept
    {
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u64(uint64_t *ptr, uint64_t v) noexcept
    {
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    inline bool atomic_compare_exchange_u64(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept
    {
        // strong CAS, seq_cst
        return __atomic_compare_exchange_n(ptr, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }

    // --- UINT32 Operations ---

    inline uint32_t atomic_load_u32(const uint32_t *ptr) noexcept
    {
        return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    inline void atomic_store_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        __atomic_store_n(ptr, v, __ATOMIC_RELEASE);
    }

    inline uint32_t atomic_fetch_add_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        return __atomic_fetch_add(ptr, v, __ATOMIC_SEQ_CST);
    }

    inline uint32_t atomic_fetch_sub_u32(uint32_t *ptr, uint32_t v) noexcept
    {
        return __atomic_fetch_sub(ptr, v, __ATOMIC_SEQ_CST);
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