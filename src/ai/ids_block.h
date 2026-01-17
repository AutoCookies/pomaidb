#pragma once
/*
 * src/ai/ids_block.h
 * Fixed: Added missing try_pack_* declarations.
 */

#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "src/core/config.h"

namespace pomai::ai::soa
{
    using Layout = pomai::config::IdsBlockLayout;

    struct IdEntry
    {
        // Use constants from centralized config
        static constexpr uint64_t TAG_SHIFT = Layout::TAG_SHIFT;
        static constexpr uint64_t TAG_MASK = Layout::TAG_MASK;
        static constexpr uint64_t PAYLOAD_MASK = Layout::PAYLOAD_MASK;

        // Tag definitions
        static constexpr uint64_t TAG_LOCAL = 0x0ULL << TAG_SHIFT;
        static constexpr uint64_t TAG_REMOTE = 0x1ULL << TAG_SHIFT;
        static constexpr uint64_t TAG_LABEL = 0x2ULL << TAG_SHIFT;
        static constexpr uint64_t TAG_RESERVED = 0x3ULL << TAG_SHIFT;

        // --- Packing Helpers (Inline) ---
        static inline uint64_t pack_local_offset(uint64_t offset) noexcept
        {
            return TAG_LOCAL | (offset & PAYLOAD_MASK);
        }

        static inline uint64_t pack_remote_id(uint64_t remote_id) noexcept
        {
            return TAG_REMOTE | (remote_id & PAYLOAD_MASK);
        }

        static inline uint64_t pack_label(uint64_t label) noexcept
        {
            return TAG_LABEL | (label & PAYLOAD_MASK);
        }

        // --- Safe Packing (Declarations Added Here) ---
        static bool try_pack_local_offset(uint64_t offset, uint64_t &out) noexcept;
        static bool try_pack_remote_id(uint64_t remote_id, uint64_t &out) noexcept;
        static bool try_pack_label(uint64_t label, uint64_t &out) noexcept;

        // --- Utils ---
        static inline bool fits_payload(uint64_t val) noexcept
        {
            return (val & ~PAYLOAD_MASK) == 0;
        }

        static inline uint64_t tag_of(uint64_t v) noexcept
        {
            return v & TAG_MASK;
        }

        static inline uint64_t payload_of(uint64_t v) noexcept
        {
            return v & PAYLOAD_MASK;
        }
    };

    static_assert(sizeof(uint64_t) == 8, "Platform requires 8-byte uint64_t");
    static_assert(Layout::PAYLOAD_BITS + Layout::TAG_BITS == 64, "Bits must sum to 64");

    // Atomic helpers (declarations)
    void atomic_store_entry(uint64_t *ptr, uint64_t v) noexcept;
    uint64_t atomic_load_entry(const uint64_t *ptr) noexcept;
    bool atomic_compare_exchange_entry(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept;

} // namespace pomai::ai::soa