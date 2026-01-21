#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <cassert>

#include "src/core/config.h"

namespace pomai::ai::soa
{
    using Layout = pomai::config::IdsBlockLayout;

    struct IdEntry
    {
        static constexpr uint64_t TAG_SHIFT = Layout::TAG_SHIFT;
        static constexpr uint64_t TAG_MASK = Layout::TAG_MASK;
        static constexpr uint64_t PAYLOAD_MASK = Layout::PAYLOAD_MASK;

        static constexpr uint64_t TAG_LOCAL = 0x0ULL << TAG_SHIFT;
        static constexpr uint64_t TAG_REMOTE = 0x1ULL << TAG_SHIFT;
        static constexpr uint64_t TAG_LABEL = 0x2ULL << TAG_SHIFT;
        static constexpr uint64_t TAG_RESERVED = 0x3ULL << TAG_SHIFT;

        [[nodiscard]] static constexpr bool fits_payload(uint64_t val) noexcept
        {
            return (val & ~PAYLOAD_MASK) == 0;
        }

        [[nodiscard]] static constexpr uint64_t pack_local_offset(uint64_t offset) noexcept
        {
            // Mask to payload bits instead of asserting to avoid aborts on overflow.
            return TAG_LOCAL | (offset & PAYLOAD_MASK);
        }

        [[nodiscard]] static constexpr uint64_t pack_remote_id(uint64_t remote_id) noexcept
        {
            return TAG_REMOTE | (remote_id & PAYLOAD_MASK);
        }

        [[nodiscard]] static constexpr uint64_t pack_label(uint64_t label) noexcept
        {
            return TAG_LABEL | (label & PAYLOAD_MASK);
        }

        [[nodiscard]] static constexpr uint64_t tag_of(uint64_t v) noexcept
        {
            return v & TAG_MASK;
        }

        [[nodiscard]] static constexpr uint64_t payload_of(uint64_t v) noexcept
        {
            return v & PAYLOAD_MASK;
        }
        [[nodiscard]] static bool try_pack_local_offset(uint64_t offset, uint64_t &out) noexcept;
        [[nodiscard]] static bool try_pack_remote_id(uint64_t remote_id, uint64_t &out) noexcept;
        [[nodiscard]] static bool try_pack_label(uint64_t label, uint64_t &out) noexcept;
        static void atomic_store(uint64_t *ptr, uint64_t val) noexcept;
        [[nodiscard]] static uint64_t atomic_load(const uint64_t *ptr) noexcept;
        [[nodiscard]] static bool atomic_compare_exchange(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept;
    };

    static_assert(sizeof(uint64_t) == 8, "Platform requires 8-byte uint64_t");
    static_assert(Layout::PAYLOAD_BITS + Layout::TAG_BITS == 64, "Bits layout must sum to 64");

} // namespace pomai::ai::soa