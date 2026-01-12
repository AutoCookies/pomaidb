/*
 * src/ai/ids_block.h
 *
 * Compact, versioned helpers for the per-vector IDs/offsets block used by the
 * SoA mmap layout. Each vector has a single 8-byte (uint64_t) entry that
 * encodes one of:
 *   - a local arena offset (payload stored in PomaiArena blob region)
 *   - a remote segment id + offset (demoted blob on disk)
 *   - an external label id (application label)
 *
 * Encoding strategy (62 usable payload bits + 2 tag bits in MSBs):
 *   - tag 00 : local offset        => value format: (0b00 << 62) | (offset & PAYLOAD_MASK)
 *   - tag 01 : remote id           => value format: (0b01 << 62) | (remote_id & PAYLOAD_MASK)
 *   - tag 10 : external label id   => value format: (0b10 << 62) | (label & PAYLOAD_MASK)
 *   - tag 11 : reserved / unused
 *
 * This provides a compact, fixed-size representation (8 bytes per vector)
 * and avoids ambiguity between offsets and labels. Consumers must ensure the
 * original numeric values fit in 62 bits (practically always true).
 *
 * Threading:
 *   - The helpers here are simple pack/unpack functions. Atomicity on update
 *     is the caller's responsibility. For in-memory concurrent updates you can
 *     use std::atomic<uint64_t> or std::atomic_ref<uint64_t> on mapped memory.
 *
 * Usage:
 *   uint64_t entry = IdEntry::pack_local_offset(offset);
 *   if (IdEntry::is_local_offset(entry)) { uint64_t off = IdEntry::unpack_local_offset(entry); ... }
 *
 * Clean, documented, minimal.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace pomai::ai::soa
{

    // IdEntry: utilities to encode/decode uint64_t entries in the ids block.
    struct IdEntry
    {
        // Tag bit positions:
        // top two MSBs reserved for type tag
        //  - 0b00 = LOCAL_OFFSET
        //  - 0b01 = REMOTE_ID
        //  - 0b10 = LABEL_ID
        //  - 0b11 = RESERVED
        static constexpr uint64_t TAG_SHIFT = 62;
        static constexpr uint64_t TAG_MASK = (uint64_t)0x3ULL << TAG_SHIFT;

        static constexpr uint64_t TAG_LOCAL = 0x0ULL << TAG_SHIFT;  // 00
        static constexpr uint64_t TAG_REMOTE = 0x1ULL << TAG_SHIFT; // 01
        static constexpr uint64_t TAG_LABEL = 0x2ULL << TAG_SHIFT;  // 10

        // Mask for the 62-bit payload (lower bits).
        static constexpr uint64_t PAYLOAD_MASK = ((uint64_t)1ULL << TAG_SHIFT) - 1ULL;

        // Sentinel for "unset/empty" entry (all zeros).
        static constexpr uint64_t EMPTY = 0ULL;

        // Pack helpers (caller must ensure value fits in 62 bits)
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

        // Predicate helpers
        static inline bool is_empty(uint64_t v) noexcept { return v == EMPTY; }
        static inline bool is_local_offset(uint64_t v) noexcept { return (v & TAG_MASK) == TAG_LOCAL && v != EMPTY; }
        static inline bool is_remote_id(uint64_t v) noexcept { return (v & TAG_MASK) == TAG_REMOTE; }
        static inline bool is_label(uint64_t v) noexcept { return (v & TAG_MASK) == TAG_LABEL; }

        // Unpack helpers (no validation performed; caller can call is_* first)
        static inline uint64_t unpack_local_offset(uint64_t v) noexcept { return v & PAYLOAD_MASK; }
        static inline uint64_t unpack_remote_id(uint64_t v) noexcept { return v & PAYLOAD_MASK; }
        static inline uint64_t unpack_label(uint64_t v) noexcept { return v & PAYLOAD_MASK; }
    };

    // Ensure platform has 8-byte uint64_t (sensible check)
    static_assert(sizeof(uint64_t) == 8, "Platform requires 8-byte uint64_t");

} // namespace pomai::ai::soa