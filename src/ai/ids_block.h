#pragma once
/*
 * src/ai/ids_block.h
 *
 * Compact, versioned helpers for the per-vector IDs/offsets block used by the
 * SoA mmap layout.
 *
 * Improvements:
 *  - Added safe "try_pack" helpers to avoid silent truncation when payload
 *    does not fit into 62 bits.
 *  - Added small atomic helpers (declarations) implemented in ids_block.cc
 *    to make it convenient and safe to update entries stored in mmap'd memory.
 *  - Extra utility functions: fits_payload, tag_of.
 *
 * Note: This header stays lightweight and suitable for inlining hot path bit ops.
 */

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

        // Safe try-pack helpers — return false when value doesn't fit (no silent truncation)
        // Implemented in ids_block.cc (non-inline) to keep header small.
        static bool try_pack_local_offset(uint64_t offset, uint64_t &out) noexcept;
        static bool try_pack_remote_id(uint64_t remote_id, uint64_t &out) noexcept;
        static bool try_pack_label(uint64_t label, uint64_t &out) noexcept;

        // Predicate helpers
        static inline bool is_empty(uint64_t v) noexcept { return v == EMPTY; }
        static inline bool is_local_offset(uint64_t v) noexcept { return ((v & TAG_MASK) == TAG_LOCAL) && (v != EMPTY); }
        static inline bool is_remote_id(uint64_t v) noexcept { return (v & TAG_MASK) == TAG_REMOTE; }
        static inline bool is_label(uint64_t v) noexcept { return (v & TAG_MASK) == TAG_LABEL; }

        // Unpack helpers (no validation performed; caller can call is_* first)
        static inline uint64_t unpack_local_offset(uint64_t v) noexcept { return v & PAYLOAD_MASK; }
        static inline uint64_t unpack_remote_id(uint64_t v) noexcept { return v & PAYLOAD_MASK; }
        static inline uint64_t unpack_label(uint64_t v) noexcept { return v & PAYLOAD_MASK; }

        // Utility helpers
        static inline bool fits_payload(uint64_t v) noexcept { return (v & ~PAYLOAD_MASK) == 0; }
        static inline uint64_t tag_of(uint64_t v) noexcept { return v & TAG_MASK; }
    };

    // Ensure platform has 8-byte uint64_t (sensible check)
    static_assert(sizeof(uint64_t) == 8, "Platform requires 8-byte uint64_t");
    static_assert(IdEntry::PAYLOAD_MASK == ((1ULL << 62) - 1ULL), "PAYLOAD_MASK must be 62 bits");

    // -------------------------
    // Atomic helpers (declarations)
    // -------------------------
    //
    // Convenience wrappers to perform atomic load/store/CAS into memory that may be mmap'd.
    // These functions use the platform/compiler atomic builtins under the hood (see ids_block.cc)
    // and are safe to call on addresses that may live in shared/mmap'd memory.
    //
    // Note: call sites that already manage atomicity can avoid these helpers and operate on
    // std::atomic<uint64_t> or use pomai::ai::atomic_utils directly.
    //
    // - atomic_store_entry(ptr, v): store v with release semantics
    // - atomic_load_entry(ptr): load value with acquire semantics
    // - atomic_compare_exchange_entry(ptr, expected, desired): CAS (seq_cst), returns true on success,
    //     on failure 'expected' is updated with current value.
    //

    void atomic_store_entry(uint64_t *ptr, uint64_t v) noexcept;
    uint64_t atomic_load_entry(const uint64_t *ptr) noexcept;
    bool atomic_compare_exchange_entry(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept;

} // namespace pomai::ai::soa