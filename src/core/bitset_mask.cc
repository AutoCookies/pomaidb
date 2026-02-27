// bitset_mask.cc — BitsetMask::BuildFromSegment implementation.
//
// Phase 3: Pre-scan a frozen segment to produce a dense bitset of which entries
// pass the user's filter. This trades one O(N) forward pass (sequential mmap
// reads — cache-friendly) for O(1) per-candidate bit tests in the hot search
// loop, eliminating per-candidate string parsing and branch mispredictions.

#include "core/bitset_mask.h"

#include <cstring>

namespace pomai::core {

// ── BuildFromSegment ────────────────────────────────────────────────────────
// Strategy: walk the segment metadata block directly (same layout as ForEach)
//   Metadata layout inside SegmentReader (see segment.h ForEach):
//     base_addr + metadata_offset → offsets array: (count+1) × uint64_t
//     blob follows immediately after the offsets array.
//   Entry i has tenant string = blob[offsets[i] .. offsets[i+1])
//
// We replicate the metadata read from ForEach() here to avoid the
// vector → float decoding cost (we only need the metadata, not the vector).

void BitsetMask::BuildFromSegment(const table::SegmentReader& seg,
                                  const SearchOptions& opts)
{
    const uint32_t n = seg.Count();
    if (n == 0) return;

    // Resize to cover this segment (caller should pass seg.Count() at ctor)
    if (n > n_) {
        n_ = n;
        words_.assign((n + 63u) / 64u, 0u);
    } else {
        // Clear existing bits
        std::fill(words_.begin(), words_.end(), 0u);
    }

    const bool has_filters = !opts.filters.empty();

    // Fast path: no filters — mark every non-tombstone entry as passing.
    // We do this via the segment's raw base address so no function-call
    // overhead per entry.
    if (!has_filters) {
        const uint8_t* p = seg.GetBaseAddr() + seg.GetEntriesStartOffset();
        const std::size_t entry_size = seg.GetEntrySize();
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t flags = *(p + 8);
            if (!(flags & table::kFlagTombstone)) Set(i);
            p += entry_size;
        }
        return;
    }

    // Filtered path: We need to read metadata per entry.
    // Use ReadAt with metadata — sequential index scan is cache-friendly.
    VectorId tmp_id;
    bool tmp_deleted;
    Metadata tmp_meta;

    for (uint32_t i = 0; i < n; ++i) {
        // ReadAt with non-null meta_ptr reads the metadata block.
        // For quantized segments we use ReadAtCodes; for float we use ReadAt.
        // Both overloads accept (index, id*, vec*, deleted*, meta*).
        //
        // We call the float overload — if quantized the vec span will still be
        // valid but we ignore it; the important output is out_meta + deleted.
        std::span<const float> ignored_vec;
        auto st = seg.ReadAt(i, &tmp_id, &ignored_vec, &tmp_deleted, &tmp_meta);
        if (!st.ok() || tmp_deleted) continue;
        if (opts.Matches(tmp_meta)) Set(i);
    }
}

// ── PopCount ─────────────────────────────────────────────────────────────────
uint32_t BitsetMask::PopCount() const
{
    uint32_t c = 0;
    for (uint64_t w : words_) {
        // __builtin_popcountll is available on GCC/Clang; portable fallback below
#if defined(__GNUC__) || defined(__clang__)
        c += static_cast<uint32_t>(__builtin_popcountll(w));
#else
        uint64_t x = w;
        while (x) { c += x & 1u; x >>= 1; }
#endif
    }
    return c;
}

} // namespace pomai::core
