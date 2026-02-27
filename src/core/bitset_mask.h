// bitset_mask.h — Pre-computed per-segment bitset for fast filtered search.
//
// Phase 3 (Milvus-Lite inspired): Instead of calling FilterEvaluator::Matches()
// per candidate inside the hot search loop — which reads metadata from mmap,
// parses strings, and stresses the branch predictor — we pre-scan the segment
// once into a uint64_t bitset. The hot loop then does a single bit test per
// candidate (1 CPU cycle), skipping string parsing entirely.
//
// Design: Milvus Lite query_task.cpp uses the same BitsetView pattern from
// knowhere for filtered HNSW/IVF. We implement a self-contained version with
// no external dependencies.
//
// Usage:
//   BitsetMask mask(seg->Count());
//   mask.BuildFromSegment(*seg, opts);          // one pass, O(N)
//   for (auto idx : candidates) {
//       if (!mask.Test(idx)) continue;           // O(1), branch-predictor-friendly
//       ...score(idx)...
//   }

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "pomai/metadata.h"
#include "table/segment.h"

namespace pomai::core {

/// Compact bitset for segment entry indices.
/// Bit i = 1  →  entry i PASSES the filter (include in search).
/// Bit i = 0  →  entry i is excluded (filtered out or deleted).
class BitsetMask {
public:
    // ── Construction ─────────────────────────────────────────────────────────
    explicit BitsetMask(uint32_t n)
        : n_(n), words_((n + 63u) / 64u, 0u) {}

    // Return a fully-set mask (all entries pass) for the no-filter fast path.
    static BitsetMask All(uint32_t n) {
        BitsetMask m(n);
        std::fill(m.words_.begin(), m.words_.end(), ~uint64_t{0});
        return m;
    }

    // ── Build from segment + SearchOptions ────────────────────────────────────
    // Single pass over the segment's metadata array. Marks each entry either
    // pass (1) or skip (0) based on `opts` filter + tombstone check.
    // Complexity: O(N) metadata reads before the search loop.
    void BuildFromSegment(const table::SegmentReader& seg,
                          const SearchOptions& opts);

    // ── Query ─────────────────────────────────────────────────────────────────
    /// True if entry at `idx` passes all filters.
    inline bool Test(uint32_t idx) const {
        return (words_[idx >> 6u] >> (idx & 63u)) & 1u;
    }

    /// How many bits are set (useful for recall debugging).
    uint32_t PopCount() const;

    uint32_t size() const { return n_; }
    bool empty() const { return n_ == 0; }

    // ── 64-vector block skip (SIMD-friendly outer loop) ───────────────────────
    // Returns true if the 64-bit word containing `idx` is all-zero.
    // Callers can use this to skip entire 64-vector blocks.
    inline bool WordEmpty(uint32_t word_idx) const {
        return words_[word_idx] == 0u;
    }
    uint32_t NumWords() const { return static_cast<uint32_t>(words_.size()); }

private:
    uint32_t n_;
    std::vector<uint64_t> words_;

    inline void Set(uint32_t idx) {
        words_[idx >> 6u] |= (uint64_t{1} << (idx & 63u));
    }
};

} // namespace pomai::core
