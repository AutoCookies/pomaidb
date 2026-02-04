#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <memory> 

#include "pomai/search.h"
#include "pomai/types.h"

namespace pomai::table {
    class MemTable;
    class SegmentReader;
}

namespace pomai::test {

// BruteForceSearch scans ALL vectors in the provided MemTable and Segments.
// It computes exact dot product distances and returns the exact top-k.
//
// Tie-breaking:
// 1. Higher score wins.
// 2. If scores equal, LOWER VectorId wins (deterministic).
//
// Thread-safety: Not thread-safe. Caller must ensure MemTable/Segments are stable.
std::vector<pomai::SearchHit> BruteForceSearch(
    std::span<const float> query,
    std::uint32_t topk,
    const pomai::table::MemTable* mem,
    const std::vector<std::shared_ptr<pomai::table::SegmentReader>>& segments
);

} // namespace pomai::test
