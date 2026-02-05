#include "tests/common/bruteforce_oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/distance.h"
#include "table/memtable.h"
#include "table/segment.h"

namespace pomai::test {

std::vector<pomai::SearchHit> BruteForceSearch(
    std::span<const float> query,
    std::uint32_t topk,
    const pomai::table::MemTable* mem,
    const std::vector<std::shared_ptr<pomai::table::SegmentReader>>& segments
) {
    if (topk == 0) return {};

    std::vector<pomai::SearchHit> all_hits;
    // Heuristic reserve: assume some reasonable density, but vectors can resize.
    // We don't know total count easily without iterating, but that's fine.
    all_hits.reserve(4096); 

    auto add_hit = [&](pomai::VectorId id, std::span<const float> vec) {
        // Assume dim matches. In oracle we can assert or check, but query size is passed.
        // We'll trust caller or add check if needed.
        if (query.size() != vec.size()) return; // Should not happen if DB consistent
        
        float score = pomai::core::Dot(query, vec);
        all_hits.push_back({id, score});
    };

    // 1. Scan MemTable
    if (mem) {
        mem->ForEach(add_hit);
    }

    // 2. Scan Segments
    for (const auto& seg : segments) {
        if (seg) {
            seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata*) {
                if (is_deleted) return;
                add_hit(id, vec);
            });
        }
    }

    // 3. Sort for exact top-k with deterministic tie-break
    // Primary: Score descending
    // Secondary: ID ascending
    std::sort(all_hits.begin(), all_hits.end(), [](const pomai::SearchHit& a, const pomai::SearchHit& b) {
        if (std::abs(a.score - b.score) > 1e-6f) { // Float tolerance? Distance is dot product.
            // Strict float comparison is usually fine for "exact" brute force logic unless optimization reorders ops.
            // But let's use exact comparison for strict determinism on same machine.
             return a.score > b.score;
        }
        // If scores exactly equal (or very close if we used tolerance, but here we strictly follow rule 1)
        // Actually, for "Exact", we should just use >.
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.id < b.id;
    });

    // 4. Truncate
    if (all_hits.size() > topk) {
        all_hits.resize(topk);
    }

    return all_hits;
}

} // namespace pomai::test
