/*
 * src/ai/refine.h
 *
 * Final refine stage helpers: exact L2 and inner-product computation on top-N
 * candidates. Fetches full-precision vectors from PomaiArena when needed and
 * computes exact distances (or scores) used to produce the final reranked list.
 *
 * Key functions:
 *   - refine_topk_l2(...)   : compute exact L2 distances for candidates and return top-K (smallest distances)
 *   - refine_topk_ip(...)   : compute exact inner-product scores for candidates and return top-K (largest dot)
 *
 * Behavior & robustness:
 *  - Candidate ids are indices into the ids/offsets block (uint64_t per-vector).
 *  - The ids block entries are encoded using IdEntry (src/ai/ids_block.h). We handle:
 *      * LOCAL_OFFSET entries => vector stored inside arena blob region
 *      * REMOTE_ID entries    => vector stored as demoted file; blob_ptr_from_offset_for_map will mmap lazily
 *      * LABEL entries        => no vector payload available -> treated as missing (skipped)
 *  - Blob layout expected: [uint32_t length_bytes][payload bytes...][\0]
 *    For full float vectors the payload should contain length_bytes == dim * sizeof(float).
 *
 * Threading:
 *  - Functions are reentrant. They do not mutate arena state except by calling
 *    PomaiArena::promote_remote when necessary (potentially blocking).
 *
 * Error handling:
 *  - If a candidate's vector cannot be fetched (missing, corrupt length), the candidate is skipped.
 *  - Caller receives only successfully refined candidates in the result vector.
 *
 * Usage example:
 *   auto top = refine::refine_topk_l2(query, dim, candidate_ids, ids_block_ptr, arena, K);
 *
 * Returns:
 *   Vector of pairs (id, score) sorted ascending for L2 (best/smallest first),
 *   and descending for inner-product (best/largest first).
 *
 * Clean, commented, production-quality code.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

#include "src/ai/candidate_collector.h"
#include "src/ai/ids_block.h"
#include "src/memory/arena.h"

namespace pomai::ai::refine
{

    // Compute top-K exact L2 distances for the given candidates.
    //
    // Parameters:
    //  - query: pointer to query float vector (length dim)
    //  - dim: dimensionality
    //  - candidate_ids: list of candidate element indices (each index used to lookup ids_block[idx])
    //  - ids_block: pointer to uint64_t ids/offsets block (length >= max(candidate_ids)+1). Each entry encoded via IdEntry.
    //  - arena: pointer to PomaiArena used to resolve local offsets / remote ids (must be non-null if any LOCAL/REMOTE entries present)
    //  - K: return up to K best candidates (smallest L2 distances)
    //
    // Returns:
    //  - vector of pairs (id, distance) sorted ascending by distance (best first).
    std::vector<std::pair<size_t, float>> refine_topk_l2(const float *query, size_t dim,
                                                         const std::vector<size_t> &candidate_ids,
                                                         const uint64_t *ids_block,
                                                         pomai::memory::PomaiArena *arena,
                                                         size_t K);

    // Compute top-K exact inner-product scores for the given candidates.
    //
    // Semantics similar to refine_topk_l2, except we compute dot(query, vec).
    // Result is sorted descending by score (best/largest first).
    std::vector<std::pair<size_t, float>> refine_topk_ip(const float *query, size_t dim,
                                                         const std::vector<size_t> &candidate_ids,
                                                         const uint64_t *ids_block,
                                                         pomai::memory::PomaiArena *arena,
                                                         size_t K);

} // namespace pomai::ai::refine