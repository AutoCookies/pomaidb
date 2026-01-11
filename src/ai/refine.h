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
 *      * LABEL entries        => may be resolved via optional callback supplied by caller
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
 *   // supply optional fetcher to resolve LABEL id_entries from index memory:
 *   std::function<bool(uint64_t id_entry, std::vector<float>& out)> fetcher = ...
 *   auto top = refine::refine_topk_l2(query, dim, candidate_ids, ids_block_ptr, arena, K, fetcher);
 *
 * Returns:
 *   Vector of pairs (id, score) sorted ascending for L2 (best/smallest first),
 *   and descending for inner-product (best/largest first).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <functional>

#include "src/ai/candidate_collector.h"
#include "src/ai/ids_block.h"
#include "src/memory/arena.h"

namespace pomai::ai::refine
{

    // Compute top-K exact L2 distances for the given candidates.
    //
    // An optional fetcher callback may be supplied to resolve LABEL id_entries:
    //   std::function<bool(uint64_t id_entry, std::vector<float>& out_buf)>
    // The callback should return true and fill out_buf (dim floats) on success.
    std::vector<std::pair<size_t, float>> refine_topk_l2(const float *query, size_t dim,
                                                         const std::vector<size_t> &candidate_ids,
                                                         const uint64_t *ids_block,
                                                         pomai::memory::PomaiArena *arena,
                                                         size_t K,
                                                         std::function<bool(uint64_t, std::vector<float>&)> label_fetcher = nullptr);

    // Compute top-K exact inner-product scores for the given candidates.
    // Optional label_fetcher accessor as above.
    std::vector<std::pair<size_t, float>> refine_topk_ip(const float *query, size_t dim,
                                                         const std::vector<size_t> &candidate_ids,
                                                         const uint64_t *ids_block,
                                                         pomai::memory::PomaiArena *arena,
                                                         size_t K,
                                                         std::function<bool(uint64_t, std::vector<float>&)> label_fetcher = nullptr);

} // namespace pomai::ai::refine