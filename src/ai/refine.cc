/*
 * src/ai/refine.cc
 *
 * Implementation of final refine helpers (exact L2 and inner-product).
 *
 * This file now consistently uses atomic_utils for reading ids_block entries
 * (std::atomic_ref when available; volatile fallback otherwise) to avoid torn
 * reads while background demote/promote updates occur.
 *
 * The code also documents that readers should treat missing/EMPTY entries
 * appropriately (skip).
 */

#include "src/ai/refine.h"
#include "src/ai/atomic_utils.h"

#include <cstring> // memcpy
#include <cmath>
#include <limits>
#include <iostream>

namespace pomai::ai::refine
{

    // Helper: fetch vector into out_buf (size dim).
    // Returns true on success, false on failure.
    // This helper now accepts an optional label_fetcher lambda to resolve LABEL entries.
    static bool fetch_vector_by_identry(uint64_t id_entry,
                                        pomai::memory::PomaiArena *arena,
                                        size_t dim,
                                        std::vector<float> &out_buf,
                                        std::function<bool(uint64_t, std::vector<float>&)> label_fetcher)
    {
        using namespace pomai::ai::soa;

        if (IdEntry::is_local_offset(id_entry))
        {
            if (!arena)
                return false;
            uint64_t local_off = IdEntry::unpack_local_offset(id_entry);
            const char *p = arena->blob_ptr_from_offset_for_map(local_off);
            if (!p)
                return false;
            uint32_t blen = *reinterpret_cast<const uint32_t *>(p);
            const char *payload = p + sizeof(uint32_t);
            size_t expect_bytes = dim * sizeof(float);
            if (static_cast<uint64_t>(blen) < expect_bytes)
                return false;
            out_buf.resize(dim);
            std::memcpy(out_buf.data(), payload, expect_bytes);
            return true;
        }
        else if (IdEntry::is_remote_id(id_entry))
        {
            if (!arena)
                return false;
            uint64_t remote_id = IdEntry::unpack_remote_id(id_entry);
            const char *p = arena->blob_ptr_from_offset_for_map(remote_id);
            if (!p)
            {
                uint64_t new_local = arena->promote_remote(remote_id);
                if (new_local == UINT64_MAX)
                    return false;
                p = arena->blob_ptr_from_offset_for_map(new_local);
                if (!p)
                    return false;
            }
            uint32_t blen = *reinterpret_cast<const uint32_t *>(p);
            const char *payload = p + sizeof(uint32_t);
            size_t expect_bytes = dim * sizeof(float);
            if (static_cast<uint64_t>(blen) < expect_bytes)
                return false;
            out_buf.resize(dim);
            std::memcpy(out_buf.data(), payload, expect_bytes);
            return true;
        }
        else if (IdEntry::is_label(id_entry))
        {
            // Attempt to resolve via supplied label_fetcher callback.
            if (label_fetcher)
            {
                return label_fetcher(id_entry, out_buf);
            }
            return false;
        }
        else
        {
            return false;
        }
    }

    static inline float compute_l2_squared(const float *query, const float *vec, size_t dim)
    {
        double acc = 0.0;
        for (size_t i = 0; i < dim; ++i)
        {
            double d = static_cast<double>(query[i]) - static_cast<double>(vec[i]);
            acc += d * d;
        }
        if (acc > static_cast<double>(std::numeric_limits<float>::max()))
            return std::numeric_limits<float>::infinity();
        return static_cast<float>(acc);
    }

    static inline float compute_dot(const float *query, const float *vec, size_t dim)
    {
        double acc = 0.0;
        for (size_t i = 0; i < dim; ++i)
            acc += static_cast<double>(query[i]) * static_cast<double>(vec[i]);
        if (acc > static_cast<double>(std::numeric_limits<float>::max()))
            return std::numeric_limits<float>::max();
        if (acc < static_cast<double>(-std::numeric_limits<float>::max()))
            return -std::numeric_limits<float>::max();
        return static_cast<float>(acc);
    }

    // Common driver used by both L2 and IP variants.
    static std::vector<std::pair<size_t, float>> refine_topk_common(const float *query, size_t dim,
                                                                    const std::vector<size_t> &candidate_ids,
                                                                    const uint64_t *ids_block,
                                                                    pomai::memory::PomaiArena *arena,
                                                                    size_t K,
                                                                    bool is_ip,
                                                                    std::function<bool(uint64_t, std::vector<float>&)> label_fetcher)
    {
        CandidateCollector coll(K);
        std::vector<float> buf;
        buf.reserve(dim);

        for (size_t idx : candidate_ids)
        {
            // Atomically read ids_block[idx] using atomic_utils
            uint64_t raw = pomai::ai::atomic_utils::atomic_load_u64(ids_block + idx);

            if (raw == pomai::ai::soa::IdEntry::EMPTY)
                continue;

            bool ok = fetch_vector_by_identry(raw, arena, dim, buf, label_fetcher);
            if (!ok)
                continue;

            float score = 0.0f;
            if (is_ip)
            {
                float dot = compute_dot(query, buf.data(), dim);
                score = -dot;
            }
            else
            {
                score = compute_l2_squared(query, buf.data(), dim);
            }

            coll.add(idx, score);
        }

        auto top = coll.topk();
        std::vector<std::pair<size_t, float>> out;
        out.reserve(top.size());
        for (const auto &p : top)
        {
            if (is_ip)
                out.emplace_back(p.first, -p.second);
            else
                out.emplace_back(p.first, p.second);
        }
        return out;
    }

    std::vector<std::pair<size_t, float>> refine_topk_l2(const float *query, size_t dim,
                                                         const std::vector<size_t> &candidate_ids,
                                                         const uint64_t *ids_block,
                                                         pomai::memory::PomaiArena *arena,
                                                         size_t K,
                                                         std::function<bool(uint64_t, std::vector<float>&)> label_fetcher)
    {
        return refine_topk_common(query, dim, candidate_ids, ids_block, arena, K, false, label_fetcher);
    }

    std::vector<std::pair<size_t, float>> refine_topk_ip(const float *query, size_t dim,
                                                         const std::vector<size_t> &candidate_ids,
                                                         const uint64_t *ids_block,
                                                         pomai::memory::PomaiArena *arena,
                                                         size_t K,
                                                         std::function<bool(uint64_t, std::vector<float>&)> label_fetcher)
    {
        return refine_topk_common(query, dim, candidate_ids, ids_block, arena, K, true, label_fetcher);
    }

} // namespace pomai::ai::refine