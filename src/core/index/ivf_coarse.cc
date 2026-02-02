#include "core/index/ivf_coarse.h"

#include <algorithm>
#include <limits>

#include "core/distance.h"

namespace pomai::index
{

    IvfCoarse::IvfCoarse(std::uint32_t dim, Options opt)
        : dim_(dim), opt_(opt)
    {
        if (opt_.nlist == 0)
            opt_.nlist = 1;
        if (opt_.nprobe == 0)
            opt_.nprobe = 1;
        if (opt_.nprobe > opt_.nlist)
            opt_.nprobe = opt_.nlist;

        centroids_.assign(static_cast<std::size_t>(opt_.nlist) * dim_, 0.0f);
        counts_.assign(opt_.nlist, 0);
        lists_.resize(opt_.nlist);

        // keep map stable-ish
        id2list_.reserve(1u << 20);
    }

    std::uint32_t IvfCoarse::AssignCentroid(std::span<const float> vec) const
    {
        // Choose centroid by maximum dot product (consistent with scoring).
        float best = -std::numeric_limits<float>::infinity();
        std::uint32_t best_id = 0;

        for (std::uint32_t c = 0; c < opt_.nlist; ++c)
        {
            const float *p = &centroids_[static_cast<std::size_t>(c) * dim_];
            float s = pomai::core::Dot(vec, std::span<const float>(p, dim_));
            if (s > best)
            {
                best = s;
                best_id = c;
            }
        }
        return best_id;
    }

    void IvfCoarse::SeedOrUpdateCentroid(std::uint32_t cid, std::span<const float> vec)
    {
        float *dst = &centroids_[static_cast<std::size_t>(cid) * dim_];

        // Seeding phase: copy vector into centroid.
        if (counts_[cid] == 0)
        {
            for (std::uint32_t i = 0; i < dim_; ++i)
                dst[i] = vec[i];
            counts_[cid] = 1;
            return;
        }

        // Online EMA update.
        const float a = opt_.ema;
        for (std::uint32_t i = 0; i < dim_; ++i)
            dst[i] = (1.0f - a) * dst[i] + a * vec[i];
        counts_[cid] += 1;
    }

    pomai::Status IvfCoarse::Put(pomai::VectorId id, std::span<const float> vec)
    {
        if (static_cast<std::uint32_t>(vec.size()) != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        // If already existed, remove from old list first.
        auto it = id2list_.find(id);
        if (it != id2list_.end())
        {
            const std::uint32_t old = it->second;
            auto &lst = lists_[old];
            // erase by swap-remove (order doesn't matter)
            auto pos = std::find(lst.begin(), lst.end(), id);
            if (pos != lst.end())
            {
                *pos = lst.back();
                lst.pop_back();
            }
            // keep live_count unchanged (upsert)
        }
        else
        {
            live_count_ += 1;
        }

        // Seeding: first nlist unique puts initialize centroids.
        std::uint32_t cid = 0;
        if (seeded_ < opt_.nlist)
        {
            cid = seeded_;
            seeded_ += 1;
            SeedOrUpdateCentroid(cid, vec);
        }
        else
        {
            cid = AssignCentroid(vec);
            SeedOrUpdateCentroid(cid, vec);
        }

        lists_[cid].push_back(id);
        id2list_[id] = cid;
        return pomai::Status::Ok();
    }

    pomai::Status IvfCoarse::Delete(pomai::VectorId id)
    {
        auto it = id2list_.find(id);
        if (it == id2list_.end())
            return pomai::Status::Ok();

        const std::uint32_t cid = it->second;
        auto &lst = lists_[cid];
        auto pos = std::find(lst.begin(), lst.end(), id);
        if (pos != lst.end())
        {
            *pos = lst.back();
            lst.pop_back();
        }
        id2list_.erase(it);

        if (live_count_ > 0)
            live_count_ -= 1;

        return pomai::Status::Ok();
    }

    pomai::Status IvfCoarse::SelectCandidates(std::span<const float> query,
                                              std::vector<pomai::VectorId> *candidates) const
    {
        if (!candidates)
            return pomai::Status::InvalidArgument("candidates null");
        candidates->clear();

        if (static_cast<std::uint32_t>(query.size()) != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        // If not ready or too small => let caller brute-force.
        if (!ready() || live_count_ < opt_.warmup)
            return pomai::Status::Ok();

        // Score centroids by dot(query, centroid), pick top nprobe.
        auto &scored = scratch_scored_;
        scored.clear();
        scored.reserve(opt_.nlist);

        for (std::uint32_t c = 0; c < opt_.nlist; ++c)
        {
            const float *p = &centroids_[static_cast<std::size_t>(c) * dim_];
            float s = pomai::core::Dot(query, std::span<const float>(p, dim_));
            scored.push_back({c, s});
        }

        const std::uint32_t p = opt_.nprobe;
        if (scored.size() > p)
        {
             std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(p), scored.end(),
                              [](const ScoredCentroid &a, const ScoredCentroid &b)
                              { return a.score > b.score; });
             scored.resize(p);
        }

        std::sort(scored.begin(), scored.end(),
                  [](const ScoredCentroid &a, const ScoredCentroid &b)
                  { return a.score > b.score; });

        // Gather candidates from selected lists.
        // NOTE: duplicates shouldn't happen (each id maps to one list), but keep it clean.
        candidates->reserve(static_cast<std::size_t>(p) * 1024);

        for (const auto &x : scored)
        {
            const auto &lst = lists_[x.id];
            candidates->insert(candidates->end(), lst.begin(), lst.end());
        }

        // Optional: if candidates huge, keep it as is; shard will heap-topk anyway.
        // If you want hard cap: you can add a cap later (but keep semantics now).
        return pomai::Status::Ok();
    }

} // namespace pomai::index
