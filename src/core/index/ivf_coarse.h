#pragma once
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::table
{
    class MemTable;
} // namespace pomai::table

namespace pomai::index
{

    // A simple IVF-Coarse index:
    // - Centroids learned online (seed first nlist vectors, then EMA updates)
    // - Posting lists store VectorId (stable); rerank uses MemTable pointer lookup.
    // - All methods are NOT thread-safe by themselves; intended to be used inside shard actor thread.
    class IvfCoarse
    {
    public:
        struct Options
        {
            std::uint32_t nlist = 64;   // #centroids
            std::uint32_t nprobe = 4;   // #centroids to probe at query time
            std::uint32_t warmup = 256; // below this #live vectors -> fallback brute force
            float ema = 0.05f;          // centroid update rate after seeded
        };

        IvfCoarse(std::uint32_t dim, Options opt);

        // Update index on upsert.
        pomai::Status Put(pomai::VectorId id, std::span<const float> vec);

        // Update index on delete (best-effort tombstone removal).
        pomai::Status Delete(pomai::VectorId id);

        // Query: choose top-nprobe centroid(s), gather candidates, rerank with exact dot.
        // If index not ready or too small -> fallback brute force scan in caller.
        //
        // Returns:
        //  - ok + candidates filled if IVF path used
        //  - ok + candidates empty if caller should brute-force
        pomai::Status SelectCandidates(std::span<const float> query, std::vector<pomai::VectorId> *candidates) const;

        std::uint32_t dim() const { return dim_; }
        const Options &options() const { return opt_; }

        // Stats
        std::uint64_t live_count() const { return live_count_; }
        bool ready() const { return seeded_ == opt_.nlist && opt_.nlist > 0; }

    private:
        std::uint32_t AssignCentroid(std::span<const float> vec) const;
        void SeedOrUpdateCentroid(std::uint32_t cid, std::span<const float> vec);

        std::uint32_t dim_;
        Options opt_;

        // centroid vectors: size = nlist * dim
        std::vector<float> centroids_;
        std::vector<std::uint32_t> counts_;

        // posting lists: vector of ids
        std::vector<std::vector<pomai::VectorId>> lists_;

        // id -> centroid assignment (for delete/move)
        std::unordered_map<pomai::VectorId, std::uint32_t> id2list_;

        std::uint32_t seeded_ = 0;
        std::uint64_t live_count_ = 0;

        struct ScoredCentroid
        {
            std::uint32_t id;
            float score;
        };
        mutable std::vector<ScoredCentroid> scratch_scored_;
    };

} // namespace pomai::index
