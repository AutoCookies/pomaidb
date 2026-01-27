#pragma once
#include <vector>
#include <shared_mutex>
#include <atomic>
#include <memory>
#include <cstring>
#include <unordered_map>
#include "types.h"

namespace pomai
{

    constexpr std::size_t kVectorsPerChunk = 1024;

    struct SeedSnapshot;

    class Seed
    {
    public:
        explicit Seed(std::size_t dim);

        Seed(const Seed &) = delete;
        Seed &operator=(const Seed &) = delete;

        void ApplyUpserts(const std::vector<UpsertRequest> &batch);
        std::shared_ptr<const SeedSnapshot> MakeSnapshot() const;

        static SearchResponse SearchSnapshot(const std::shared_ptr<const SeedSnapshot> &snap, const SearchRequest &req);

        std::size_t Count() const;
        std::size_t Dim() const { return dim_; }

        // API QUAN TRỌNG CHO INDEX BUILDER (KMEANS)
        std::vector<float> GetFlatData() const;
        std::vector<Id> GetFlatIds() const;

    private:
        std::size_t dim_;
        mutable std::shared_mutex mu_;

        std::vector<Id> ids_;
        std::unordered_map<Id, std::pair<std::uint32_t, std::uint32_t>> id_to_loc_;

        using Chunk = std::vector<float>;
        std::vector<std::shared_ptr<Chunk>> chunks_;
    };

    struct SeedSnapshot
    {
        std::vector<Id> ids;
        std::vector<std::shared_ptr<std::vector<float>>> chunks;
        std::size_t dim;
    };

}