#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace pomai
{

    class Seed
    {
    public:
        // Immutable snapshot payload for readers.
        struct Store
        {
            std::size_t dim{0};
            std::vector<Id> ids;     // row ids [N]
            std::vector<float> data; // row-major vectors [N * dim]
        };

        using Snapshot = std::shared_ptr<const Store>;

        explicit Seed(std::size_t dim);

        // Apply a batch of upserts (owner thread only).
        void ApplyUpserts(const std::vector<UpsertRequest> &batch);

        // Create an immutable snapshot for concurrent readers.
        Snapshot MakeSnapshot() const;

        // Exact scan using snapshot (L2 squared), returns best K with score = -dist.
        static SearchResponse SearchSnapshot(const Snapshot &snap, const SearchRequest &req);

        std::size_t Count() const { return ids_.size(); }
        std::size_t Dim() const { return dim_; }

    private:
        std::size_t dim_{0};

        // Mutable hot store (owner thread only):
        std::vector<Id> ids_;                       // [N]
        std::vector<float> data_;                   // [N * dim]
        std::unordered_map<Id, std::uint32_t> pos_; // id -> row index

        void ReserveForAppend(std::size_t add_rows);
    };

} // namespace pomai
