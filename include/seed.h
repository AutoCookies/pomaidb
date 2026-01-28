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
            std::size_t accounted_bytes{0};
        };

        using Snapshot = std::shared_ptr<const Store>;

        explicit Seed(std::size_t dim);
        Seed(const Seed &other);
        Seed &operator=(const Seed &other);
        Seed(Seed &&other) noexcept;
        Seed &operator=(Seed &&other) noexcept;
        ~Seed();

        // Apply a batch of upserts (owner thread only).
        void ApplyUpserts(const std::vector<UpsertRequest> &batch);

        // Create an immutable snapshot for concurrent readers.
        Snapshot MakeSnapshot() const;
        static bool TryDetachSnapshot(Snapshot &snap, std::vector<float> &data, std::vector<Id> &ids);

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
        void UpdateMemtableAccounting();
        void ReleaseMemtableAccounting();

        std::size_t accounted_bytes_{0};
    };

} // namespace pomai
