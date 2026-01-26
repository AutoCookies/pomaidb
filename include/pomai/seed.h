#pragma once
#include <memory>
#include <unordered_map>
#include "types.h"

namespace pomai
{

    class Seed
    {
    public:
        using Store = std::unordered_map<Id, std::vector<float>>;
        using Snapshot = std::shared_ptr<const Store>;

        explicit Seed(std::size_t dim);

        void ApplyUpserts(const std::vector<UpsertRequest> &batch);

        // Create an immutable snapshot handle (cheap enough for step 2).
        Snapshot MakeSnapshot() const;

        // Step 2: simple exact scan using the snapshot (still basic).
        static SearchResponse SearchSnapshot(const Snapshot &snap, const SearchRequest &req);

        std::size_t Count() const { return store_.size(); }
        std::size_t Dim() const { return dim_; }

    private:
        std::size_t dim_{0};
        Store store_;
    };

} // namespace pomai
