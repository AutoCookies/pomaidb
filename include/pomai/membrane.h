#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include "shard.h"
#include <future> 

namespace pomai
{

    // Membrane routes requests to shards. In Step 1: hash(id) % S.
    class MembraneRouter
    {
    public:
        explicit MembraneRouter(std::vector<std::unique_ptr<Shard>> shards);

        void Start();
        void Stop();

        std::future<Lsn> Upsert(Id id, Vector vec, bool wait_durable = true);
        std::future<Lsn> UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable = true);
        SearchResponse Search(const SearchRequest& req) const;

        std::size_t ShardCount() const { return shards_.size(); }
        std::size_t TotalApproxCountUnsafe() const;

    private:
        std::size_t PickShard(Id id) const;

        std::vector<std::unique_ptr<Shard>> shards_;
    };

} // namespace pomai
