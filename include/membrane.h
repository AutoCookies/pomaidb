#pragma once
#include <vector>
#include <memory>
#include <future>
#include "shard.h"
#include "server/config.h"
#include "whispergrain.h"

namespace pomai
{

    class MembraneRouter
    {
    public:
        explicit MembraneRouter(std::vector<std::unique_ptr<Shard>> shards, pomai::server::WhisperConfig w_cfg);

        void Start();
        void Stop();

        std::future<Lsn> Upsert(Id id, Vector vec, bool wait_durable = true);
        std::future<Lsn> UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable = true);
        SearchResponse Search(const SearchRequest &req) const;

        std::size_t ShardCount() const { return shards_.size(); }
        std::size_t TotalApproxCountUnsafe() const;

    private:
        std::size_t PickShard(Id id) const;

        std::vector<std::unique_ptr<Shard>> shards_;
        mutable pomai::ai::WhisperGrain brain_;
    };

}