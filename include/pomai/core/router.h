#pragma once
#include <future>
#include <memory>
#include <vector>

#include "pomai/core/shard_runtime.h"
#include "pomai/core/topk.h"
#include "pomai/core/stats.h"

namespace pomai::core
{

    struct RouterOptions
    {
        bool block_on_full_queue{false}; // if false: return BUSY
    };

    struct UpsertBatchResult
    {
        pomai::Status status;
        std::size_t ok_count{0};
        std::size_t fail_count{0};
    };

    class Router final
    {
    public:
        Router(std::vector<std::unique_ptr<ShardRuntime>> shards, RouterOptions opt);

        pomai::Status Start();
        void Stop();

        UpsertBatchResult UpsertBatch(std::vector<UpsertItem> items);
        SearchReply Search(const SearchRequest &req);
        std::vector<ShardStatsSnapshot> GetStats() const;
        pomai::Status FlushAll();

    private:
        std::uint32_t PickShard(pomai::VectorId id) const;

        std::vector<std::unique_ptr<ShardRuntime>> shards_;
        RouterOptions opt_;
    };

} // namespace pomai::core
