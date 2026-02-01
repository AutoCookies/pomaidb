#include "pomai/core/pomai.h"

#include <filesystem>
#include <vector>

#include "pomai/core/shard.h"
#include "pomai/core/shard_runtime.h"

namespace pomai
{

    PomaiDB::PomaiDB(DbOptions opt) : opt_(std::move(opt)) {}
    PomaiDB::~PomaiDB() { Stop(); }

    Status PomaiDB::Start()
    {
        std::filesystem::create_directories(opt_.dir);

        std::vector<std::unique_ptr<core::ShardRuntime>> shards;
        shards.reserve(opt_.num_shards);

        for (std::uint32_t i = 0; i < opt_.num_shards; ++i)
        {
            core::ShardOptions so;
            so.wal_path = opt_.dir / ("shard_" + std::to_string(i) + ".wal");
            so.fsync_policy = opt_.fsync_policy;
            so.vector_dim = opt_.vector_dim;
            so.checkpoint_interval_ops = opt_.checkpoint_interval;

            auto shard = std::make_unique<core::Shard>(i, so);

            core::ShardRuntimeOptions ro;
            ro.inbox_capacity = opt_.shard_inbox_capacity;

            shards.push_back(std::make_unique<core::ShardRuntime>(i, std::move(shard), ro));
        }

        core::RouterOptions rOpt;
        router_ = std::make_unique<core::Router>(std::move(shards), rOpt);

        return router_->Start();
    }

    void PomaiDB::Stop()
    {
        if (router_)
            router_->Stop();
        router_.reset();
    }

    Status PomaiDB::Upsert(VectorId id, VectorData vec)
    {
        // Default empty payload for simple Upsert
        pomai::UpsertItem it{id, std::move(vec), ""};
        std::vector<pomai::UpsertItem> items;
        items.push_back(std::move(it));
        auto r = UpsertBatch(std::move(items));
        return r.status;
    }

    UpsertBatchResult PomaiDB::UpsertBatch(std::vector<pomai::UpsertItem> items)
    {
        auto r = router_->UpsertBatch(std::move(items));
        UpsertBatchResult out;
        out.status = r.status;
        out.ok_count = r.ok_count;
        out.fail_count = r.fail_count;
        return out;
    }

    Result<std::vector<SearchHit>> PomaiDB::Search(VectorData query, std::uint32_t topk)
    {
        core::SearchRequest req;
        req.query = std::move(query);
        req.topk = topk;
        req.include_payload = false;

        auto rep = router_->Search(req);
        if (!rep.status.ok())
            return Result<std::vector<SearchHit>>::Err(rep.status);
        return Result<std::vector<SearchHit>>::Ok(std::move(rep.hits));
    }

    Result<std::vector<SearchHit>> PomaiDB::SearchWithPayload(VectorData query, std::uint32_t topk)
    {
        core::SearchRequest req;
        req.query = std::move(query);
        req.topk = topk;
        req.include_payload = true;

        auto rep = router_->Search(req);
        if (!rep.status.ok())
            return Result<std::vector<SearchHit>>::Err(rep.status);
        return Result<std::vector<SearchHit>>::Ok(std::move(rep.hits));
    }

    Status PomaiDB::Flush() { return router_->FlushAll(); }

    Result<std::vector<core::ShardStatsSnapshot>> PomaiDB::Stats()
    {
        if (!router_)
            return Result<std::vector<core::ShardStatsSnapshot>>::Err(Status::Internal("db not started"));
        return Result<std::vector<core::ShardStatsSnapshot>>::Ok(router_->GetStats());
    }

} // namespace pomai