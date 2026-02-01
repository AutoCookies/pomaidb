#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>

#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/core/router.h"
#include "pomai/core/stats.h"

namespace pomai
{
    struct UpsertBatchResult
    {
        Status status;
        std::size_t ok_count{0};
        std::size_t fail_count{0};
    };

    struct DbOptions
    {
        std::filesystem::path dir{"./pomai_data"};
        std::uint32_t num_shards{4};
        std::uint32_t vector_dim{0}; // 0 = allow any
        std::size_t shard_inbox_capacity{4096};
        core::FsyncPolicy fsync_policy{core::FsyncPolicy::Never};
    };

    class PomaiDB final
    {
    public:
        explicit PomaiDB(DbOptions opt);
        ~PomaiDB();

        Status Start();
        void Stop();

        Status Upsert(VectorId id, VectorData vec);
        Result<std::vector<SearchHit>> Search(VectorData query, std::uint32_t topk);

        Status Flush();
        Result<std::vector<core::ShardStatsSnapshot>> Stats();
        UpsertBatchResult UpsertBatch(std::vector<pomai::UpsertItem> items);

    private:
        DbOptions opt_;
        std::unique_ptr<core::Router> router_;
    };

} // namespace pomai