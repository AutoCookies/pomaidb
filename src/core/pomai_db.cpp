#include "pomai/pomai_db.h"
#include <sstream>
#include <utility>

namespace pomai
{

    PomaiDB::PomaiDB(DbOptions opt) : opt_(std::move(opt))
    {
        std::vector<std::unique_ptr<Shard>> shards;
        shards.reserve(opt_.shards);

        for (std::size_t i = 0; i < opt_.shards; ++i)
        {
            std::ostringstream ss;
            ss << "shard-" << i;

            shards.push_back(std::make_unique<Shard>(
                ss.str(),
                opt_.dim,
                opt_.shard_queue_capacity,
                opt_.wal_dir 
                ));
        }

        membrane_ = std::make_unique<MembraneRouter>(std::move(shards));
    }

    void PomaiDB::Start() { membrane_->Start(); }
    void PomaiDB::Stop() { membrane_->Stop(); }

    std::future<Lsn> PomaiDB::Upsert(Id id, Vector vec, bool wait_durable)
    {
        return membrane_->Upsert(id, std::move(vec), wait_durable);
    }

    std::future<Lsn> PomaiDB::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        return membrane_->UpsertBatch(std::move(batch), wait_durable);
    }

    SearchResponse PomaiDB::Search(const SearchRequest &req) const
    {
        return membrane_->Search(req);
    }

    std::size_t PomaiDB::TotalApproxCountUnsafe() const
    {
        return membrane_->TotalApproxCountUnsafe();
    }

} // namespace pomai
