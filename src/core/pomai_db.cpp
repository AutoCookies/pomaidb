#include "pomai_db.h"

#include <sstream>
#include <thread>
#include <utility>

namespace pomai
{

    static std::size_t clamp_workers(std::size_t x)
    {
        if (x == 0)
            return 1;
        if (x > 8)
            return 8; // hard cap to avoid runaway CPU on small devices
        return x;
    }

    std::size_t PomaiDB::AutoIndexBuildThreads()
    {
        // Production default for your "single user local DB":
        // Keep index build from stealing CPU from ingest/search.
        std::size_t hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 4;

        // Weak device: 1 worker. Stronger: 2 workers.
        if (hc < 8)
            return 1;
        return 2;
    }

    PomaiDB::PomaiDB(DbOptions opt, LogFn info, LogFn error)
        : opt_(std::move(opt)), log_info_(std::move(info)), log_error_(std::move(error))
    {
        // Create global index build pool (not started yet)
        std::size_t workers = opt_.index_build_threads;
        if (workers == 0)
            workers = AutoIndexBuildThreads();
        workers = clamp_workers(workers);

        build_pool_ = std::make_unique<IndexBuildPool>(workers);

        // Construct shards
        std::vector<std::unique_ptr<Shard>> shards;
        shards.reserve(opt_.shards);

        for (std::size_t i = 0; i < opt_.shards; ++i)
        {
            std::ostringstream ss;
            ss << "shard-" << i;

            auto sh = std::make_unique<Shard>(
                ss.str(),
                opt_.dim,
                opt_.shard_queue_capacity,
                opt_.wal_dir,
                log_info_,
                log_error_);

            // Inject pool pointer
            sh->SetIndexBuildPool(build_pool_.get());

            shards.push_back(std::move(sh));
        }

        membrane_ = std::make_unique<MembraneRouter>(std::move(shards), opt_.whisper);
    }

    void PomaiDB::Start()
    {
        if (started_)
            return;
        started_ = true;

        // Start pool first so early freezes can enqueue
        if (build_pool_)
            build_pool_->Start();
        membrane_->Start();
    }

    void PomaiDB::Stop()
    {
        if (!started_)
            return;
        started_ = false;

        // Stop shards first (they stop enqueuing new index jobs)
        membrane_->Stop();

        // Then stop background pool
        if (build_pool_)
            build_pool_->Stop();
    }

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