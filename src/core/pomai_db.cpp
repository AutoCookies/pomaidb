#include "pomai_db.h"

#include <sstream>
#include <thread>
#include <utility>
#include <future>
#include <chrono>

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
        if (opt_.centroids_path.empty() && opt_.centroids_load_mode != MembraneRouter::CentroidsLoadMode::None)
        {
            opt_.centroids_path = opt_.wal_dir + "/centroids.bin";
        }

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

        membrane_ = std::make_unique<MembraneRouter>(std::move(shards), opt_.whisper, opt_.dim);
        membrane_->SetCentroidsFilePath(opt_.centroids_path);
        membrane_->SetCentroidsLoadMode(opt_.centroids_load_mode);
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

        if (opt_.centroids_load_mode == MembraneRouter::CentroidsLoadMode::None)
            return;

        if (opt_.centroids_load_mode != MembraneRouter::CentroidsLoadMode::Async &&
            membrane_->HasCentroids())
        {
            if (log_info_)
                log_info_("Centroids already loaded; skipping background recompute");
            return;
        }

        // Spawn a background non-blocking centroid recompute after startup to avoid blocking Start().
        // This provides a reasonable default: k = shards * 8, samples = shards * 1024.
        // If you prefer control, call RecomputeCentroids(...) from the embedding app or admin API.
        std::thread([this]()
                    {
            // small delay to let shards finish replay/freeze work
            std::this_thread::sleep_for(std::chrono::seconds(1));

            const std::size_t shards = membrane_->ShardCount();
            if (shards == 0)
                return;

            const std::size_t k = shards * 8;
            const std::size_t total_samples = shards * 1024;

            try
            {
                auto fut = RecomputeCentroids(k, total_samples);
                bool ok = fut.get();
                if (log_info_)
                {
                    if (ok)
                        log_info_("Centroid recompute succeeded at startup");
                    else
                        log_info_("Centroid recompute failed at startup");
                }
            }
            catch (...)
            {
                if (log_error_)
                    log_error_("Centroid recompute threw exception at startup");
            } })
            .detach();
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
        // Enforce collection-level policy: if the collection disallows sync-on-append,
        // ignore client's request for synchronous durability.
        bool effective_wait = wait_durable && opt_.allow_sync_on_append;
        if (log_info_)
        {
            // Optional debug log; keep concise in production
            // log_info_("Upsert: client_wait=" + std::to_string(wait_durable) +
            //           " effective_wait=" + std::to_string(effective_wait));
        }
        return membrane_->Upsert(id, std::move(vec), effective_wait);
    }

    std::future<Lsn> PomaiDB::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        // Enforce collection-level policy as above.
        bool effective_wait = wait_durable && opt_.allow_sync_on_append;
        if (log_info_)
        {
            // Optional debug log
            // log_info_("UpsertBatch: client_wait=" + std::to_string(wait_durable) +
            //           " effective_wait=" + std::to_string(effective_wait) +
            //           " batch_size=" + std::to_string(batch.size()));
        }
        return membrane_->UpsertBatch(std::move(batch), effective_wait);
    }

    SearchResponse PomaiDB::Search(const SearchRequest &req) const
    {
        return membrane_->Search(req);
    }

    std::size_t PomaiDB::TotalApproxCountUnsafe() const
    {
        return membrane_->TotalApproxCountUnsafe();
    }

    std::future<bool> PomaiDB::RequestCheckpoint()
    {
        return membrane_->RequestCheckpoint();
    }

    // Trigger recompute of centroids (samples shards, runs k-means, installs centroids).
    std::future<bool> PomaiDB::RecomputeCentroids(std::size_t k, std::size_t total_samples)
    {
        if (k == 0)
        {
            std::promise<bool> p;
            p.set_value(false);
            return p.get_future();
        }

        // Delegate to MembraneRouter if it exposes compute helper; otherwise perform orchestration here by
        // asking MembraneRouter to expose a method for sampling/building. To keep this API stable we call
        // a MembraneRouter helper `ComputeAndConfigureCentroids` if available.
        //
        // The MembraneRouter implementation may either:
        //  - provide ComputeAndConfigureCentroids(k, total_samples) (recommended), or
        //  - you can implement this PomaiDB method to gather samples from shards (Shard::SampleVectors)
        //    and then call SpatialRouter::BuildKMeans(...) and membrane_->ConfigureCentroids(...).
        //
        // Implementation below will perform sampling via MembraneRouter if it has ComputeAndConfigureCentroids;
        // otherwise it will fallback to a simple orchestration by calling an expected helper on MembraneRouter.
        //
        return std::async(std::launch::async, [this, k, total_samples]() -> bool
                          {
            try
            {
                // Prefer if MembraneRouter provides a direct compute method.
#if 1
                // If your MembraneRouter implements ComputeAndConfigureCentroids(k, total_samples),
                // this will compile and run. If not, replace this block with the fallback below.
                return membrane_->ComputeAndConfigureCentroids(k, total_samples);
#else
                // Fallback orchestration (if MembraneRouter does NOT implement ComputeAndConfigureCentroids).
                // Collect samples from shards via an imagined MembraneRouter->SampleAllShards(per_shard)
                // or by exposing shards; if unavailable, you'll need to add a small helper to MembraneRouter.
                //
                // Pseudocode:
                //   per_shard = max(1, total_samples / membrane_->ShardCount());
                //   aggregate_samples = membrane_->CollectShardSamples(per_shard);
                //   centroids = SpatialRouter::BuildKMeans(aggregate_samples, k, 10);
                //   membrane_->ConfigureCentroids(centroids);
                //   return true;
                return false;
#endif
            }
            catch (const std::exception &e)
            {
                if (log_error_)
                    log_error_(std::string("RecomputeCentroids failed: ") + e.what());
                return false;
            }
            catch (...)
            {
                if (log_error_)
                    log_error_("RecomputeCentroids failed: unknown exception");
                return false;
            } });
    }

} // namespace pomai
