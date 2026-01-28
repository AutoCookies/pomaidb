#include "pomai_db.h"

#include <sstream>
#include <thread>
#include <utility>
#include <future>
#include <chrono>

#include "memory_manager.h"

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

        // Pass search_pool_workers through to MembraneRouter so operator can tune pool size.
        membrane_ = std::make_unique<MembraneRouter>(std::move(shards),
                                                     opt_.whisper,
                                                     opt_.dim,
                                                     opt_.search_pool_workers,
                                                     opt_.search_timeout_ms,
                                                     [this]()
                                                     {
                                                         metrics_.rejected_upsert_batches_total.fetch_add(1, std::memory_order_relaxed);
                                                     });
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

    void PomaiDB::SetProbeCount(std::size_t p)
    {
        membrane_->SetProbeCount(p);
    }

    std::string PomaiDB::GetStats() const
    {
        std::ostringstream ss;
        ss << "{";
        ss << "\"rejected_upsert_batches_total\":" << metrics_.rejected_upsert_batches_total.load(std::memory_order_relaxed);
        ss << ",\"search_queue_avg_latency_ms\":" << membrane_->SearchQueueAvgLatencyMs();
        ss << ",\"active_index_builds\":" << (build_pool_ ? build_pool_->ActiveBuilds() : 0);

        const std::size_t snapshot_bytes = MemoryManager::Instance().Usage(MemoryManager::Pool::Search);
        const std::size_t index_bytes = MemoryManager::Instance().Usage(MemoryManager::Pool::Indexing);
        const std::size_t memtable_bytes = MemoryManager::Instance().Usage(MemoryManager::Pool::Memtable);
        ss << ",\"memory_usage_bytes\":{";
        ss << "\"snapshot\":" << snapshot_bytes;
        ss << ",\"index\":" << index_bytes;
        ss << ",\"memtable\":" << memtable_bytes;
        ss << ",\"total\":" << (snapshot_bytes + index_bytes + memtable_bytes);
        ss << "}";

        auto hotspot = membrane_->CurrentHotspot();
        ss << ",\"hotspot\":{";
        if (hotspot)
        {
            ss << "\"detected\":true";
            ss << ",\"shard_id\":" << hotspot->shard_id;
            ss << ",\"centroid_idx\":" << hotspot->centroid_idx;
            ss << ",\"ratio\":" << hotspot->ratio;
        }
        else
        {
            ss << "\"detected\":false";
        }
        ss << "}";
        ss << "}";
        return ss.str();
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

        return std::async(std::launch::async, [this, k, total_samples]() -> bool
                          {
            try
            {
                return membrane_->ComputeAndConfigureCentroids(k, total_samples);
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
