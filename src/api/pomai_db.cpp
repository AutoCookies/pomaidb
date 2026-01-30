#include <pomai/api/pomai_db.h>
#include <sstream>
#include <thread>
#include <utility>
#include <future>
#include <chrono>
#include <pomai/concurrency/memory_manager.h>
#include <pomai/storage/file_util.h>

namespace pomai
{
    static std::size_t clamp_workers(std::size_t x)
    {
        if (x == 0)
            return 1;
        if (x > 8)
            return 8;
        return x;
    }

    std::size_t PomaiDB::AutoIndexBuildThreads()
    {
        std::size_t hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 4;
        if (hc < 8)
            return 1;
        return 2;
    }

    PomaiDB::PomaiDB(DbOptions opt, Logger *logger)
        : opt_(std::move(opt)), logger_(logger)
    {
        if (logger_ && opt_.debug_logging)
            logger_->EnableDebug(true);
        if (opt_.centroids_path.empty() && opt_.centroids_load_mode != MembraneRouter::CentroidsLoadMode::None)
        {
            opt_.centroids_path = opt_.wal_dir + "/centroids.bin";
        }

        std::size_t workers = opt_.index_build_threads;
        if (workers == 0)
            workers = AutoIndexBuildThreads();
        workers = clamp_workers(workers);
        build_pool_ = std::make_unique<IndexBuildPool>(workers);

        std::vector<std::unique_ptr<Shard>> shards;
        shards.reserve(opt_.shards);

        pomai::storage::DbPaths paths = pomai::storage::MakeDbPaths(opt_.wal_dir);
        pomai::storage::EnsureDbDirs(paths);
        for (std::size_t i = 0; i < opt_.shards; ++i)
        {
            std::ostringstream ss;
            ss << "shard-" << i;
            CompactionConfig compaction_cfg;
            compaction_cfg.level_fanout = opt_.level_fanout;
            compaction_cfg.max_concurrent_compactions = opt_.max_concurrent_compactions;
            compaction_cfg.compaction_trigger_threshold = opt_.compaction_trigger_threshold;
            auto sh = std::make_unique<Shard>(
                ss.str(),
                static_cast<std::uint32_t>(i),
                opt_.dim,
                opt_.shard_queue_capacity,
                paths.wal_dir,
                compaction_cfg,
                logger_);
            sh->SetIndexBuildPool(build_pool_.get());
            shards.push_back(std::move(sh));
        }

        MembraneRouter::FilterConfig filter_cfg = MembraneRouter::FilterConfig::Default();
        filter_cfg.filtered_candidate_k = opt_.filtered_candidate_k;
        filter_cfg.filter_expand_factor = opt_.filter_expand_factor;
        filter_cfg.filter_max_visits = opt_.filter_max_visits;
        filter_cfg.filter_time_budget_us = opt_.filter_time_budget_us;
        filter_cfg.max_filtered_candidate_k = opt_.max_filtered_candidate_k;
        filter_cfg.max_filter_graph_ef = opt_.max_filter_graph_ef;
        filter_cfg.max_filter_visits = opt_.max_filter_visits;
        filter_cfg.max_filter_time_budget_us = opt_.max_filter_time_budget_us;
        filter_cfg.filter_max_retries = opt_.filter_max_retries;
        filter_cfg.tag_dictionary_max_size = opt_.tag_dictionary_max_size;
        filter_cfg.max_tags_per_vector = opt_.max_tags_per_vector;
        filter_cfg.max_filter_tags = opt_.max_filter_tags;

        membrane_ = std::make_unique<MembraneRouter>(std::move(shards), opt_.whisper, opt_.dim, opt_.metric, opt_.search_pool_workers, opt_.search_timeout_ms, opt_.scan_batch_cap, opt_.scan_id_order_max_rows, filter_cfg, [this]()
                                                     { metrics_.rejected_upsert_batches_total.fetch_add(1, std::memory_order_relaxed); }, logger_);

        membrane_->SetCentroidsFilePath(opt_.centroids_path);
        membrane_->SetCentroidsLoadMode(opt_.centroids_load_mode);
        membrane_->SetDbDir(opt_.wal_dir);
    }

    Status PomaiDB::Start()
    {
        if (started_)
            return Status::Ok();
        started_ = true;

        if (build_pool_)
            build_pool_->Start();
        std::string err;
        if (!membrane_->RecoverFromStorage(opt_.wal_dir, &err))
        {
            if (logger_)
                logger_->Warn("db.recover", "Recovery failed: " + err);
        }
        auto st = membrane_->Start();
        if (!st.ok())
            return st;

        if (opt_.centroids_load_mode == MembraneRouter::CentroidsLoadMode::None)
            return Status::Ok();

        auto task = [this]()
        {
            const std::size_t shards = membrane_->ShardCount();
            if (shards == 0)
                return;
            try
            {
                auto fut = RecomputeCentroids(shards * 8, shards * 1024);
                fut.get();
            }
            catch (...)
            {
            }
        };
        if (!membrane_->ScheduleCompletion(task, std::chrono::seconds(1)))
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            task();
        }
        return Status::Ok();
    }

    Status PomaiDB::Stop()
    {
        Status st = Status::Ok();

        // 1. Nếu đang chạy, dừng Membrane trước (để flush dữ liệu vào pool)
        if (started_)
        {
            started_ = false;
            st = membrane_->Stop();
        }

        // 2. BẮT BUỘC: Ngắt kết nối Pool khỏi Shard.
        // Phải chạy VÔ ĐIỀU KIỆN (kể cả khi !started_), vì Destructor sẽ chạy sau này.
        // Nếu không làm bước này, ~Shard() sẽ truy cập vào con trỏ build_pool_ đã chết.
        if (membrane_)
            membrane_->DetachBuildPool();

        // 3. Dừng và dọn dẹp Pool
        if (build_pool_)
            build_pool_->Stop();

        return st;
    }

    std::future<Result<Lsn>> PomaiDB::Upsert(Id id, Vector vec, bool wait_durable)
    {
        std::vector<UpsertRequest> batch;
        batch.push_back({id, std::move(vec)});
        return UpsertBatch(std::move(batch), wait_durable);
    }

    std::future<Result<Lsn>> PomaiDB::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        if (MemoryManager::Instance().AtOrAboveSoftWatermark())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        std::size_t estimated_bytes = batch.size() * (opt_.dim * sizeof(float) + sizeof(Id));
        if (!MemoryManager::Instance().CanAllocate(estimated_bytes))
        {
            metrics_.rejected_upsert_batches_total.fetch_add(1, std::memory_order_relaxed);
            std::promise<Result<Lsn>> p;
            p.set_value(Result<Lsn>(Status::Exhausted("PomaiDB: Hard memory limit reached. Batch rejected.")));
            return p.get_future();
        }

        bool effective_wait = wait_durable && opt_.allow_sync_on_append;
        return membrane_->UpsertBatch(std::move(batch), effective_wait);
    }

    Result<SearchResponse> PomaiDB::Search(const SearchRequest &req) const
    {
        try
        {
            return membrane_->Search(req);
        }
        catch (const std::exception &e)
        {
            return Result<SearchResponse>(Status::Internal(std::string("search failed: ") + e.what()));
        }
        catch (...)
        {
            return Result<SearchResponse>(Status::Internal("search failed: unknown exception"));
        }
    }

    Result<ScanResponse> PomaiDB::Scan(const ScanRequest &req) const
    {
        try
        {
            return membrane_->Scan(req);
        }
        catch (const std::exception &e)
        {
            return Result<ScanResponse>(Status::Internal(std::string("scan failed: ") + e.what()));
        }
        catch (...)
        {
            return Result<ScanResponse>(Status::Internal("scan failed: unknown exception"));
        }
    }

    void PomaiDB::SetProbeCount(std::size_t p)
    {
        membrane_->SetProbeCount(p);
    }

    std::string PomaiDB::GetStats() const
    {
        std::ostringstream ss;
        auto &mm = MemoryManager::Instance();
        ss << "{";
        ss << "\"rejected_upsert_batches_total\":" << metrics_.rejected_upsert_batches_total.load(std::memory_order_relaxed);
        ss << ",\"search_queue_avg_latency_ms\":" << membrane_->SearchQueueAvgLatencyMs();
        ss << ",\"search_overload_total\":" << membrane_->SearchOverloadCount();
        ss << ",\"search_inline_total\":" << membrane_->SearchInlineCount();
        ss << ",\"search_partial_total\":" << membrane_->SearchPartialCount();
        ss << ",\"search_budget_time_hit_total\":" << membrane_->SearchBudgetTimeHitCount();
        ss << ",\"search_budget_visit_hit_total\":" << membrane_->SearchBudgetVisitHitCount();
        ss << ",\"search_budget_exhausted_total\":" << membrane_->SearchBudgetExhaustedCount();
        ss << ",\"scan_items_per_sec\":" << membrane_->ScanItemsPerSec();
        ss << ",\"compaction_backlog\":" << membrane_->CompactionBacklog();
        ss << ",\"last_compaction_ms\":" << membrane_->LastCompactionDurationMs();
        ss << ",\"last_checkpoint_lsn\":" << membrane_->LastCheckpointLsn();
        ss << ",\"wal_lag_lsns\":[";
        auto lags = membrane_->WalLagLsns();
        for (std::size_t i = 0; i < lags.size(); ++i)
        {
            if (i > 0)
                ss << ",";
            ss << lags[i];
        }
        ss << "]";
        ss << ",\"active_index_builds\":" << (build_pool_ ? build_pool_->ActiveBuilds() : 0);
        ss << ",\"memory_usage_bytes\":{";
        ss << "\"snapshot\":" << mm.Usage(MemoryManager::Pool::Search);
        ss << ",\"index\":" << mm.Usage(MemoryManager::Pool::Indexing);
        ss << ",\"memtable\":" << mm.Usage(MemoryManager::Pool::Memtable);
        ss << ",\"total\":" << mm.TotalUsage();
        ss << ",\"limit\":" << mm.HardWatermarkBytes();
        ss << "}";
        auto hotspot = membrane_->CurrentHotspot();
        ss << ",\"hotspot\":{";
        if (hotspot)
        {
            ss << "\"detected\":true,\"shard_id\":" << hotspot->shard_id
               << ",\"ratio\":" << hotspot->ratio;
        }
        else
        {
            ss << "\"detected\":false";
        }
        ss << "}}";
        return ss.str();
    }

    std::future<Result<bool>> PomaiDB::RecomputeCentroids(std::size_t k, std::size_t total_samples)
    {
        if (k == 0)
        {
            std::promise<Result<bool>> p;
            p.set_value(Result<bool>(Status::Invalid("RecomputeCentroids requires k > 0")));
            return p.get_future();
        }

        return std::async(std::launch::async, [this, k, total_samples]() -> Result<bool>
                          {
            try {
                auto flush_fut = membrane_->RequestCheckpoint();
                auto flush_res = flush_fut.get();
                if (!flush_res.ok())
                    return flush_res.status();
                auto res = membrane_->ComputeAndConfigureCentroids(k, total_samples);
                if (!res.ok())
                    return res.status();
                return Result<bool>(true);
            } catch (const std::exception &e) {
                if (logger_) logger_->Error("db.centroids", std::string("RecomputeCentroids failed: ") + e.what());
                return Result<bool>(Status::Internal(std::string("RecomputeCentroids failed: ") + e.what()));
            } catch (...) {
                if (logger_) logger_->Error("db.centroids", "RecomputeCentroids failed: unknown exception");
                return Result<bool>(Status::Internal("RecomputeCentroids failed: unknown exception"));
            } });
    }

    std::size_t PomaiDB::TotalApproxCountUnsafe() const { return membrane_->TotalApproxCountUnsafe(); }
    std::future<Result<bool>> PomaiDB::RequestCheckpoint() { return membrane_->RequestCheckpoint(); }

}
