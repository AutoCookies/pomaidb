#include "pomai_db.h"

#include <sstream>
#include <thread>
#include <utility>
#include <future>
#include <chrono>

#include "memory_manager.h"

namespace pomai
{
    // Helper nội bộ để giới hạn số lượng thread tùy theo thiết bị
    static std::size_t clamp_workers(std::size_t x)
    {
        if (x == 0)
            return 1;
        if (x > 8)
            return 8; // Hard cap để tránh vắt kiệt CPU trên máy tính yếu
        return x;
    }

    std::size_t PomaiDB::AutoIndexBuildThreads()
    {
        // Tự động chọn số luồng xây dựng Index dựa trên phần cứng thực tế
        std::size_t hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 4;

        // Ưu tiên để lại tài nguyên cho tác vụ Search và Ingest
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

        // 1. Khởi tạo Index Build Pool (Xử lý hậu trường)
        std::size_t workers = opt_.index_build_threads;
        if (workers == 0)
            workers = AutoIndexBuildThreads();
        workers = clamp_workers(workers);

        build_pool_ = std::make_unique<IndexBuildPool>(workers);

        // 2. Khởi tạo các Grains (Shards)
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

            // Tiêm con trỏ Pool vào từng Shard để tự động hóa việc Indexing
            sh->SetIndexBuildPool(build_pool_.get());
            shards.push_back(std::move(sh));
        }

        // 3. Khởi tạo Membrane (Bộ điều hướng thần kinh)
        // Tích hợp Callback để đếm số lượng Batch bị từ chối do áp lực bộ nhớ (Backpressure)
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

        if (build_pool_)
            build_pool_->Start();
        membrane_->Start();

        if (opt_.centroids_load_mode == MembraneRouter::CentroidsLoadMode::None)
            return;

        // Tự động tính toán lại Centroids ở chế độ nền sau khi khởi động
        std::thread([this]()
                    {
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Đợi Shard hoàn thành replay WAL

            const std::size_t shards = membrane_->ShardCount();
            if (shards == 0) return;

            try {
                // Mặc định: k = shards * 8 để đảm bảo phân phối đều
                auto fut = RecomputeCentroids(shards * 8, shards * 1024);
                if (fut.get() && log_info_) log_info_("Centroid recompute succeeded at startup");
            } catch (...) {
                if (log_error_) log_error_("Centroid recompute exception at startup");
            } })
            .detach();
    }

    void PomaiDB::Stop()
    {
        if (!started_)
            return;
        started_ = false;
        membrane_->Stop();
        if (build_pool_)
            build_pool_->Stop();
    }

    // --- Ingest Path với Backpressure ---

    std::future<Lsn> PomaiDB::Upsert(Id id, Vector vec, bool wait_durable)
    {
        // Chuyển đổi thành Batch để tận dụng logic kiểm soát bộ nhớ
        std::vector<UpsertRequest> batch;
        batch.push_back({id, std::move(vec)});
        return UpsertBatch(std::move(batch), wait_durable);
    }

    std::future<Lsn> PomaiDB::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        // BACKPRESSURE LOGIC: Ngăn chặn OOM bằng cách kiểm tra MemoryManager
        // Nếu bộ nhớ vượt ngưỡng Soft Watermark (70%), thực hiện làm chậm (Throttle)
        if (MemoryManager::Instance().AtOrAboveSoftWatermark())
        {
            // Tạm dừng 20ms để IndexBuildPool có thời gian giải phóng Snapshot
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        // Nếu chạm ngưỡng Hard Watermark (90%), từ chối ngay lập tức để tránh bị OS Kill
        std::size_t estimated_bytes = batch.size() * (opt_.dim * sizeof(float) + sizeof(Id));
        if (!MemoryManager::Instance().CanAllocate(estimated_bytes))
        {
            metrics_.rejected_upsert_batches_total.fetch_add(1, std::memory_order_relaxed);
            std::promise<Lsn> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("PomaiDB: Hard memory limit reached. Batch rejected.")));
            return p.get_future();
        }

        bool effective_wait = wait_durable && opt_.allow_sync_on_append;
        return membrane_->UpsertBatch(std::move(batch), effective_wait);
    }

    // --- Search Path ---

    SearchResponse PomaiDB::Search(const SearchRequest &req) const
    {
        return membrane_->Search(req);
    }

    void PomaiDB::SetProbeCount(std::size_t p)
    {
        membrane_->SetProbeCount(p);
    }

    // --- Observability (Insight vào quái vật) ---

    std::string PomaiDB::GetStats() const
    {
        std::ostringstream ss;
        auto &mm = MemoryManager::Instance();

        ss << "{";
        ss << "\"rejected_upsert_batches_total\":" << metrics_.rejected_upsert_batches_total.load(std::memory_order_relaxed);
        ss << ",\"search_queue_avg_latency_ms\":" << membrane_->SearchQueueAvgLatencyMs();
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
            try {
                return membrane_->ComputeAndConfigureCentroids(k, total_samples);
            } catch (const std::exception &e) {
                if (log_error_) log_error_(std::string("RecomputeCentroids failed: ") + e.what());
                return false;
            } catch (...) {
                return false;
            } });
    }

    std::size_t PomaiDB::TotalApproxCountUnsafe() const { return membrane_->TotalApproxCountUnsafe(); }
    std::future<bool> PomaiDB::RequestCheckpoint() { return membrane_->RequestCheckpoint(); }

} // namespace pomai
