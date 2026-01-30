#pragma once
#include <vector>
#include <memory>
#include <future>
#include <string>
#include <deque>
#include <optional>
#include <functional>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <chrono>
#include <pomai/core/shard.h>
#include <pomai/api/options.h>
#include <pomai/api/scan.h>
#include <pomai/index/whispergrain.h>
#include <pomai/core/spatial_router.h>
#include <pomai/concurrency/search_thread_pool.h> // new
#include <pomai/concurrency/completion_executor.h>
#include <pomai/core/status.h>
#include <pomai/util/logger.h>

namespace pomai
{

    class MembraneRouter
    {
    public:
        using CentroidsLoadMode = pomai::CentroidsLoadMode;

        struct FilterConfig
        {
            std::size_t filtered_candidate_k{5000};
            std::uint32_t filter_expand_factor{4};
            std::uint32_t filter_max_visits{20000};
            std::uint64_t filter_time_budget_us{5000};
            std::size_t max_filtered_candidate_k{20000};
            std::uint32_t max_filter_graph_ef{2048};
            std::size_t max_filter_visits{100000};
            std::uint64_t max_filter_time_budget_us{50000};
            std::size_t filter_max_retries{3};
            std::size_t tag_dictionary_max_size{100000};
            std::size_t max_tags_per_vector{32};
            std::size_t max_filter_tags{64};

            static FilterConfig Default();
        };

        // Added 'search_pool_workers' parameter so callers can tune the bounded search pool.
        //  - search_pool_workers == 0 -> auto (hw concurrency, capped)
        explicit MembraneRouter(std::vector<std::unique_ptr<Shard>> shards,
                                pomai::WhisperConfig w_cfg,
                                std::size_t dim,
                                Metric metric,
                                std::size_t search_pool_workers,
                                std::size_t search_timeout_ms,
                                std::size_t scan_batch_cap,
                                std::size_t scan_id_order_max_rows,
                                FilterConfig filter_config,
                                std::function<void()> on_rejected_upsert,
                                Logger *logger = nullptr);

        Status Start();
        Status Stop();

        std::future<Result<Lsn>> Upsert(Id id, Vector vec, bool wait_durable = true);
        std::future<Result<Lsn>> UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable = true);
        Result<SearchResponse> Search(const SearchRequest &req) const;
        Result<ScanResponse> Scan(const ScanRequest &req) const;

        std::size_t ShardCount() const { return shards_.size(); }
        std::size_t TotalApproxCountUnsafe() const;
        std::future<Result<bool>> RequestCheckpoint();
        bool RecoverFromStorage(const std::string &db_dir, std::string *err = nullptr);
        void SetDbDir(const std::string &dir) { db_dir_ = dir; }

        // Admin helpers to manage routing / centroids
        // - ConfigureCentroids: replace router centroids (atomic) and assign centroids to shards.
        // - SetProbeCount: number of centroid probes per query (multi-probe).
        void ConfigureCentroids(const std::vector<Vector> &centroids);
        void SetProbeCount(std::size_t p);
        std::vector<Vector> SnapshotCentroids() const;
        double SearchQueueAvgLatencyMs() const;
        std::uint64_t SearchOverloadCount() const { return search_overload_.load(std::memory_order_relaxed); }
        std::uint64_t SearchInlineCount() const { return search_inline_.load(std::memory_order_relaxed); }
        std::uint64_t SearchPartialCount() const { return search_partial_.load(std::memory_order_relaxed); }
        std::uint64_t SearchBudgetTimeHitCount() const { return search_budget_time_hit_.load(std::memory_order_relaxed); }
        std::uint64_t SearchBudgetVisitHitCount() const { return search_budget_visit_hit_.load(std::memory_order_relaxed); }
        std::uint64_t SearchBudgetExhaustedCount() const { return search_budget_exhausted_.load(std::memory_order_relaxed); }
        std::uint64_t ScanItemsPerSec() const { return scan_items_per_sec_.load(std::memory_order_relaxed); }
        std::size_t CompactionBacklog() const;
        std::uint64_t LastCompactionDurationMs() const;
        std::uint64_t LastCheckpointLsn() const { return last_checkpoint_lsn_.load(std::memory_order_relaxed); }
        std::vector<std::uint64_t> WalLagLsns() const;

        struct HotspotInfo
        {
            std::size_t shard_id{0};
            std::size_t centroid_idx{0};
            double ratio{0.0};
        };

        std::optional<HotspotInfo> CurrentHotspot() const;

        // Compute centroids from samples across shards and install them atomically.
        // - k: number of centroids to produce.
        // - total_samples: target total number of sample vectors aggregated across all shards.
        // Returns true on success.
        Result<void> ComputeAndConfigureCentroids(std::size_t k, std::size_t total_samples = 4096);

        Status LoadCentroidsFromFile(const std::string &path);
        Status SaveCentroidsToFile(const std::string &path) const;
        void SetCentroidsFilePath(const std::string &path);
        void SetCentroidsLoadMode(CentroidsLoadMode mode);
        bool HasCentroids() const;
        bool ScheduleCompletion(std::function<void()> fn, std::chrono::steady_clock::duration delay = {});
        void DetachBuildPool();

    private:
        // Legacy id-based pick (used as fallback)
        std::size_t PickShardById(Id id) const;

        // Smart pick: prefer vector-based routing if vec_opt provided and router configured
        std::size_t PickShard(Id id, const Vector *vec_opt = nullptr) const;

        Result<Metadata> NormalizeMetadata(const Metadata &meta);
        Result<std::shared_ptr<const Filter>> NormalizeFilter(const SearchRequest &req) const;

        std::vector<std::unique_ptr<Shard>> shards_;
        mutable pomai::ai::WhisperGrain brain_;

        // Spatial router for centroid-based routing
        SpatialRouter router_;

        // Mapping from centroid idx -> shard id (simple round-robin or custom mapping)
        std::vector<std::size_t> centroid_to_shard_;

        // How many centroid neighbors to probe per query. Default 2 (multi-probe).
        std::size_t probe_P_{6};

        std::string centroids_path_;
        CentroidsLoadMode centroids_load_mode_{CentroidsLoadMode::Auto};
        std::size_t dim_{0};
        Metric metric_{Metric::L2};
        std::string db_dir_;
        std::size_t search_timeout_ms_{500};
        FilterConfig filter_config_{};

        // Thread-pool for bounded parallel search fanout. Mutable so Search() can be const.
        mutable SearchThreadPool search_pool_;
        mutable CompletionExecutor completion_;

        std::function<void()> on_rejected_upsert_;
        Logger *logger_{nullptr};

        mutable std::mutex hotspot_mu_;
        mutable std::optional<HotspotInfo> hotspot_;
        mutable std::size_t last_hotspot_shard_{static_cast<std::size_t>(-1)};
        mutable std::atomic<std::uint64_t> search_count_{0};
        mutable std::atomic<std::uint64_t> search_overload_{0};
        mutable std::atomic<std::uint64_t> search_inline_{0};
        mutable std::atomic<std::uint64_t> search_partial_{0};
        mutable std::atomic<std::uint64_t> search_budget_time_hit_{0};
        mutable std::atomic<std::uint64_t> search_budget_visit_hit_{0};
        mutable std::atomic<std::uint64_t> search_budget_exhausted_{0};

        struct ScanGrain
        {
            Seed::Snapshot snap;
            std::size_t shard_id{0};
        };

        struct ScanView
        {
            std::uint64_t epoch{0};
            std::vector<ScanGrain> grains;
            std::vector<std::size_t> grain_row_counts;
            std::vector<std::pair<std::size_t, std::size_t>> id_index;
            std::size_t total_rows{0};
        };

        std::shared_ptr<const ScanView> GetOrCreateView(ScanOrder order, ScanStatus &status) const;
        std::shared_ptr<const ScanView> FindView(std::uint64_t epoch) const;
        std::string EncodeCursor(std::uint64_t epoch, std::size_t grain, std::size_t row) const;
        bool DecodeCursor(const std::string &cursor, std::uint64_t &epoch, std::size_t &grain, std::size_t &row) const;

        mutable std::mutex scan_views_mu_;
        mutable std::deque<std::shared_ptr<const ScanView>> scan_views_;
        mutable std::atomic<std::uint64_t> scan_epoch_{1};
        std::size_t scan_batch_cap_{4096};
        std::size_t scan_id_order_max_rows_{1000000};
        mutable std::atomic<std::uint64_t> scan_items_per_sec_{0};
        mutable std::atomic<std::uint64_t> last_checkpoint_lsn_{0};
        mutable std::mutex scan_stats_mu_;
        mutable std::chrono::steady_clock::time_point scan_last_time_{};
        mutable std::uint64_t scan_last_count_{0};
        pomai::Logger* log_{nullptr};
    };

}
