#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <atomic>

#include <pomai/api/options.h>
#include <pomai/api/search.h>
#include <pomai/api/scan.h>
#include <pomai/core/membrane.h>
#include <pomai/core/status.h>
#include <pomai/util/logger.h>
#include <pomai/concurrency/index_build_pool.h>

namespace pomai
{

    class PomaiDB
    {
    public:
        struct Metrics
        {
            std::atomic<std::uint64_t> rejected_upsert_batches_total{0};
        };

        // Constructor: optional logger for diagnostics.
        explicit PomaiDB(DbOptions opt, Logger *logger = nullptr);
        ~PomaiDB() { Stop(); } // Ensure Stop is called

        Status Start();
        Status Stop();

        std::future<Result<Lsn>> Upsert(Id id, Vector vec, bool wait_durable = true);
        std::future<Result<Lsn>> UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable = true);

        Result<SearchResponse> Search(const SearchRequest &req) const;
        Result<ScanResponse> Scan(const ScanRequest &req) const;
        std::size_t TotalApproxCountUnsafe() const;
        std::future<Result<bool>> RequestCheckpoint();
        void SetProbeCount(std::size_t p);
        std::string GetStats() const;

        std::future<Result<bool>> RecomputeCentroids(std::size_t k, std::size_t total_samples = 4096);

    private:
        static std::size_t AutoIndexBuildThreads();

    private:
        DbOptions opt_;

        // Critical: membrane_ must be declared BEFORE build_pool_
        // so that build_pool_ is destroyed FIRST (reverse declaration order).
        // This prevents Shards (owned by membrane) from being destroyed while
        // build_pool_ workers are still running jobs that access them.
        std::unique_ptr<MembraneRouter> membrane_;
        std::unique_ptr<IndexBuildPool> build_pool_;

        bool started_{false};

        Logger *logger_{nullptr};

        Metrics metrics_{};
    };

} // namespace pomai