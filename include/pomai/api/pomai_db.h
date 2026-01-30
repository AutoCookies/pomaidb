#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

#include <pomai/api/options.h>
#include <pomai/api/search.h>
#include <pomai/core/membrane.h>
#include <pomai/util/index_build_pool.h>

namespace pomai
{

    class PomaiDB
    {
    public:
        struct Metrics
        {
            std::atomic<std::uint64_t> rejected_upsert_batches_total{0};
        };

        // Accept optional logging callbacks so caller (server) can forward logs.
        using LogFn = std::function<void(const std::string &)>;

        // Constructor: optional info/error logging callbacks.
        explicit PomaiDB(DbOptions opt, LogFn info = {}, LogFn error = {});

        void Start();
        void Stop();

        std::future<Lsn> Upsert(Id id, Vector vec, bool wait_durable = true);
        std::future<Lsn> UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable = true);

        SearchResponse Search(const SearchRequest &req) const;
        std::size_t TotalApproxCountUnsafe() const;
        std::future<bool> RequestCheckpoint();
        void SetProbeCount(std::size_t p);
        std::string GetStats() const;

        // Trigger recompute of routing centroids across shards.
        // - k: desired number of centroids (e.g., shards * 8)
        // - total_samples: total vectors to sample across all shards (e.g., shards * 1024)
        // Returns a future that resolves to true on success.
        std::future<bool> RecomputeCentroids(std::size_t k, std::size_t total_samples = 4096);

    private:
        static std::size_t AutoIndexBuildThreads();

    private:
        DbOptions opt_;

        std::unique_ptr<IndexBuildPool> build_pool_;
        std::unique_ptr<MembraneRouter> membrane_;
        bool started_{false};

        // optional logging callbacks (forwarded into Shards)
        LogFn log_info_;
        LogFn log_error_;

        Metrics metrics_{};
    };

} // namespace pomai
