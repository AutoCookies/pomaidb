#pragma once
#include <future>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "membrane.h"
#include "types.h"
#include "server/config.h"
#include "index_build_pool.h"

namespace pomai
{

    struct DbOptions
    {
        std::size_t dim{384};
        Metric metric{Metric::Cosine};
        std::size_t shards{4};
        std::size_t shard_queue_capacity{1024};
        std::string wal_dir{"./data"};
        pomai::server::WhisperConfig whisper;

        // New: index build pool threads (0 = auto)
        std::size_t index_build_threads{0};
        bool allow_sync_on_append{true};

        // Centroid persistence
        std::string centroids_path{};
        MembraneRouter::CentroidsLoadMode centroids_load_mode{MembraneRouter::CentroidsLoadMode::Auto};

        // NEW: search pool worker count (0 = auto)
        std::size_t search_pool_workers{0};
    };

    class PomaiDB
    {
    public:
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
    };

} // namespace pomai