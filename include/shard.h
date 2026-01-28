#pragma once
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <optional>

#include "types.h"
#include "wal.h"
#include "seed.h"
#include "orbit_index.h"
#include "bounded_queue.h"
#include "whispergrain.h"
#include "index_build_pool.h"

namespace pomai
{

    struct UpsertTask
    {
        std::vector<UpsertRequest> batch;
        bool wait_durable{true};
        std::promise<Lsn> done;

        // Checkpoint control
        bool is_checkpoint{false};
        std::optional<std::promise<bool>> checkpoint_done;
    };

    struct IndexedSegment
    {
        Seed::Snapshot snap;
        std::shared_ptr<pomai::core::OrbitIndex> index; // null until built
    };

    class Shard
    {
    public:
        // Logging callback type: accept a single formatted message string.
        using LogFn = std::function<void(const std::string &msg)>;

        // Added optional logging callbacks so shard can emit startup/replay logs
        // to the server logger without depending on the Logger type at link time.
        Shard(std::string name,
              std::size_t dim,
              std::size_t queue_cap,
              std::string wal_dir,
              LogFn info = {},
              LogFn error = {});
        ~Shard();

        void SetIndexBuildPool(IndexBuildPool *pool) { build_pool_ = pool; }

        void Start();
        void Stop();

        std::future<Lsn> EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable);

        // Request checkpoint (snapshot + wal truncation)
        std::future<bool> RequestCheckpoint();

        // Sample up to max_samples vectors from this shard (from frozen segments and live memtable).
        // Non-blocking for writers (uses snapshot copies under short lock).
        std::vector<Vector> SampleVectors(std::size_t max_samples) const;

        SearchResponse Search(const SearchRequest &req, const pomai::ai::Budget &budget) const;
        std::size_t ApproxCountUnsafe() const;

    private:
        void RunLoop();
        void MaybeFreezeSegment();

        void AttachIndex(std::size_t segment_pos,
                         Seed::Snapshot snap,
                         std::shared_ptr<pomai::core::OrbitIndex> idx);

        static void MergeTopK(SearchResponse &out, const SearchResponse &in, std::size_t k);

    private:
        std::string name_;
        std::string wal_dir_;

        Wal wal_;
        Seed seed_;
        BoundedQueue<UpsertTask> ingest_q_;

        IndexBuildPool *build_pool_{nullptr};

        // Optional logging callbacks to avoid linking server::Logger into core.
        LogFn log_info_;
        LogFn log_error_;

        mutable std::mutex state_mu_;
        std::vector<IndexedSegment> segments_;
        Seed::Snapshot live_snap_;

        std::thread owner_;

        static constexpr std::size_t kFreezeEveryVectors = 50'000;
        std::size_t since_freeze_{0};

        static constexpr std::size_t kPublishLiveEveryVectors = 10'000;
        std::size_t since_live_publish_{0};
    };

} // namespace pomai