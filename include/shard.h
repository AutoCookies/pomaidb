#pragma once
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <functional>
#include <optional>
#include <atomic>

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
        bool is_checkpoint{false};
        std::optional<std::promise<bool>> checkpoint_done;
        bool is_emergency_freeze{false};
    };

    struct GrainIndex
    {
        std::size_t dim;
        std::vector<Vector> centroids;
        std::vector<std::size_t> offsets;
        std::vector<std::uint32_t> postings;
    };

    struct IndexedSegment
    {
        Seed::Snapshot snap;
        std::shared_ptr<const GrainIndex> grains;
        std::shared_ptr<const pomai::core::OrbitIndex> index;
    };

    struct ShardState
    {
        std::vector<IndexedSegment> segments;
        Seed::Snapshot live_snap;
        std::shared_ptr<const GrainIndex> live_grains;
    };

    class Shard
    {
    public:
        using LogFn = std::function<void(const std::string &msg)>;

        Shard(std::string name,
              std::size_t dim,
              std::size_t queue_cap,
              std::string wal_dir,
              LogFn info = nullptr,
              LogFn error = nullptr);
        ~Shard();

        void Start();
        void Stop();

        std::future<Lsn> EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable);
        std::future<bool> RequestCheckpoint();
        void RequestEmergencyFreeze();

        SearchResponse Search(const SearchRequest &req, const pomai::ai::Budget &budget) const;
        std::size_t ApproxCountUnsafe() const;
        std::vector<Vector> SampleVectors(std::size_t max_samples) const;

    private:
        void RunLoop();
        void MaybeFreezeSegment();
        void PublishState(std::shared_ptr<const ShardState> next);
        void ScheduleLiveGrainBuild(const Seed::Snapshot &snap);

        void AttachIndex(std::size_t segment_pos,
                         Seed::Snapshot snap,
                         std::shared_ptr<pomai::core::OrbitIndex> idx,
                         std::shared_ptr<GrainIndex> grains = nullptr);
        void AttachLiveGrains(Seed::Snapshot snap, std::shared_ptr<GrainIndex> grains);

        static void MergeTopK(SearchResponse &out, const SearchResponse &in, std::size_t k);
        std::shared_ptr<GrainIndex> BuildGrainIndex(const Seed::Snapshot &snap) const;
        SearchResponse SearchGrains(const Seed::Snapshot &snap, const GrainIndex &grains, const SearchRequest &req, const pomai::ai::Budget &budget) const;

    private:
        std::string name_;
        std::string wal_dir_;
        Wal wal_;
        Seed seed_;
        BoundedQueue<UpsertTask> ingest_q_;
        IndexBuildPool *build_pool_{nullptr};
        LogFn log_info_;
        LogFn log_error_;
        mutable std::mutex writer_mu_;
        std::atomic<std::shared_ptr<const ShardState>> state_{nullptr};
        std::thread owner_;
        static constexpr std::size_t kFreezeEveryVectors = 50000;
        std::size_t since_freeze_{0};
        static constexpr std::size_t kMaxSegments = 64;
        static constexpr std::size_t kPublishLiveEveryVectors = 10000;
        std::size_t since_live_publish_{0};
        std::atomic<bool> emergency_freeze_pending_{false};

    public:
        void SetIndexBuildPool(IndexBuildPool *pool) { build_pool_ = pool; }
    };
}
