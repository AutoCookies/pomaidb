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

#include <pomai/core/types.h>
#include <pomai/core/status.h>
#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>
#include <pomai/index/orbit_index.h>
#include <pomai/concurrency/bounded_queue.h>
#include <pomai/index/whispergrain.h>
#include <pomai/concurrency/index_build_pool.h>
#include <pomai/util/logger.h>

namespace pomai
{
    struct ShardCheckpointState
    {
        std::uint32_t shard_id{0};
        Seed::PersistedState live;
        std::vector<Seed::PersistedState> segments;
        Lsn durable_lsn{0};
    };

    struct UpsertTask
    {
        std::vector<UpsertRequest> batch;
        bool wait_durable{true};
        std::promise<Result<Lsn>> done;
        bool is_checkpoint{false};
        bool is_checkpoint_state{false};
        std::optional<std::promise<Result<bool>>> checkpoint_done;
        std::optional<std::promise<Result<ShardCheckpointState>>> checkpoint_state_done;
        bool is_emergency_freeze{false};
    };

    struct GrainIndex
    {
        std::size_t dim;
        std::vector<Vector> centroids;
        std::vector<std::size_t> offsets;
        std::vector<std::uint32_t> postings;
        std::vector<std::size_t> namespace_offsets;
        std::vector<std::uint32_t> namespace_ids;
    };

    struct IndexedSegment
    {
        Seed::Snapshot snap;
        std::shared_ptr<const GrainIndex> grains;
        std::shared_ptr<const pomai::core::OrbitIndex> index;
        std::uint32_t level{0};
        std::uint64_t created_at{0};
    };

    struct ShardState
    {
        std::vector<IndexedSegment> segments;
        Seed::Snapshot live_snap;
        std::shared_ptr<const GrainIndex> live_grains;
    };

    struct CompactionConfig
    {
        std::size_t level_fanout{4};
        std::size_t max_concurrent_compactions{1};
        std::size_t compaction_trigger_threshold{4};
    };


    class Shard
    {
    public:
        Shard(std::string name,
              std::size_t dim,
              std::size_t queue_cap,
              std::string wal_dir,
              CompactionConfig compaction,
              Logger *logger = nullptr);
        ~Shard();

        void Start();
        void Stop();

        std::future<Result<Lsn>> EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable);
        std::future<Result<bool>> RequestCheckpoint();
        std::future<Result<ShardCheckpointState>> RequestCheckpointState();
        void RequestEmergencyFreeze();

        SearchResponse Search(const SearchRequest &req, const pomai::ai::Budget &budget) const;
        std::size_t ApproxCountUnsafe() const;
        std::shared_ptr<const ShardState> SnapshotState() const;
        std::size_t CompactionBacklog() const;
        std::uint64_t LastCompactionDurationMs() const;
        Lsn DurableLsn() const;
        Lsn WrittenLsn() const;
        std::vector<Vector> SampleVectors(std::size_t max_samples) const;
        void LoadFromCheckpoint(const ShardCheckpointState &state, Lsn checkpoint_lsn);

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
        void RunCompactionLoop();
        void MaybeScheduleCompaction();
        bool CompactLevel(std::uint32_t level);
        std::size_t ComputeCompactionBacklog(const std::shared_ptr<const ShardState> &state) const;
        Seed MergeSnapshots(const std::vector<Seed::Snapshot> &snaps) const;

    private:
        std::string name_;
        std::string wal_dir_;
        Wal wal_;
        Seed seed_;
        BoundedQueue<UpsertTask> ingest_q_;
        IndexBuildPool *build_pool_{nullptr};
        Logger *logger_{nullptr};
        mutable std::mutex writer_mu_;
        std::atomic<std::shared_ptr<const ShardState>> state_{nullptr};
        std::thread owner_;
        std::thread compactor_;
        static constexpr std::size_t kFreezeEveryVectors = 50000;
        std::size_t since_freeze_{0};
        static constexpr std::size_t kMaxSegments = 64;
        static constexpr std::size_t kPublishLiveEveryVectors = 10000;
        std::size_t since_live_publish_{0};
        std::atomic<bool> emergency_freeze_pending_{false};
        std::optional<Lsn> checkpoint_lsn_;
        bool recovered_{false};
        CompactionConfig compaction_;
        std::atomic<bool> compactor_running_{false};
        std::atomic<std::size_t> active_compactions_{0};
        std::atomic<std::uint64_t> last_compaction_ms_{0};
        std::atomic<std::size_t> compaction_backlog_{0};
        std::atomic<std::uint64_t> segment_epoch_{0};

    public:
        void SetIndexBuildPool(IndexBuildPool *pool) { build_pool_ = pool; }
    };
}
