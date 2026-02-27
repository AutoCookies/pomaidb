#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <variant>
#include <vector>

#include "core/shard/mailbox.h"
#include "core/shard/snapshot.h"
#include "core/shard/shard_stats.h"
#include "pomai/metadata.h"
#include "pomai/search.h"
#include "pomai/iterator.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/options.h"
#include "util/thread_pool.h"

namespace pomai::storage
{
    class Wal;
    class CompactionManager;
}
namespace pomai::table
{
    class MemTable;
    class SegmentReader;
}

// Forward declare IVF (avoid heavy include in header).
namespace pomai::index
{
    class IvfCoarse;
}

namespace pomai::core
{

    struct PutCmd
    {
        VectorId id{};
        pomai::VectorView vec{};
        pomai::Metadata meta{}; // Added
        std::promise<pomai::Status> done;
    };

    struct DelCmd
    {
        VectorId id{};
        std::promise<pomai::Status> done;
    };

    struct BatchPutCmd
    {
        std::vector<pomai::VectorId> ids;
        std::vector<pomai::VectorView> vectors;  // Borrowed views, valid until command completes
        std::promise<pomai::Status> done;
    };

    struct FlushCmd
    {
        std::promise<pomai::Status> done;
    };

    // MUST be complete before being used in std::promise<SearchReply>.
    struct SearchReply
    {
        pomai::Status st;
        std::vector<pomai::SearchHit> hits;
    };

    struct SearchCmd
    {
        std::vector<float> query;
        std::uint32_t topk{0};
        std::promise<SearchReply> done;
    };

    struct StopCmd
    {
        std::promise<void> done;
    };

    struct FreezeCmd
    {
        std::promise<pomai::Status> done;
    };

    struct CompactCmd
    {
        std::promise<pomai::Status> done;
    };

    struct IteratorReply
    {
        pomai::Status st;
        std::unique_ptr<pomai::SnapshotIterator> iterator;
    };

    struct IteratorCmd
    {
        std::promise<IteratorReply> done;
    };

    using Command = std::variant<PutCmd, DelCmd, BatchPutCmd, FlushCmd, SearchCmd, StopCmd, FreezeCmd, CompactCmd, IteratorCmd>;

    class SearchMergePolicy;

    class ShardRuntime
    {
    public:
        ShardRuntime(std::uint32_t shard_id,
                     std::string shard_dir,
                     std::uint32_t dim,
                     pomai::MembraneKind kind,
                     pomai::MetricType metric,
                     std::unique_ptr<storage::Wal> wal,
                     std::unique_ptr<table::MemTable> mem,
                     std::size_t mailbox_cap,
                     const pomai::IndexParams& index_params,
                     pomai::util::ThreadPool* thread_pool = nullptr,
                     pomai::util::ThreadPool* segment_pool = nullptr); // Added
                     
        ~ShardRuntime();

        ShardRuntime(const ShardRuntime &) = delete;
        ShardRuntime &operator=(const ShardRuntime &) = delete;

        pomai::Status Start();
        pomai::Status Enqueue(Command &&cmd);

        pomai::Status Put(pomai::VectorId id, std::span<const float> vec);
        pomai::Status Put(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata& meta); // Overload
        pomai::Status PutBatch(const std::vector<pomai::VectorId>& ids,
                               const std::vector<std::span<const float>>& vectors);
        pomai::Status Get(pomai::VectorId id, std::vector<float> *out);
        pomai::Status Get(pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta); // Added
        pomai::Status Exists(pomai::VectorId id, bool *exists);
        pomai::Status Delete(pomai::VectorId id);

        pomai::Status Flush(); // WAL Flush
        pomai::Status Freeze(); // MemTable -> Segment
        pomai::Status Compact(); // Compact Segments

        pomai::Status NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out); // Create snapshot iterator
        pomai::Status NewIterator(std::shared_ptr<ShardSnapshot> snap, std::unique_ptr<pomai::SnapshotIterator>* out); // Added
        
        std::shared_ptr<ShardSnapshot> GetSnapshot() {
             return current_snapshot_.load(std::memory_order_acquire);
        }

        pomai::Status GetSemanticPointer(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, pomai::SemanticPointer* out);

        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             std::vector<pomai::SearchHit> *out);
        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             const SearchOptions& opts,
                             std::vector<pomai::SearchHit> *out); // Overload

        pomai::Status SearchBatchLocal(std::span<const float> queries,
                                       const std::vector<uint32_t>& query_indices,
                                       std::uint32_t topk,
                                       const SearchOptions& opts,
                                       std::vector<std::vector<pomai::SearchHit>>* out_results);

        // Non-blocking enqueue. Returns ResourceExhausted if full.
        pomai::Status TryEnqueue(Command &&cmd);

        std::size_t GetQueueDepth() const { return mailbox_.Size(); }
        std::uint64_t GetOpsProcessed() const { return ops_processed_.load(std::memory_order_relaxed); }
        std::uint64_t LastQueryCandidatesScanned() const { return last_query_candidates_scanned_.load(std::memory_order_relaxed); }

        // Phase 4: per-shard snapshot of runtime metrics (lock-free, any thread).
        ShardStats GetStats() const noexcept;


    private:
        struct BackgroundJob;

        void RunLoop();

        // Internal helpers
        pomai::Status HandlePut(PutCmd &c);
        pomai::Status HandleBatchPut(BatchPutCmd &c);
        pomai::Status HandleDel(DelCmd &c);
        pomai::Status HandleFlush(FlushCmd &c);
        std::optional<pomai::Status> HandleFreeze(FreezeCmd &c);
        std::optional<pomai::Status> HandleCompact(CompactCmd &c);
        IteratorReply HandleIterator(IteratorCmd &c);
        SearchReply HandleSearch(SearchCmd &c);
        // GetReply HandleGet(GetCmd &c); // Deprecated
        // std::pair<pomai::Status, bool> HandleExists(ExistsCmd &c); // Deprecated

        // Lock-free internal helpers
        pomai::Status GetFromSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta = nullptr);
        std::pair<pomai::Status, bool> ExistsInSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id);

        // Core scoring routine that uses a prebuilt merge_policy
        pomai::Status SearchLocalInternal(std::shared_ptr<table::MemTable> active,
                                          std::shared_ptr<ShardSnapshot> snap, 
                                          std::span<const float> query,
                                          float query_sum,
                                          std::uint32_t topk,
                                          const pomai::SearchOptions& opts,
                                          SearchMergePolicy& merge_policy,
                                          bool use_visibility,
                                          std::vector<pomai::SearchHit>* out,
                                          bool use_pool);

                                          
        // Helper to load segments
        pomai::Status LoadSegments();

        // Snapshot management
        void PublishSnapshot();
        
        // Soft Freeze: Move active memtable to frozen.
        pomai::Status RotateMemTable();

        void PumpBackgroundWork(std::chrono::milliseconds budget);
        void CancelBackgroundJob(const std::string& reason);

        const std::uint32_t shard_id_;
        const std::string shard_dir_;
        const std::uint32_t dim_;
        const pomai::MembraneKind kind_;
        const pomai::MetricType metric_;

        std::unique_ptr<storage::Wal> wal_;
        std::atomic<std::shared_ptr<table::MemTable>> mem_;
        // New: Frozen memtables (awaiting flush to disk)
        std::vector<std::shared_ptr<table::MemTable>> frozen_mem_;
        
        std::vector<std::shared_ptr<table::SegmentReader>> segments_;

        // Snapshot
        std::atomic<std::shared_ptr<ShardSnapshot>> current_snapshot_;
        std::uint64_t next_snapshot_version_ = 1;

        // IVF coarse index for candidate selection (centroid routing).
        std::unique_ptr<pomai::index::IvfCoarse> ivf_;

        BoundedMpscQueue<Command> mailbox_;
        std::atomic<std::uint64_t> ops_processed_{0};
        std::atomic<std::uint64_t> last_query_candidates_scanned_{0};

        std::atomic<bool> started_{false};

        pomai::util::ThreadPool* thread_pool_{nullptr};
        pomai::util::ThreadPool* segment_pool_{nullptr}; // Added
        pomai::IndexParams index_params_;

        std::unique_ptr<storage::CompactionManager> compaction_manager_;
        std::unique_ptr<BackgroundJob> background_job_;
        std::uint64_t wal_epoch_{0};

        std::jthread worker_;
    };

} // namespace pomai::core
