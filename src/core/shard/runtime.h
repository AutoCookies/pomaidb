#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <thread>
#include <variant>
#include <vector>

#include "core/shard/mailbox.h"
#include "core/shard/seen_tracker.h"
#include "core/shard/snapshot.h"
#include "pomai/metadata.h"
#include "pomai/search.h" // Restored
#include "pomai/iterator.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/options.h" 
#include "util/thread_pool.h" 

namespace pomai::storage
{
    class Wal;
}
namespace pomai::table
{
    class MemTable;
    class SegmentReader;
}

namespace pomai::core
{

    struct PutCmd
    {
        VectorId id{};
        const float *vec{};
        std::uint32_t dim{};
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
        std::vector<std::vector<float>> vectors;  // Owned copies
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

    class ShardRuntime
    {
    public:
        ShardRuntime(std::uint32_t shard_id,
                     std::string shard_dir,
                     std::uint32_t dim,
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

        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             std::vector<pomai::SearchHit> *out);
        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             const SearchOptions& opts,
                             std::vector<pomai::SearchHit> *out); // Overload

        // Non-blocking enqueue. Returns ResourceExhausted if full.
        pomai::Status TryEnqueue(Command &&cmd);

        std::size_t GetQueueDepth() const { return mailbox_.Size(); }
        std::uint64_t GetOpsProcessed() const { return ops_processed_.load(std::memory_order_relaxed); }

    private:
        void RunLoop();

        // Internal helpers
        pomai::Status HandlePut(PutCmd &c);
        pomai::Status HandleBatchPut(BatchPutCmd &c);
        pomai::Status HandleDel(DelCmd &c);
        pomai::Status HandleFlush(FlushCmd &c);
        pomai::Status HandleFreeze(FreezeCmd &c);
        pomai::Status HandleCompact(CompactCmd &c);
        IteratorReply HandleIterator(IteratorCmd &c);
        SearchReply HandleSearch(SearchCmd &c);
        // GetReply HandleGet(GetCmd &c); // Deprecated
        // std::pair<pomai::Status, bool> HandleExists(ExistsCmd &c); // Deprecated

        // Lock-free internal helpers
        pomai::Status GetFromSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta = nullptr);
        std::pair<pomai::Status, bool> ExistsInSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id);

        pomai::Status SearchLocalInternal(std::shared_ptr<table::MemTable> active,
                                          std::shared_ptr<ShardSnapshot> snap, 
                                          std::span<const float> query,
                                          std::uint32_t topk,
                                          const SearchOptions& opts,
                                          std::vector<pomai::SearchHit> *out);

                                          
        // Helper to load segments
        pomai::Status LoadSegments();

        // Snapshot management
        void PublishSnapshot();
        
        // Soft Freeze: Move active memtable to frozen.
        pomai::Status RotateMemTable();

        const std::uint32_t shard_id_;
        const std::string shard_dir_;
        const std::uint32_t dim_;

        std::unique_ptr<storage::Wal> wal_;
        std::atomic<std::shared_ptr<table::MemTable>> mem_;
        // New: Frozen memtables (awaiting flush to disk)
        std::vector<std::shared_ptr<table::MemTable>> frozen_mem_;
        
        std::vector<std::shared_ptr<table::SegmentReader>> segments_;

        // Snapshot
        std::atomic<std::shared_ptr<ShardSnapshot>> current_snapshot_;
        std::uint64_t next_snapshot_version_ = 1;

        BoundedMpscQueue<Command> mailbox_;
        std::atomic<std::uint64_t> ops_processed_{0};

        std::jthread worker_;
        std::atomic<bool> started_{false};

        pomai::util::ThreadPool* thread_pool_{nullptr};
        pomai::util::ThreadPool* segment_pool_{nullptr}; // Added
        pomai::IndexParams index_params_;
    };

} // namespace pomai::core
