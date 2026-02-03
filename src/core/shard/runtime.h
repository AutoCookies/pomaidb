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
#include "core/shard/snapshot.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::storage
{
    class Wal;
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
        const float *vec{};
        std::uint32_t dim{};
        std::promise<pomai::Status> done;
    };

    struct DelCmd
    {
        VectorId id{};
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

    struct GetReply
    {
        pomai::Status st;
        std::vector<float> vec;
    };

    struct GetCmd
    {
        VectorId id{};
        std::promise<GetReply> done;
    };

    struct ExistsCmd
    {
        VectorId id{};
        std::promise<std::pair<pomai::Status, bool>> done;
    };

    struct FreezeCmd
    {
        std::promise<pomai::Status> done;
    };

    struct CompactCmd
    {
        std::promise<pomai::Status> done;
    };

    using Command = std::variant<PutCmd, DelCmd, FlushCmd, SearchCmd, StopCmd, GetCmd, ExistsCmd, FreezeCmd, CompactCmd>;

    class ShardRuntime
    {
    public:
        ShardRuntime(std::uint32_t shard_id,
                     std::string shard_dir,
                     std::uint32_t dim,
                     std::unique_ptr<storage::Wal> wal,
                     std::unique_ptr<table::MemTable> mem,
                     std::size_t mailbox_cap);
                     
        ~ShardRuntime();

        ShardRuntime(const ShardRuntime &) = delete;
        ShardRuntime &operator=(const ShardRuntime &) = delete;

        pomai::Status Start();
        pomai::Status Enqueue(Command &&cmd);

        pomai::Status Put(pomai::VectorId id, std::span<const float> vec);
        pomai::Status Get(pomai::VectorId id, std::vector<float> *out);
        pomai::Status Exists(pomai::VectorId id, bool *exists);
        pomai::Status Delete(pomai::VectorId id);

        pomai::Status Flush(); // WAL Flush
        pomai::Status Freeze(); // MemTable -> Segment
        pomai::Status Compact(); // Compact Segments

        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             std::vector<pomai::SearchHit> *out);

        // Non-blocking enqueue. Returns ResourceExhausted if full.
        pomai::Status TryEnqueue(Command &&cmd);

        std::size_t GetQueueDepth() const { return mailbox_.Size(); }
        std::uint64_t GetOpsProcessed() const { return ops_processed_.load(std::memory_order_relaxed); }

    private:
        void RunLoop();

        // Internal helpers
        pomai::Status HandlePut(PutCmd &c);
        pomai::Status HandleDel(DelCmd &c);
        pomai::Status HandleFlush(FlushCmd &c);
        pomai::Status HandleFreeze(FreezeCmd &c);
        pomai::Status HandleCompact(CompactCmd &c);
        SearchReply HandleSearch(SearchCmd &c);
        // GetReply HandleGet(GetCmd &c); // Deprecated
        // std::pair<pomai::Status, bool> HandleExists(ExistsCmd &c); // Deprecated

        // Lock-free internal helpers
        pomai::Status GetFromSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, std::vector<float> *out);
        std::pair<pomai::Status, bool> ExistsInSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id);

        pomai::Status SearchLocalInternal(std::shared_ptr<ShardSnapshot> snap, 
                                          std::span<const float> query,
                                          std::uint32_t topk,
                                          std::vector<pomai::SearchHit> *out);
                                          
        // Helper to load segments
        pomai::Status LoadSegments();

        // Snapshot management
        void PublishSnapshot();
        std::shared_ptr<ShardSnapshot> GetSnapshot() {
             return current_snapshot_.load(std::memory_order_acquire);
        }
        
        // Soft Freeze: Move active memtable to frozen.
        pomai::Status RotateMemTable();

        const std::uint32_t shard_id_;
        const std::string shard_dir_;
        const std::uint32_t dim_;

        std::unique_ptr<storage::Wal> wal_;
        std::unique_ptr<table::MemTable> mem_;
        // New: Frozen memtables (awaiting flush to disk)
        std::vector<std::shared_ptr<table::MemTable>> frozen_mem_;
        
        std::vector<std::shared_ptr<table::SegmentReader>> segments_;

        // Snapshot
        std::atomic<std::shared_ptr<ShardSnapshot>> current_snapshot_;
        std::uint64_t next_snapshot_version_ = 1;

        // IVF coarse index for candidate selection (centroid routing).
        std::unique_ptr<pomai::index::IvfCoarse> ivf_;
        std::vector<pomai::VectorId> candidates_scratch_;

        BoundedMpscQueue<Command> mailbox_;
        std::atomic<std::uint64_t> ops_processed_{0};

        std::jthread worker_;
        std::atomic<bool> started_{false};
    };

} // namespace pomai::core
