#pragma once

#include <cstdint>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "core/shard/snapshot.h"
#include "core/shard/shard_stats.h"  // RuntimeStats
#include "pomai/metadata.h"
#include "pomai/search.h"
#include "pomai/iterator.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/options.h"
#include "core/concurrency/concurrency_macros.h"
#include "core/concurrency/executor.h"
#include "core/memory/local_pool.h"

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

    // Single-threaded command payloads (no std::promise; handlers return directly).
    struct PutCmd
    {
        VectorId id{};
        pomai::VectorView vec{};
        pomai::Metadata meta{};
    };

    struct DelCmd
    {
        VectorId id{};
    };

    struct BatchPutCmd
    {
        std::vector<pomai::VectorId> ids;
        std::vector<pomai::VectorView> vectors;
    };

    struct FlushCmd {};

    struct SearchReply
    {
        pomai::Status st;
        std::vector<pomai::SearchHit> hits;
    };

    struct SearchCmd
    {
        std::vector<float> query;
        std::uint32_t topk{0};
    };

    struct FreezeCmd {};
    struct CompactCmd {};

    struct IteratorReply
    {
        pomai::Status st;
        std::unique_ptr<pomai::SnapshotIterator> iterator;
    };

    struct IteratorCmd {};

    using Command = std::variant<PutCmd, DelCmd, BatchPutCmd, FlushCmd, SearchCmd, FreezeCmd, CompactCmd, IteratorCmd>;

    class SearchMergePolicy;

    class POMAI_CACHE_ALIGNED VectorRuntime
    {
    public:
        VectorRuntime(std::uint32_t runtime_id,
                     std::string data_dir,
                     std::uint32_t dim,
                     pomai::MembraneKind kind,
                     pomai::MetricType metric,
                     std::unique_ptr<storage::Wal> wal,
                     std::unique_ptr<table::MemTable> mem,
                     const pomai::IndexParams& index_params);
                     
        ~VectorRuntime();

        VectorRuntime(const VectorRuntime &) = delete;
        VectorRuntime &operator=(const VectorRuntime &) = delete;

        pomai::Status Start();

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
        pomai::Status NewIterator(std::shared_ptr<VectorSnapshot> snap, std::unique_ptr<pomai::SnapshotIterator>* out);
        
        std::shared_ptr<VectorSnapshot> GetSnapshot() {
             return current_snapshot_;
        }

        pomai::Status GetSemanticPointer(std::shared_ptr<VectorSnapshot> snap, pomai::VectorId id, pomai::SemanticPointer* out);

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

        std::uint64_t GetOpsProcessed() const { return ops_processed_; }
        std::uint64_t LastQueryCandidatesScanned() const { return last_query_candidates_scanned_; }

        RuntimeStats GetStats() const noexcept;

        /** Active memtable bytes used (for backpressure). Single-threaded: no lock. */
        std::size_t MemTableBytesUsed() const noexcept;


    private:
        struct BackgroundJob;

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
        pomai::Status GetFromSnapshot(std::shared_ptr<VectorSnapshot> snap, pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta = nullptr);
        std::pair<pomai::Status, bool> ExistsInSnapshot(std::shared_ptr<VectorSnapshot> snap, pomai::VectorId id);

        // Core scoring routine that uses a prebuilt merge_policy
        pomai::Status SearchLocalInternal(std::shared_ptr<table::MemTable> active,
                                          std::shared_ptr<VectorSnapshot> snap, 
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

        const std::uint32_t runtime_id_;
        const std::string data_dir_;
        const std::uint32_t dim_;
        const pomai::MembraneKind kind_;
        const pomai::MetricType metric_;

        std::unique_ptr<storage::Wal> wal_;
        std::shared_ptr<table::MemTable> mem_;
        std::vector<std::shared_ptr<table::MemTable>> frozen_mem_;

        std::vector<std::shared_ptr<table::SegmentReader>> segments_;

        std::shared_ptr<VectorSnapshot> current_snapshot_;
        std::uint64_t next_snapshot_version_ = 1;

        // IVF coarse index for candidate selection (centroid routing).
        std::unique_ptr<pomai::index::IvfCoarse> ivf_;

        // Elite Concurrency & Memory Base
        concurrency::Executor executor_;
        memory::RuntimeMemoryManager mem_manager_;

        POMAI_CACHE_ALIGNED std::uint64_t ops_processed_{0};
        POMAI_CACHE_ALIGNED std::uint64_t last_query_candidates_scanned_{0};

        bool started_{false};

        pomai::IndexParams index_params_;

        std::unique_ptr<storage::CompactionManager> compaction_manager_;
        std::unique_ptr<BackgroundJob> background_job_;
        std::optional<pomai::Status> last_background_result_;  // Set when background job completes (single-threaded)
        std::uint64_t wal_epoch_{0};

        // Reusable scratch buffers for search hot path (single-threaded; avoids per-query allocations).
        mutable std::vector<pomai::SearchHit> search_candidates_scratch_;
        mutable std::vector<std::vector<pomai::SearchHit>> search_segment_hits_scratch_;
        mutable std::vector<float> search_query_sums_scratch_;
    };

} // namespace pomai::core
