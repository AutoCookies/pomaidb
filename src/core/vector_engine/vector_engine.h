#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "pomai/options.h"
#include "pomai/metadata.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/iterator.h"
#include "pomai/snapshot.h"
#include "util/thread_pool.h"
#include "core/routing/routing_table.h"

namespace pomai::core
{

    class Shard;

    class VectorEngine
    {
    public:
        explicit VectorEngine(pomai::DBOptions opt, pomai::MembraneKind kind);
        ~VectorEngine();

        VectorEngine(const VectorEngine &) = delete;
        VectorEngine &operator=(const VectorEngine &) = delete;

        Status Open();
        Status Close();

        Status Put(VectorId id, std::span<const float> vec);
        Status Put(VectorId id, std::span<const float> vec, const pomai::Metadata& meta); // Overload
        Status PutBatch(const std::vector<VectorId>& ids,
                        const std::vector<std::span<const float>>& vectors);
        Status Get(VectorId id, std::vector<float> *out);
        Status Get(VectorId id, std::vector<float> *out, pomai::Metadata* out_meta); // Added
        Status Exists(VectorId id, bool *exists);
        Status Delete(VectorId id);
        Status Flush();
        Status Freeze();
        Status Compact();
        Status NewIterator(std::unique_ptr<pomai::SnapshotIterator> *out);
        Status GetSnapshot(std::shared_ptr<pomai::Snapshot>* out);
        Status NewIterator(const std::shared_ptr<pomai::Snapshot>& snap, std::unique_ptr<pomai::SnapshotIterator> *out);

        Status Search(std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status Search(std::span<const float> query, std::uint32_t topk, const SearchOptions& opts,
                      pomai::SearchResult *out);

        const pomai::DBOptions &options() const { return opt_; }
        pomai::MembraneKind kind() const { return kind_; }

    private:
        Status OpenLocked();
        static std::uint32_t ShardOf(VectorId id, std::uint32_t shard_count);
        std::uint32_t RouteShardForVector(VectorId id, std::span<const float> vec);
        void MaybeWarmupAndInitRouting(std::span<const float> vec);
        void MaybePersistRoutingAsync();
        std::vector<std::uint32_t> BuildProbeShards(std::span<const float> query, const SearchOptions& opts);

        pomai::DBOptions opt_;
        pomai::MembraneKind kind_;
        bool opened_ = false;

        std::vector<std::unique_ptr<Shard>> shards_;
        std::unique_ptr<util::ThreadPool> search_pool_;
        std::unique_ptr<util::ThreadPool> segment_pool_; // Added

        std::atomic<routing::RoutingMode> routing_mode_{routing::RoutingMode::kDisabled};
        std::shared_ptr<routing::RoutingTable> routing_mutable_;
        routing::RoutingTablePtr routing_current_;
        routing::RoutingTablePtr routing_prev_;
        mutable std::mutex routing_mu_;
        std::vector<float> warmup_reservoir_;
        std::uint32_t warmup_count_ = 0;
        std::uint32_t warmup_target_ = 0;
        std::uint64_t puts_since_persist_ = 0;
        bool routing_persist_inflight_ = false;

        std::atomic<std::uint32_t> routed_shards_last_query_count_{0};
        std::atomic<std::uint32_t> routed_probe_centroids_last_query_{0};
    };

} // namespace pomai::core
