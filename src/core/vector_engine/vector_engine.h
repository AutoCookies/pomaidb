#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pomai/iterator.h"
#include "pomai/metadata.h"
#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/snapshot.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "core/query/lexical_index.h"

namespace pomai::compute::vulkan {
class VulkanComputeContext;
}

namespace pomai::core {

class VectorRuntime;
class SyncReceiver;

// Monolithic, single-threaded vector engine for one membrane.
// One WAL, one MemTable, one VectorRuntime — no partitioning.
class VectorEngine {
public:
    explicit VectorEngine(pomai::DBOptions opt,
                          pomai::MembraneKind kind,
                          pomai::MetricType metric,
                          uint32_t ttl_sec = 0,
                          uint32_t retention_max_count = 0,
                          uint64_t retention_max_bytes = 0,
                          uint64_t sync_lsn = 0);
    ~VectorEngine();

    VectorEngine(const VectorEngine&) = delete;
    VectorEngine& operator=(const VectorEngine&) = delete;

    Status Open();
    Status Close();

    // Ingestion ---------------------------------------------------------------
    Status Put(VectorId id, std::span<const float> vec);
    Status Put(VectorId id,
               std::span<const float> vec,
               const pomai::Metadata& meta);
    Status PutBatch(const std::vector<VectorId>& ids,
                    const std::vector<std::span<const float>>& vectors);
    /** Batch ingestion (vector-of-vectors); single bulk path to runtime. */
    Status PutBatch(const std::vector<VectorId>& ids,
                    const std::vector<std::vector<float>>& vectors);

    // Point lookups -----------------------------------------------------------
    Status Get(VectorId id, std::vector<float>* out);
    Status Get(VectorId id,
               std::vector<float>* out,
               pomai::Metadata* out_meta);
    Status Exists(VectorId id, bool* exists);
    Status Delete(VectorId id);

    // Maintenance -------------------------------------------------------------
    Status Flush();
    Status Freeze();
    Status Compact();

    // Snapshots & iteration ---------------------------------------------------
    Status NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out);
    Status GetSnapshot(std::shared_ptr<pomai::Snapshot>* out);
    Status NewIterator(const std::shared_ptr<pomai::Snapshot>& snap,
                       std::unique_ptr<pomai::SnapshotIterator>* out);

    // Search ------------------------------------------------------------------
    Status Search(std::span<const float> query,
                  std::uint32_t topk,
                  pomai::SearchResult* out);
    Status Search(std::span<const float> query,
                  std::uint32_t topk,
                  const SearchOptions& opts,
                  pomai::SearchResult* out);
    /** @brief Zero-copy search directly into a sink. skips SearchResult allocation/copy. */
    Status Search(std::span<const float> query,
                  std::uint32_t topk,
                  const SearchOptions& opts,
                  pomai::SearchHitSink& sink);
    Status SearchLexical(const std::string& query,
                        std::uint32_t topk,
                        std::vector<LexicalHit>* out);
    Status SearchBatch(std::span<const float> queries,
                       std::uint32_t num_queries,
                       std::uint32_t topk,
                       std::vector<pomai::SearchResult>* out);
    Status SearchBatch(std::span<const float> queries,
                       std::uint32_t num_queries,
                       std::uint32_t topk,
                       const SearchOptions& opts,
                       std::vector<pomai::SearchResult>* out);

    Status PushSync(SyncReceiver* receiver);
    uint64_t GetLastSyncedLSN() const;

    /** Current active memtable bytes used for this membrane (for backpressure). */
    std::size_t MemTableBytesUsed() const noexcept;

    const pomai::DBOptions& options() const { return opt_; }
    pomai::MembraneKind kind() const { return kind_; }
    pomai::MetricType metric() const { return metric_; }

private:
    Status OpenLocked();

    Status EnsureOpen() const;
    Status ValidateVector(std::span<const float> vec) const;

    pomai::DBOptions opt_;
    pomai::MembraneKind kind_;
    pomai::MetricType metric_;
    uint32_t ttl_sec_{0};
    uint32_t retention_max_count_{0};
    uint64_t retention_max_bytes_{0};
    bool opened_{false};
    uint64_t sync_lsn_{0};

    std::unique_ptr<VectorRuntime> runtime_;

    /// Optional Vulkan compute/memory context when `vulkan_enable_memory_bridge` is set.
    std::unique_ptr<pomai::compute::vulkan::VulkanComputeContext> vulkan_ctx_;
};

} // namespace pomai::core

