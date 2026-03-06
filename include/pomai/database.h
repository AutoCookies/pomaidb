// PomaiDB Embedded: Single-instance, non-thread-safe by design for maximum raw throughput.
// Shared-nothing monolithic pattern: one StorageEngine, one Arena, one append-only WAL.
// Data flow is strictly sequential (Input -> Database -> Single Arena -> Single Append-Only File)
// for optimal performance on MicroSD and low-end embedded hardware.

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

namespace pomai {

// Forward declaration: single-instance storage (one Arena, one WAL, one index).
class StorageEngine;

// Embedded database options: no sharding, no routing, no thread count.
struct EmbeddedOptions {
    std::string path;
    std::uint32_t dim = 512;
    MetricType metric = MetricType::kL2;
    FsyncPolicy fsync = FsyncPolicy::kNever;
    /** Index/quantization params. Use IndexParams::ForEdge() for low-memory edge devices. */
    IndexParams index_params;

    /** Memtable backpressure: max size in MiB before rejecting Put (0 = from env or default). */
    std::uint32_t max_memtable_mb = 0;
    /** Pressure threshold as percent of max (0 = use 80). When exceeded, Put returns ResourceExhausted or auto-freeze runs. */
    std::uint8_t pressure_threshold_percent = 0;
    /** If true, when over threshold call Freeze() internally before allowing Put. Default true for edge to prevent OOM. */
    bool auto_freeze_on_pressure = true;
    /** Memtable flush threshold in MiB; when exceeded, auto-freeze runs. 0 = use pressure percent of max_memtable_mb. */
    std::uint32_t memtable_flush_threshold_mb = 64u;
};

/**
 * Database: thin wrapper around one StorageEngine and one vector index.
 * Single-threaded only; caller serializes access or runs on one thread.
 *
 * Visibility and staleness: Search and Get use the latest published snapshot
 * (updated after Freeze/Compact). During heavy ingestion, reads may be at most
 * one freeze cycle behind the most recent Put. Use GetSnapshot + NewIterator
 * for a fixed point-in-time view. Deletes are tombstones; "newest wins" per id.
 */
class Database {
public:
    Database();
    ~Database();

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    /** Open at the given path with embedded options. */
    Status Open(const EmbeddedOptions& options);

    /** Close and release storage. */
    Status Close();

    /** Flush WAL to storage (sequential append). */
    Status Flush();

    /** Freeze: move active memtable to segment (single-instance). */
    Status Freeze();

    /**
     * If memtable is over pressure threshold, call Freeze() once. Safe to call periodically.
     * Single-threaded backpressure: no lock; app can call from event loop to avoid ResourceExhausted.
     */
    Status TryFreezeIfPressured();

    /** Current active memtable bytes used (for monitoring). */
    std::size_t GetMemTableBytesUsed() const;

    /** Append one vector; direct path to storage. */
    Status AddVector(VectorId id, std::span<const float> vec);

    /** Append one vector with metadata. */
    Status AddVector(VectorId id, std::span<const float> vec,
                    const Metadata& meta);

    /** Batch append: single bulk path to storage (fewer syscalls, better throughput). */
    Status AddVectorBatch(const std::vector<VectorId>& ids,
                         const std::vector<std::span<const float>>& vectors);

    /**
     * PutBatch: batch ingestion API (vector-of-vectors overload).
     * Prefer this over repeated AddVector for bulk loads; uses one bulk append path.
     * Backward compatible: existing AddVector/AddVectorBatch unchanged.
     */
    Status PutBatch(const std::vector<VectorId>& ids,
                    const std::vector<std::vector<float>>& vectors);

    /**
     * PutBatch: zero-copy batch API for embedded/edge callers.
     * - ids: N vector IDs
     * - vectors: flattened N * dimension buffer (id0[0..dim), id1[0..dim), ...)
     * - dimension: per-vector dimension (must match EmbeddedOptions.dim)
     *
     * Visibility is RAM-first: vectors are appended into the in-memory arena and
     * indexed immediately; disk flush happens later via StorageEngine::Flush().
     */
    Status PutBatch(std::span<const VectorId> ids,
                    std::span<const float> vectors,
                    std::size_t dimension);

    /** Get vector by id. */
    Status Get(VectorId id, std::vector<float>* out);
    Status Get(VectorId id, std::vector<float>* out,
              Metadata* out_meta);

    /** Check existence. */
    Status Exists(VectorId id, bool* exists);

    /** Delete by id (tombstone). */
    Status Delete(VectorId id);

    /** Search: direct path to index_->search(). */
    Status Search(std::span<const float> query, std::uint32_t topk,
                  SearchResult* out);

    Status Search(std::span<const float> query, std::uint32_t topk,
                  const SearchOptions& opts,
                  SearchResult* out);

    /** Batch search (multiple queries). */
    Status SearchBatch(std::span<const float> queries, std::uint32_t num_queries,
                       std::uint32_t topk, const SearchOptions& opts,
                       std::vector<SearchResult>* out);

    /** Snapshot (point-in-time view). */
    Status GetSnapshot(std::shared_ptr<Snapshot>* out);
    /** Iterator over snapshot; snap must be from GetSnapshot(). */
    Status NewIterator(const std::shared_ptr<Snapshot>& snap,
                      std::unique_ptr<SnapshotIterator>* out);

    [[nodiscard]] bool IsOpen() const { return opened_; }

private:
    std::unique_ptr<StorageEngine> storage_engine_;
    bool opened_ = false;

    /** When memtable exceeds threshold: return ResourceExhausted or Freeze() if auto. Single-threaded. */
    Status MaybeApplyBackpressure();

    // Memtable backpressure (set in Open from options or env). Single-threaded: no lock.
    std::size_t max_memtable_bytes_ = 0;
    std::size_t pressure_threshold_bytes_ = 0;
    bool auto_freeze_on_pressure_ = false;
};

} // namespace pomai
