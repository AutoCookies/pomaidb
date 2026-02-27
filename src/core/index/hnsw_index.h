// hnsw_index.h — PomaiDB wrapper around faiss::IndexHNSWFlat.
//
// Phase 3: Wraps FAISS HNSW for use as a drop-in sidecar index per Shard.
// Activated via DBOptions::index_type = IndexType::kHNSW.
// Backward-compatible: when not set, IvfFlatIndex is used as before.
//
// Phase 4: Save/Load via faiss::index_io for .idx sidecar persistence.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pomai/status.h"
#include "pomai/types.h"

// Forward-declare FAISS types to avoid pulling FAISS headers into every TU.
namespace faiss {
struct Index;
struct IndexHNSWFlat;
} // namespace faiss

namespace pomai::index {

/// Options for the HNSW index.
struct HnswOptions {
    int M            = 32;    // Number of HNSW neighbors per node. Higher = better recall, more RAM.
    int ef_construction = 200; // Candidates during build. Higher = better graph quality.
    int ef_search    = 64;    // Candidates during query. Tunable at query time.
};

/// Per-Shard HNSW index. Thread-safe for concurrent reads; single-writer Add().
class HnswIndex {
public:
    /// Create an empty HNSW index (not yet trained/populated).
    HnswIndex(uint32_t dim, HnswOptions opts = {});
    ~HnswIndex();

    // Non-copyable
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    // ── Build Phase ───────────────────────────────────────────────────────────
    /// Add one vector with the given PomaiDB VectorId.
    /// NOTE: FAISS HNSW stores vectors contiguously; internal FAISS ids are
    ///       sequential. We keep a mapping faiss_id → VectorId.
    pomai::Status Add(VectorId id, std::span<const float> vec);

    /// Add a batch of vectors.
    pomai::Status AddBatch(const VectorId* ids,
                           const float*    vecs,
                           std::size_t     n);

    // ── Query Phase ───────────────────────────────────────────────────────────
    /// Approximate nearest neighbor search.
    /// @param query     Query vector (dim() floats).
    /// @param topk      Number of results requested.
    /// @param ef_search Override for ef_search (0 = use default from HnswOptions).
    /// @param out_ids   Output VectorIds (size topk).
    /// @param out_dists Output distances (size topk).
    pomai::Status Search(std::span<const float> query,
                         uint32_t               topk,
                         int                    ef_search,
                         std::vector<VectorId>* out_ids,
                         std::vector<float>*    out_dists) const;

    // ── Metadata ──────────────────────────────────────────────────────────────
    uint32_t    dim()   const { return dim_; }
    std::size_t count() const { return id_map_.size(); }
    HnswOptions opts()  const { return opts_; }

    // ── Persistence (Phase 4) ─────────────────────────────────────────────────
    /// Write FAISS index to path (e.g. "seg_00001.hnsw").
    pomai::Status Save(const std::string& path) const;

    /// Load FAISS index from path. Also loads the VectorId map.
    static pomai::Status Load(const std::string& path,
                              std::unique_ptr<HnswIndex>* out);

private:
    uint32_t    dim_;
    HnswOptions opts_;

    // Owned FAISS index (heap-allocated to avoid including faiss headers here)
    std::unique_ptr<faiss::IndexHNSWFlat> index_;

    // faiss internal idx → PomaiDB VectorId mapping (sequential, index = faiss id)
    std::vector<VectorId> id_map_;
};

} // namespace pomai::index
