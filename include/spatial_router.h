#pragma once
// Spatial router (centroid-based) for Pomai:
// - Maintain a set of centroids
// - Route inserts to a single shard by nearest centroid
// - Route queries to top-P nearest shard ids (multi-probe)
//
// This header exposes a small, embeddable Router class and a
// simple offline k-means helper to compute centroids from a sample.
//
// Integration notes:
// - Your codebase already defines Vector, kernels::L2Sqr, etc.
//   This Router uses those types and functions. Adjust includes if names differ.
//
// - Typical usage:
//   SpatialRouter router(K); // K = number of centroids (>= num_shards * s)
//   router.ReplaceCentroids(centroids);
//   size_t target_shard = router.PickShardForInsert(vec);               // one shard
//   auto shards = router.CandidateShardsForQuery(query_vec, probe_P);  // top-P shards
//
// - Supports atomic centroid replacement (ReplaceCentroids).
//
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <cstddef>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>

#include "types.h"       // for Vector, Id, etc.
#include "cpu_kernels.h" // for kernels::L2Sqr

namespace pomai
{

    class SpatialRouter
    {
    public:
        // Construct empty router. centroids_ size is zero until ReplaceCentroids is called.
        SpatialRouter() = default;

        // Return nearest shard index (closest centroid) for an insert vector.
        // Assumes centroids_.size() >= num_shards (centroids are mapped to shards externally).
        // This is a low-latency hot path (uses L2Sqr).
        std::size_t PickShardForInsert(const Vector &vec) const;

        // Return top-P unique shard indices (indices into centroids_). If P > #centroids,
        // returns all centroid indices sorted by distance ascending.
        std::vector<std::size_t> CandidateShardsForQuery(const Vector &q, std::size_t P) const;

        // Replace centroids atomically.
        void ReplaceCentroids(std::vector<Vector> new_centroids);

        // Read-only access (snapshot copy) to centroids for external tooling or debugging.
        std::vector<Vector> SnapshotCentroids() const;

        // ----- Helper: build centroids using simple k-means (Lloyd)
        // This is a convenience offline method. It does not touch internal centroids_;
        // it returns computed centroids so caller can decide when to atomically replace.
        // - data: sample vectors (flattened), must be non-empty and each vector Dim()
        // - k: number of centroids to produce
        // - iterations: number of Lloyd iterations (default 10)
        // Throws std::runtime_error on invalid input.
        static std::vector<Vector> BuildKMeans(const std::vector<Vector> &data, std::size_t k, int iterations = 10);

    private:
        mutable std::shared_mutex mu_;
        std::vector<Vector> centroids_;
    };

} // namespace pomai