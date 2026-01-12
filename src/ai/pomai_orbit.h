/*
 * src/ai/pomai_orbit.h
 *
 * POMAI ORBIT: The Unified Proprietary Vector Engine.
 *
 * Architecture:
 * 1. Routing Layer (RAM): A custom Navigable Small World (NSW) graph of Centroids.
 * 2. Storage Layer (Arena): "Gravity Buckets" storing contiguous SoA blocks.
 *
 * Philosophy:
 * - Robustness: No complex pointer chasing. Append-only buckets.
 * - Performance: Sequential access patterns optimized for SIMD and SSD prefetching.
 * - Simplicity: Unified mode. No distinction between Index and Storage.
 */

#pragma once

#include <vector>
#include <atomic>
#include <shared_mutex>
#include <memory>
#include <cmath>

#include "src/memory/arena.h"
#include "src/ai/fingerprint.h"
#include "src/ai/pq.h"
#include "src/core/config.h"

namespace pomai::ai::orbit
{

    // --- Core Constants ---
    constexpr uint32_t MAX_BUCKET_CAPACITY = 4096; // Fits nicely in hugepages / sequential reads
    constexpr uint32_t ORBIT_SEED_DIM = 64;        // Dimension of centroids (can be compressed)

    // --- On-Disk Structures (Must be POD / Packed) ---

    // 1. The Bucket Header (Lives at the start of a blob in Arena)
    struct BucketHeader
    {
        uint32_t centroid_id;        // ID of the centroid owning this bucket
        std::atomic<uint32_t> count; // Number of vectors currently in this bucket
        uint64_t next_bucket_offset; // Linked list if bucket overflows (0 = end)

        // SoA Offsets (Relative to struct start)
        uint32_t off_fingerprints;
        uint32_t off_pq_codes;
        uint32_t off_vectors; // Offset to raw float data (optional/cold)
        uint32_t off_ids;     // Offset to external IDs (labels)
    };

    // 2. The Centroid Node (Lives in RAM for Routing)
    struct OrbitNode
    {
        std::vector<float> vector;       // The centroid vector
        std::vector<uint32_t> neighbors; // NSW graph connections
        uint64_t bucket_offset;          // Pointer to the first bucket in Arena
        std::shared_mutex mu;            // Protects bucket expansion
    };

    // --- The Engine ---

    class PomaiOrbit
    {
    public:
        // Config
        struct Config
        {
            size_t dim;
            size_t num_centroids = 1024; // Default routing complexity
            size_t m_neighbors = 16;     // Graph connectivity
            bool use_pq = true;
            bool use_fingerprint = true;
        };

        PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena);
        ~PomaiOrbit();

        // --- Core API ---

        // Train the routing layer (Centroids + PQ).
        // Must be called once before inserting massive data.
        bool train(const float *data, size_t n);

        // Insert a vector.
        // 1. Search Orbit Graph -> Find nearest Centroid.
        // 2. Append to that Centroid's Bucket.
        bool insert(const float *vec, uint64_t label);

        // Search.
        // 1. Search Orbit Graph -> Find 'nprobe' nearest Centroids.
        // 2. Scan those Buckets (Fingerprint -> PQ -> Exact).
        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k, size_t nprobe = 3);

        // Durability
        bool save_routing(const std::string &path);
        bool load_routing(const std::string &path);

    private:
        Config cfg_;
        pomai::memory::PomaiArena *arena_;

        // AI Components
        std::unique_ptr<ProductQuantizer> pq_;
        std::unique_ptr<FingerprintEncoder> fp_;

        // Routing Graph (In-Memory)
        std::vector<OrbitNode> centroids_;

        // Helper: Find closest centroid to a vector (Greedy Graph Search)
        uint32_t find_nearest_centroid(const float *vec);

        // Helper: Find Top-N closest centroids (Beam Search)
        std::vector<uint32_t> find_routing_centroids(const float *vec, size_t n);

        // Helper: Allocate a new bucket in Arena and link it
        uint64_t alloc_new_bucket(uint32_t centroid_id);
    };

} // namespace pomai::ai::orbit