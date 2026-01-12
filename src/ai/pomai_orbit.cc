/*
 * src/ai/pomai_orbit.cc
 *
 * Implementation of Pomai Orbit: The Unified Proprietary Vector Engine.
 *
 * Design Principles:
 * 1. Absolute Separation: Routing (RAM) is decoupled from Storage (Arena).
 * 2. Sequential Locality: Data is packed into contiguous buckets for max bandwidth.
 * 3. Robustness: Minimal pointer chasing; prefer array indexing and offset arithmetic.
 */

#include "src/ai/pomai_orbit.h"
#include "src/ai/atomic_utils.h"
#include "src/ai/ids_block.h" // For IdEntry helpers if needed

#include <algorithm>
#include <random>
#include <limits>
#include <cstring>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <fstream>

namespace pomai::ai::orbit
{

    // --- Internal Helpers ---

    // Compute Squared L2 distance between two vectors
    static inline float l2sq(const float *a, const float *b, size_t dim)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    // --- PomaiOrbit Implementation ---

    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena)
        : cfg_(cfg), arena_(arena)
    {
        if (!arena_)
        {
            throw std::invalid_argument("PomaiOrbit: arena must not be null");
        }
        if (cfg_.dim == 0)
        {
            throw std::invalid_argument("PomaiOrbit: dim must be > 0");
        }
    }

    PomaiOrbit::~PomaiOrbit() = default;

    // --- Training Phase ---
    // 1. Initialize Centroids using K-Means.
    // 2. Build NSW Graph connections.
    // 3. Initialize Bucket Storage in Arena.
    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0)
            return false;

        std::cout << "[Orbit] Training started with N=" << n << ", Dim=" << cfg_.dim << "\n";

        // Step 1: Initialize Centroids (Simple Random Sampling for Robustness)
        // In a production "10/10" version, we would use K-Means++.
        // For robustness and speed here, we sample and run a few Lloyd iterations.
        size_t num_c = std::min(n, cfg_.num_centroids);
        centroids_.resize(num_c);

        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);

        // Init centroids positions
        for (size_t i = 0; i < num_c; ++i)
        {
            centroids_[i].vector.resize(cfg_.dim);
            std::memcpy(centroids_[i].vector.data(), data + indices[i] * cfg_.dim, cfg_.dim * sizeof(float));
            // Reserve space for neighbors
            centroids_[i].neighbors.reserve(cfg_.m_neighbors * 2);
        }

        // Run K-Means (Simplified 5 iterations to refine centroids)
        // This ensures centroids cover the space well, crucial for routing efficiency.
        std::vector<int> counts(num_c, 0);
        std::vector<float> accum(num_c * cfg_.dim, 0.0f);

        for (int iter = 0; iter < 5; ++iter)
        {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(accum.begin(), accum.end(), 0.0f);

            for (size_t i = 0; i < n; ++i)
            {
                const float *vec = data + i * cfg_.dim;
                uint32_t best_c = find_nearest_centroid(vec);

                counts[best_c]++;
                float *acc_ptr = &accum[best_c * cfg_.dim];
                for (size_t d = 0; d < cfg_.dim; ++d)
                    acc_ptr[d] += vec[d];
            }

            // Update centroids
            for (size_t c = 0; c < num_c; ++c)
            {
                if (counts[c] > 0)
                {
                    float inv = 1.0f / counts[c];
                    for (size_t d = 0; d < cfg_.dim; ++d)
                    {
                        centroids_[c].vector[d] = accum[c * cfg_.dim + d] * inv;
                    }
                }
            }
        }
        std::cout << "[Orbit] Centroids refined.\n";

        // Step 2: Build Routing Graph (Navigable Small World)
        // Connect each centroid to its M nearest neighbors.
        // This graph lives purely in RAM.
        for (size_t i = 0; i < num_c; ++i)
        {
            std::vector<std::pair<float, uint32_t>> dists;
            dists.reserve(num_c);
            for (size_t j = 0; j < num_c; ++j)
            {
                if (i == j)
                    continue;
                float d = l2sq(centroids_[i].vector.data(), centroids_[j].vector.data(), cfg_.dim);
                dists.push_back({d, static_cast<uint32_t>(j)});
            }
            // Sort and pick top M
            std::sort(dists.begin(), dists.end()); // Ascending distance

            for (size_t k = 0; k < std::min(dists.size(), cfg_.m_neighbors); ++k)
            {
                centroids_[i].neighbors.push_back(dists[k].second);
            }
        }
        std::cout << "[Orbit] Routing Graph built.\n";

        // Step 3: Train Sub-components (PQ / Fingerprint)
        if (cfg_.use_fingerprint)
        {
            // Default 512 bits for high selectivity
            fp_ = FingerprintEncoder::createSimHash(cfg_.dim, 512);
        }

        if (cfg_.use_pq)
        {
            // Auto-configure PQ M (subspaces)
            size_t pq_m = 16;
            while (pq_m > 1 && (cfg_.dim % pq_m != 0))
                pq_m /= 2;

            pq_ = std::make_unique<ProductQuantizer>(cfg_.dim, pq_m, 256); // K=256 fixed for 1 byte/sub
            pq_->train(data, n, 10);
        }

        // Step 4: Pre-allocate initial Buckets in Arena
        // Every centroid gets an empty bucket ready to accept data.
        for (size_t i = 0; i < num_c; ++i)
        {
            centroids_[i].bucket_offset = alloc_new_bucket(static_cast<uint32_t>(i));
        }

        std::cout << "[Orbit] Initialization complete. Ready for orbit.\n";
        return true;
    }

    // --- Helper: Allocate a new Gravity Bucket in Arena ---
    // Calculates strict memory offsets for SoA layout within the blob.
    uint64_t PomaiOrbit::alloc_new_bucket(uint32_t centroid_id)
    {
        // 1. Calculate Layout
        // Layout: [BucketHeader] [Fingerprints...] [PQ Codes...] [Vectors...] [IDs...]

        size_t cap = MAX_BUCKET_CAPACITY;
        size_t head_sz = sizeof(BucketHeader);

        size_t fp_sz = 0;
        if (cfg_.use_fingerprint && fp_)
            fp_sz = fp_->bytes() * cap;

        size_t pq_sz = 0;
        if (cfg_.use_pq && pq_)
            pq_sz = pq_->m() * cap; // raw 8-bit codes (K=256)

        size_t vec_sz = cfg_.dim * sizeof(float) * cap;
        size_t ids_sz = sizeof(uint64_t) * cap; // External Labels

        // Align offsets to 64 bytes for AVX friendliness
        auto align64 = [](size_t s)
        { return (s + 63) & ~63; };

        uint32_t off_fp = static_cast<uint32_t>(align64(head_sz));
        uint32_t off_pq = static_cast<uint32_t>(align64(off_fp + fp_sz));
        uint32_t off_vec = static_cast<uint32_t>(align64(off_pq + pq_sz));
        uint32_t off_ids = static_cast<uint32_t>(align64(off_vec + vec_sz));

        size_t total_bytes = off_ids + ids_sz;

        // 2. Allocate Blob in Arena
        char *blob_ptr = arena_->alloc_blob(static_cast<uint32_t>(total_bytes));
        if (!blob_ptr)
        {
            std::cerr << "[Orbit] CRITICAL: Arena allocation failed for bucket.\n";
            return 0; // 0 is invalid offset in PomaiArena
        }

        // 3. Initialize Header
        // Arena blob structure: [blob_len (4b)] [USER DATA...]
        // So blob_ptr points to the 4-byte length prefix.
        // We need the offset relative to arena base.
        uint64_t offset = arena_->offset_from_blob_ptr(blob_ptr);

        // Skip the 4-byte arena length header to get to User Data (BucketHeader)
        BucketHeader *hdr = reinterpret_cast<BucketHeader *>(blob_ptr + sizeof(uint32_t));

        // Zero init
        std::memset(hdr, 0, sizeof(BucketHeader));

        // Set metadata
        hdr->centroid_id = centroid_id;
        hdr->count.store(0, std::memory_order_relaxed);
        hdr->next_bucket_offset = 0;

        hdr->off_fingerprints = off_fp;
        hdr->off_pq_codes = off_pq;
        hdr->off_vectors = off_vec;
        hdr->off_ids = off_ids;

        return offset;
    }

    // --- Helper: Find Nearest Centroid (Exhaustive Scan for Training) ---
    uint32_t PomaiOrbit::find_nearest_centroid(const float *vec)
    {
        uint32_t best = 0;
        float min_d = std::numeric_limits<float>::max();
        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            float d = l2sq(vec, centroids_[i].vector.data(), cfg_.dim);
            if (d < min_d)
            {
                min_d = d;
                best = static_cast<uint32_t>(i);
            }
        }
        return best;
    }

    // --- Helper: Find Routing Centroids (Greedy Graph Search) ---
    // Used during Insert/Search to find the best entry points in the graph.
    std::vector<uint32_t> PomaiOrbit::find_routing_centroids(const float *vec, size_t n)
    {
        if (centroids_.empty())
            return {};

        // Priority queue for beam search: <Distance, CentroidID>
        using NodeDist = std::pair<float, uint32_t>;
        std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> candidates;
        std::unordered_set<uint32_t> visited;

        // Entry point: Assume 0 for simplicity (or can store a global entry point)
        // Optimization: pick a random one to distribute load if graph is fully connected enough
        uint32_t entry = 0;

        float d = l2sq(vec, centroids_[entry].vector.data(), cfg_.dim);
        candidates.push({d, entry});
        visited.insert(entry);

        std::vector<uint32_t> result;

        // Greedy walk
        // A robust, simplified NSW search
        while (!candidates.empty())
        {
            auto current = candidates.top();
            candidates.pop();

            result.push_back(current.second);
            if (result.size() >= n)
                break; // Collect top N closest encountered

            const auto &node = centroids_[current.second];
            for (uint32_t neighbor_id : node.neighbors)
            {
                if (visited.find(neighbor_id) == visited.end())
                {
                    visited.insert(neighbor_id);
                    float d_n = l2sq(vec, centroids_[neighbor_id].vector.data(), cfg_.dim);
                    candidates.push({d_n, neighbor_id});
                }
            }
        }

        return result;
    }

    // --- INSERT Operation ---
    bool PomaiOrbit::insert(const float *vec, uint64_t label)
    {
        if (centroids_.empty())
            return false;

        // 1. Route to the single nearest centroid
        std::vector<uint32_t> route = find_routing_centroids(vec, 1);
        if (route.empty())
            return false;

        uint32_t cid = route[0];
        OrbitNode &node = centroids_[cid];

        // 2. Lock the node to safely append to its bucket chain
        std::unique_lock<std::shared_mutex> lock(node.mu);

        // 3. Traverse bucket chain to find the last non-full bucket
        uint64_t current_off = node.bucket_offset;
        char *bucket_ptr = nullptr;
        BucketHeader *hdr = nullptr;

        while (current_off != 0)
        {
            const char *blob_ptr = arena_->blob_ptr_from_offset_for_map(current_off);
            if (!blob_ptr)
                return false; // Data corruption?

            // Skip arena length header
            bucket_ptr = const_cast<char *>(blob_ptr) + sizeof(uint32_t);
            hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);

            // Check capacity
            uint32_t count = hdr->count.load(std::memory_order_relaxed);
            if (count < MAX_BUCKET_CAPACITY)
            {
                // Found space!
                break;
            }

            // Bucket full, move to next
            if (hdr->next_bucket_offset == 0)
            {
                // Last bucket is full, allocate new one
                uint64_t new_off = alloc_new_bucket(cid);
                if (new_off == 0)
                    return false;

                // Link old -> new
                hdr->next_bucket_offset = new_off;

                // Move to new
                current_off = new_off;
            }
            else
            {
                current_off = hdr->next_bucket_offset;
            }
        }

        // 4. Append Data to the found Bucket
        // 'hdr' points to the header of the target bucket
        // 'bucket_ptr' points to the start of user data

        uint32_t idx = hdr->count.load(std::memory_order_relaxed);

        // Write Fingerprint
        if (cfg_.use_fingerprint && fp_)
        {
            uint8_t *fp_base = reinterpret_cast<uint8_t *>(bucket_ptr + hdr->off_fingerprints);
            fp_->compute(vec, fp_base + idx * fp_->bytes());
        }

        // Write PQ Code
        if (cfg_.use_pq && pq_)
        {
            uint8_t *pq_base = reinterpret_cast<uint8_t *>(bucket_ptr + hdr->off_pq_codes);
            pq_->encode(vec, pq_base + idx * pq_->m());
        }

        // Write Raw Vector
        {
            float *vec_base = reinterpret_cast<float *>(bucket_ptr + hdr->off_vectors);
            std::memcpy(vec_base + idx * cfg_.dim, vec, cfg_.dim * sizeof(float));
        }

        // Write ID/Label
        {
            uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);
            // Use atomic store for 64-bit alignment safety
            pomai::ai::atomic_utils::atomic_store_u64(id_base + idx, label);
        }

        // 5. Commit (Atomic Increment)
        // This makes the new data visible to readers
        hdr->count.fetch_add(1, std::memory_order_release);

        return true;
    }

    // --- SEARCH Operation ---
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        if (centroids_.empty())
            return {};

        // 1. Find 'nprobe' buckets to scan
        std::vector<uint32_t> targets = find_routing_centroids(query, nprobe);

        // Pre-compute query derived data
        std::vector<uint8_t> qfp;
        if (cfg_.use_fingerprint && fp_)
        {
            qfp.resize(fp_->bytes());
            fp_->compute(query, qfp.data());
        }

        std::vector<float> pq_tables;
        if (cfg_.use_pq && pq_)
        {
            pq_tables.resize(pq_->m() * 256); // K=256
            pq_->compute_distance_tables(query, pq_tables.data());
        }

        // Max heap for top-K results
        using ResPair = std::pair<float, uint64_t>; // <dist, label>
        std::priority_queue<ResPair> results;

        // 2. Scan Buckets
        for (uint32_t cid : targets)
        {
            // Acquire shared lock for the centroid to ensure bucket chain consistency
            // (Though for append-only, relaxed reading is mostly fine, shared lock prevents
            // reading a bucket that is being allocated/linked).
            std::shared_lock<std::shared_mutex> lock(centroids_[cid].mu);

            uint64_t current_off = centroids_[cid].bucket_offset;
            while (current_off != 0)
            {
                const char *blob_ptr = arena_->blob_ptr_from_offset_for_map(current_off);
                if (!blob_ptr)
                    break;

                const char *bucket_ptr = blob_ptr + sizeof(uint32_t);
                const BucketHeader *hdr = reinterpret_cast<const BucketHeader *>(bucket_ptr);

                uint32_t count = hdr->count.load(std::memory_order_acquire);
                if (count == 0)
                {
                    current_off = hdr->next_bucket_offset;
                    continue;
                }

                // --- SCAN LOOP (Sequential Access) ---

                // Pointers to SoA blocks
                const uint8_t *fp_base = reinterpret_cast<const uint8_t *>(bucket_ptr + hdr->off_fingerprints);
                const uint8_t *pq_base = reinterpret_cast<const uint8_t *>(bucket_ptr + hdr->off_pq_codes);
                const float *vec_base = reinterpret_cast<const float *>(bucket_ptr + hdr->off_vectors);
                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(bucket_ptr + hdr->off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    // 2a. Phase 1: Fingerprint Filter
                    if (cfg_.use_fingerprint)
                    {
                        // Use SIMD-optimized Hamming distance
                        uint32_t hamming = pomai::ai::simhash::hamming_dist(
                            qfp.data(),
                            fp_base + i * fp_->bytes(),
                            fp_->bytes());
                        // Threshold logic (hardcoded 128 for now, should be adaptive)
                        if (hamming > 128)
                            continue;
                    }

                    // 2b. Phase 2: PQ Approximation
                    if (cfg_.use_pq)
                    {
                        // Calculate approx distance using lookup tables
                        // Can be optimized with AVX gather in future
                        float approx_dist = 0.0f;
                        const uint8_t *code = pq_base + i * pq_->m();
                        for (size_t m = 0; m < pq_->m(); ++m)
                        {
                            approx_dist += pq_tables[m * 256 + code[m]];
                        }

                        // Pruning heuristic: if approx_dist is significantly worse than current k-th, skip
                        // (Careful: approx dist is not exact, be loose)
                        if (results.size() == k && approx_dist > results.top().first * 1.2f)
                            continue;
                    }

                    // 2c. Phase 3: Exact Distance
                    float dist = l2sq(query, vec_base + i * cfg_.dim, cfg_.dim);

                    // Maintain Heap
                    if (results.size() < k)
                    {
                        uint64_t label = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                        results.push({dist, label});
                    }
                    else if (dist < results.top().first)
                    {
                        results.pop();
                        uint64_t label = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                        results.push({dist, label});
                    }
                }

                current_off = hdr->next_bucket_offset;
            }
        }

        // 3. Format Output
        std::vector<std::pair<uint64_t, float>> final_out;
        final_out.reserve(results.size());
        while (!results.empty())
        {
            auto p = results.top();
            results.pop();
            final_out.emplace_back(p.second, p.first);
        }
        std::reverse(final_out.begin(), final_out.end()); // Nearest first

        return final_out;
    }

    // --- Persistence Stubs (Can be implemented with simple serialization) ---
    bool PomaiOrbit::save_routing(const std::string &path) { return false; } // TODO
    bool PomaiOrbit::load_routing(const std::string &path) { return false; } // TODO

} // namespace pomai::ai::orbit