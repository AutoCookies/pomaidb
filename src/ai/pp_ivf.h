#pragma once
// src/ai/pp_ivf.h
// Minimal PP-IVF (IVF + simple PQ) MVP to reduce memory footprint and provide a
// cluster-probe filter for VectorStore searches.
// NOTE: This is an MVP: centroids/codebooks are initialized with simple heuristics.
// Replace with proper KMeans/PQ training for production.

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <limits>
#include <algorithm>

namespace pomai::ai
{

    class PPIVF
    {
    public:
        // dim: vector dimensionality
        // k_clusters: number of coarse clusters
        // m_sub: number of PQ subspaces
        // nbits: bits per subquant (typically 8)
        PPIVF(size_t dim, size_t k_clusters = 1024, size_t m_sub = 16, size_t nbits = 8);
        ~PPIVF();

        // Initialize centroids/codebooks with a cheap random scheme (MVP).
        // For production replace with k-means training over dataset or offline tool.
        bool init_random_seed(uint64_t seed = 12345);

        // Assign a vector to cluster (nearest centroid)
        int assign_cluster(const float *vec) const;

        // Encode a vector to PQ codes (m bytes when nbits==8)
        // returns pointer to internal buffer (owned by PPIVF) or nullptr on error
        const uint8_t *encode_pq(const float *vec);

        // Register label -> cluster and store PQ code for later retrieval
        void add_label(size_t label, int cluster, const uint8_t *code);

        // Probe top-k clusters for a query (linear scan of centroids in MVP)
        std::vector<int> probe_clusters(const float *query, size_t probe_k) const;

        // Access stored code for a label; returns nullptr if not found
        const uint8_t *get_code_for_label(size_t label) const;

        // Get cluster for a label; -1 if not found
        int get_cluster_for_label(size_t label) const;

        // Basic stats
        size_t num_clusters() const noexcept { return k_; }
        size_t dim() const noexcept { return dim_; }
        size_t pq_bytes_per_code() const noexcept { return m_sub_; }

    private:
        size_t dim_;
        size_t k_;     // number of clusters
        size_t m_sub_; // PQ subspaces (bytes per code when nbits==8)
        size_t nbits_;

        // Centroids: k_ * dim_
        std::vector<float> centroids_;

        // For this MVP we implement per-subspace uniform scalar quantization
        // per-subspace codebooks are not pre-trained; encode_pq uses a simple mapping.
        std::vector<uint8_t> code_scratch_; // m_sub_ bytes scratch

        // label -> cluster + label -> code
        mutable std::mutex mu_;
        std::unordered_map<uint64_t, int> label_to_cluster_;
        std::unordered_map<uint64_t, std::vector<uint8_t>> label_to_code_;

        // RNG for initialization
        std::mt19937_64 rng_;

        // Helpers
        static inline float l2sq_distance(const float *a, const float *b, size_t dim);
        void compute_centroid_randomly();
    };

} // namespace pomai::ai