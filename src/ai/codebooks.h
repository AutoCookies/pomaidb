/*
 * src/ai/codebooks.h
 *
 * In-memory codebooks container and query-time helpers for Product Quantization.
 *
 * Responsibilities:
 *  - Hold PQ codebooks in RAM in a simple contiguous layout:
 *      codebooks_[ sub * (k * subdim) + centroid * subdim + d ]
 *  - Provide fast access to centroid pointers for a given (sub, centroid).
 *  - Compute per-query lookup tables: for a given query vector compute an
 *    array of size (m * k) of L2 distances between each sub-centroid and
 *    the corresponding query subvector. This is the core work performed
 *    once per query when using table-lookup PQ evaluation.
 *
 * Design notes:
 *  - The container is lightweight and copyable/movable (std::vector backing).
 *  - compute_distance_tables writes into a preallocated contiguous buffer
 *    (size must be >= m * k). This avoids nested vector allocations in hot path.
 *
 * Usage:
 *   Codebooks cb(dim, m, k);
 *   cb.load_from_pqfile("pq.bin");  // optional
 *   std::vector<float> tables(m*k);
 *   cb.compute_distance_tables(query_vec, tables.data());
 *   // tables[sub*k + centroid] => distance
 *
 * Thread-safety:
 *  - The object is read-only after construction (const methods are thread-safe).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace pomai::ai
{

    class Codebooks
    {
    public:
        // Construct empty container (must call load or set_codebooks before use).
        Codebooks() = default;

        // Construct with explicit sizes. codebooks memory is zero-initialized.
        Codebooks(size_t dim, size_t m, size_t k);

        // Initialize from an existing raw codebooks vector (must be sized m*k*subdim).
        // Takes ownership (copy) of provided vector.
        Codebooks(size_t dim, size_t m, size_t k, const std::vector<float> &raw_codebooks);

        ~Codebooks() = default;

        // Load codebooks previously saved with ProductQuantizer::save_codebooks().
        // Returns true on success.
        bool load_from_file(const std::string &path);

        // Save codebooks to file (compatible with ProductQuantizer::save_codebooks()).
        bool save_to_file(const std::string &path) const;

        // Accessors
        size_t dim() const noexcept { return dim_; }
        size_t m() const noexcept { return m_; }
        size_t k() const noexcept { return k_; }
        size_t subdim() const noexcept { return subdim_; }

        // Return pointer to centroid data for given sub and centroid index.
        // Layout: pointer points to an array of length subdim() floats.
        // No bounds checking in this accessor (use for performance in hot path).
        const float *centroid_ptr(size_t sub, size_t centroid) const noexcept
        {
            return &codebooks_[sub * (k_ * subdim_) + centroid * subdim_];
        }

        // Compute per-sub, per-centroid squared L2 distances between query and centroids.
        // - query: pointer to float vector of length dim()
        // - out_tables: pre-allocated buffer of size at least m() * k()
        //   write layout: out_tables[sub * k() + centroid] = dist (float)
        // This function is performance-critical; it is implemented straightforwardly
        // and can be vectorized later if needed.
        void compute_distance_tables(const float *query, float *out_tables) const;

        // Fill codebooks from raw vector (must be length m * k * subdim)
        void set_codebooks_from_raw(size_t dim, size_t m, size_t k, const std::vector<float> &raw);

        // Raw access (const)
        const std::vector<float> &raw_codebooks() const noexcept { return codebooks_; }

    private:
        size_t dim_{0};
        size_t m_{0};
        size_t k_{0};
        size_t subdim_{0};

        // contiguous storage: size = m * k * subdim (sub-major)
        std::vector<float> codebooks_;
    };

} // namespace pomai::ai