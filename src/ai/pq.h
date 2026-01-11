/*
 * src/ai/pq.h
 *
 * Lightweight Product Quantizer (PQ) helper.
 *
 * - Supports:
 *   - Training per-subquantizer k-means (simple, reproducible).
 *   - Encoding a vector into an m-byte code (one uint8 per subquantizer).
 *   - Packing 8-bit codes into 4-bit packed bytes for on-disk storage.
 *   - Unpacking 4-bit packed bytes back into 8-bit codes.
 *   - Compute per-query distance tables (m * k floats) used by pq_eval.
 *
 * - Design goals:
 *   - Simple, clear API suitable for integration into the Pomai Thaut65 pipeline.
 *   - No external dependencies beyond the standard library.
 *   - Documented and easy to replace with a more optimized implementation later.
 *
 * Notes:
 * - Input vectors are assumed to be contiguous float arrays of length `dim`.
 * - `dim` must be divisible by `m` (subdim = dim / m); if not, the last subquantizer
 *   will take the remaining dimensions.
 *
 * Usage:
 *   ProductQuantizer pq(dim, m, k);
 *   pq.train(samples, n_samples, max_iters);
 *   std::vector<uint8_t> code(m);
 *   pq.encode(vec, code.data());
 *   // optionally pack to 4-bit:
 *   std::vector<uint8_t> nib((m+1)/2);
 *   ProductQuantizer::pack4From8(code.data(), nib.data(), m);
 *
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <random> // <- needed for std::mt19937_64 used in kmeans helper

namespace pomai::ai
{

    class ProductQuantizer
    {
    public:
        ProductQuantizer(size_t dim, size_t m, size_t k);
        ~ProductQuantizer();

        // Accessors
        size_t dim() const noexcept { return dim_; }
        size_t m() const noexcept { return m_; }
        size_t k() const noexcept { return k_; }
        size_t subdim() const noexcept { return subdim_; }

        // Train and encode
        void train(const float *samples, size_t n_samples, size_t max_iters = 10);
        void encode(const float *vec, uint8_t *out_codes) const;
        std::vector<uint8_t> encode_vec(const float *vec) const;

        // 4-bit packing helpers
        static void pack4From8(const uint8_t *src8, uint8_t *dst_nibbles, size_t m);
        static void unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8, size_t m);
        static inline size_t packed4BytesPerVec(size_t m) noexcept { return (m + 1) / 2; }

        // save/load codebooks to file
        bool save_codebooks(const std::string &path) const;
        bool load_codebooks(const std::string &path);

        // per-query tables
        void compute_distance_tables(const float *query, float *out_tables) const;

        // in-memory codebook helpers (for embedding into SoA)
        const float *codebooks_data() const noexcept { return codebooks_.empty() ? nullptr : codebooks_.data(); }
        size_t codebooks_float_count() const noexcept { return codebooks_.size(); }
        bool load_codebooks_from_buffer(const float *src, size_t float_count);

    private:
        size_t dim_;
        size_t m_;
        size_t k_;
        size_t subdim_;
        std::vector<float> codebooks_;

        void kmeans_per_sub(const float *samples, size_t n_samples, size_t sub, size_t max_iters, std::mt19937_64 &rng);
    };

} // namespace pomai::ai