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
#include <random> // <- added: needed for std::mt19937_64 used in kmeans helper

namespace pomai::ai
{

    class ProductQuantizer
    {
    public:
        // Construct a PQ instance:
        // - dim: dimensionality of input vectors
        // - m: number of subquantizers (bytes per code)
        // - k: number of centroids per subquantizer (codebook size, e.g., 256)
        ProductQuantizer(size_t dim, size_t m, size_t k);

        ~ProductQuantizer();

        // Accessors
        size_t dim() const noexcept { return dim_; }
        size_t m() const noexcept { return m_; }
        size_t k() const noexcept { return k_; }
        size_t subdim() const noexcept { return subdim_; }

        // Train codebooks with `n_samples` samples. Samples layout: consecutive vectors
        // each of length `dim`. This performs independent k-means on each subquantizer.
        // max_iters controls the per-subquantizer k-means iterations.
        void train(const float *samples, size_t n_samples, size_t max_iters = 10);

        // Encode a single vector `vec` (length dim) into `out_codes` (must have m bytes).
        // Each byte is a centroid index in [0..k-1].
        void encode(const float *vec, uint8_t *out_codes) const;

        // Convenience: encode into a std::vector<uint8_t> of length m.
        std::vector<uint8_t> encode_vec(const float *vec) const;

        // Pack m 8-bit codes into (m+1)/2 bytes of 4-bit nibbles.
        // - src8: pointer to m bytes (each value assumed < 16 when packing to 4-bit)
        // - dst_nibbles: buffer of length packed4BytesPerVec(m)
        static void pack4From8(const uint8_t *src8, uint8_t *dst_nibbles, size_t m);

        // Unpack 4-bit nibbles into m 8-bit codes (values in [0..15]).
        static void unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8, size_t m);

        // Helper: number of bytes needed to store packed-4 representation for m subquantizers.
        static inline size_t packed4BytesPerVec(size_t m) noexcept { return (m + 1) / 2; }

        // Optional: save/load codebooks to/from a simple binary file (for reuse).
        bool save_codebooks(const std::string &path) const;
        bool load_codebooks(const std::string &path);

    private:
        size_t dim_;
        size_t m_;
        size_t k_;
        size_t subdim_; // floor(dim / m) or last chunk larger if dim not divisible

        // Codebooks stored as contiguous floats: codebooks_[sub * k * subdim + centroid * subdim + d]
        // Layout: for sub in [0..m-1], for centroid in [0..k-1], for d in [0..subdim-1]
        std::vector<float> codebooks_;

        // Internal helpers
        void kmeans_per_sub(const float *samples, size_t n_samples, size_t sub, size_t max_iters, std::mt19937_64 &rng);
    };

} // namespace pomai::ai