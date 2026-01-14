#pragma once
/*
 * src/ai/simhash.h
 *
 * Simple SimHash / sign-bit fingerprint generator for use in SoA prefiltering.
 *
 * - Produces a bit-packed fingerprint (N bits) for an input float vector.
 * - Default configuration: 512 bits (recommended for good prefilter selectivity).
 *
 * Updates:
 * - Moved hamming_dist implementation to simhash.cc so we can provide
 *   a tuned implementation (64-bit chunks + builtin popcount / optional SIMD).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>

// Use GCC/Clang builtin for popcount if available for max performance
#ifdef __GNUC__
#define POPCOUNT64(x) __builtin_popcountll(x)
#define POPCOUNT32(x) __builtin_popcount(x)
#else
#include <bit>
#define POPCOUNT64(x) std::popcount(x)
#define POPCOUNT32(x) std::popcount(x)
#endif

namespace pomai::ai
{

    class SimHash
    {
    public:
        // Construct a SimHash object.
        // - dim: input vector dimensionality (number of floats per vector)
        // - bits: number of sign-bits to produce (typical 256 or 512). If 0, factory may pick runtime default.
        // - seed: RNG seed for reproducible projection matrices
        SimHash(size_t dim, size_t bits = 512, uint64_t seed = 123456789ULL);

        ~SimHash();

        // Accessors
        size_t dim() const noexcept { return dim_; }
        size_t bits() const noexcept { return bits_; }
        // Number of output bytes required to hold the bitpacked fingerprint
        size_t bytes() const noexcept { return bytes_; }

        // Compute the bitpacked fingerprint for `vec` (length == dim()) and write
        // into `out_bytes` which must have at least bytes() bytes available.
        // The fingerprint is packed little-endian inside each byte: bit 0 is LSB.
        void compute(const float *vec, uint8_t *out_bytes) const;

        // Convenience: compute and return a std::vector<uint8_t> of length bytes().
        std::vector<uint8_t> compute_vec(const float *vec) const;

        // Convenience: compute into an array of uint64_t words (word_count must be >= (bits+63)/64).
        // Unused high bits in final word are zeroed.
        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const;

    private:
        size_t dim_;
        size_t bits_;
        size_t bytes_;

        // Projections stored row-major: proj_[bit_index * dim_ + d]
        // Note: this uses bits_ * dim_ floats of memory.
        std::vector<float> proj_;

        // Helper: compute sign for one projection row (dot product)
        inline bool dot_sign(const float *vec, const float *proj_row) const;

    public:
        // Compute Hamming distance between two bit-packed fingerprints (a,b) of length `bytes`.
        // Tuned implementation lives in simhash.cc (64-bit-chunk loop + builtin popcount).
        static uint32_t hamming_dist(const uint8_t *a, const uint8_t *b, size_t bytes);
    };

    // --- Helper namespace for distance functions (Required by HolographicScanner) ---
    // hamming_dist declared above and implemented in simhash.cc

} // namespace pomai::ai