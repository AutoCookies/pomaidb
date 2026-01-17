#pragma once
/*
 * src/ai/simhash.h
 *
 * Simple SimHash / sign-bit fingerprint generator for use in SoA prefiltering.
 * Refactored to use centralized pomai::config::FingerprintConfig.
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include "src/core/config.h" // [CHANGED] Include config

// Use GCC/Clang builtin for popcount if available for max performance
#if defined(__GNUC__) || defined(__clang__)
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
        // [CHANGED] Constructor now takes FingerprintConfig
        // Seed is passed separately because it's usually per-instance or from Runtime
        SimHash(size_t dim, const pomai::config::FingerprintConfig &cfg, uint64_t seed);

        // Core API
        void compute(const float *vec, uint8_t *out_bytes) const;
        std::vector<uint8_t> compute_vec(const float *vec) const;
        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const;

        // Accessors
        size_t dim() const noexcept { return dim_; }
        size_t bits() const noexcept { return bits_; }
        size_t bytes() const noexcept { return bytes_; }
        const pomai::config::FingerprintConfig &config() const noexcept { return cfg_; }

        // Compute Hamming distance (Static utility)
        static uint32_t hamming_dist(const uint8_t *a, const uint8_t *b, size_t bytes);

    private:
        size_t dim_;
        size_t bits_;
        size_t bytes_;
        pomai::config::FingerprintConfig cfg_;

        // Projections stored row-major
        std::vector<float> proj_;

        // Helper
        inline bool dot_sign(const float *vec, const float *proj_row) const;
    };

} // namespace pomai::ai