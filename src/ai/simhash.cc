/*
 * src/ai/simhash.cc
 *
 * Vectorized SimHash implementation — High Performance & Portable.
 */

#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <random>

// Platform-specific intrinsics for population count
#if defined(_MSC_VER)
#include <intrin.h>
#define POPCNT64(x) __popcnt64(x)
#define POPCNT32(x) __popcnt(x)
#elif defined(__GNUC__) || defined(__clang__)
#define POPCNT64(x) __builtin_popcountll(x)
#define POPCNT32(x) __builtin_popcount(x)
#else
static inline int POPCNT64(uint64_t n)
{
    int c = 0;
    while (n)
    {
        n &= (n - 1);
        c++;
    }
    return c;
}
static inline int POPCNT32(uint32_t n)
{
    int c = 0;
    while (n)
    {
        n &= (n - 1);
        c++;
    }
    return c;
}
#endif

namespace pomai::ai
{

    // [CHANGED] Constructor implementation
    SimHash::SimHash(size_t dim, const pomai::config::FingerprintConfig &cfg, uint64_t seed)
        : dim_(dim), cfg_(cfg)
    {
        bits_ = cfg_.fingerprint_bits;

        if (dim_ == 0)
            throw std::invalid_argument("SimHash: dim must be > 0");
        if (bits_ == 0)
            throw std::invalid_argument("SimHash: bits must be > 0");

        // Calculate required bytes (ceil(bits / 8))
        bytes_ = (bits_ + 7) / 8;

        // Initialize Projections (Gaussian Random)
        // Memory layout: bits_ rows, dim_ columns.
        proj_.resize(bits_ * dim_);

        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (float &v : proj_)
        {
            v = dist(rng);
        }
    }

    inline bool SimHash::dot_sign(const float *vec, const float *proj_row) const
    {
        // Use optimized kernel for dot product
        float d = ::pomai_dot(vec, proj_row, dim_);
        return d >= 0.0f;
    }

    void SimHash::compute(const float *vec, uint8_t *out_bytes) const
    {
        if (!vec || !out_bytes)
            return;

        std::memset(out_bytes, 0, bytes_);

        // Loop over each bit
        for (size_t i = 0; i < bits_; ++i)
        {
            const float *row = &proj_[i * dim_];
            if (dot_sign(vec, row))
            {
                // Set bit i
                out_bytes[i / 8] |= (1 << (i % 8));
            }
        }
    }

    std::vector<uint8_t> SimHash::compute_vec(const float *vec) const
    {
        std::vector<uint8_t> out(bytes_);
        compute(vec, out.data());
        return out;
    }

    void SimHash::compute_words(const float *vec, uint64_t *out_words, size_t word_count) const
    {
        if (!vec || !out_words || word_count == 0)
            return;

        std::memset(out_words, 0, word_count * sizeof(uint64_t));

        // Direct write to words to avoid casting later
        for (size_t i = 0; i < bits_; ++i)
        {
            const float *row = &proj_[i * dim_];
            if (dot_sign(vec, row))
            {
                out_words[i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Hamming Distance
    // -------------------------------------------------------------------------
    uint32_t SimHash::hamming_dist(const uint8_t *a, const uint8_t *b, size_t bytes)
    {
        if (!a || !b || bytes == 0)
            return 0;

        uint32_t dist = 0;
        size_t i = 0;

        // 1. Process 64-bit blocks (Fast Path)
        const size_t step = sizeof(uint64_t);
        size_t limit = bytes >= step ? (bytes - (bytes % step)) : 0;

        for (; i < limit; i += step)
        {
            uint64_t va, vb;
            // Safe unaligned load
            std::memcpy(&va, a + i, sizeof(va));
            std::memcpy(&vb, b + i, sizeof(vb));

            dist += POPCNT64(va ^ vb);
        }

        // 2. Process Tail bytes
        for (; i < bytes; ++i)
        {
            dist += POPCNT32(static_cast<uint32_t>(a[i] ^ b[i]));
        }

        return dist;
    }

} // namespace pomai::ai