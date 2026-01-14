/*
 * src/ai/simhash.cc
 *
 * Vectorized SimHash implementation — AVX2 accelerated dot-products with safe scalar fallback.
 *
 * Notes:
 *  - We keep the public API unchanged (SimHash::compute, compute_vec, compute_words).
 *  - At runtime, when compiled with AVX2 support, we use __builtin_cpu_supports("avx2")
 *    to guard the AVX2 kernel for portability on heterogeneous machines.
 *  - This implementation focuses on accelerating the inner dot-product (the hot path).
 */

#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h" // use pomai_dot adaptive kernel

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace pomai::ai
{

    SimHash::SimHash(size_t dim, size_t bits, uint64_t seed)
        : dim_(dim), bits_(bits), bytes_((bits + 7) / 8), proj_()
    {
        if (dim_ == 0)
            throw std::invalid_argument("SimHash: dim must be > 0");
        if (bits_ == 0)
            throw std::invalid_argument("SimHash: bits must be > 0");

        // Reserve storage for projection matrix (bits * dim floats)
        proj_.resize(bits_ * dim_);

        // Initialize RNG and sample standard normal floats for projections
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);

        for (size_t b = 0; b < bits_; ++b)
        {
            float *row = &proj_[b * dim_];
            for (size_t d = 0; d < dim_; ++d)
                row[d] = nd(rng);
        }
    }

    SimHash::~SimHash() = default;

    // Vectorized dot-sign: use adaptive kernel via pomai_dot
    inline bool SimHash::dot_sign(const float *vec, const float *proj_row) const
    {
        // Call adaptive dot kernel from cpu_kernels.h (selects AVX2/AVX512/NEON/scalar at init)
        float acc = ::pomai_dot(vec, proj_row, dim_);
        return acc >= 0.0f;
    }

    void SimHash::compute(const float *vec, uint8_t *out_bytes) const
    {
        if (!vec || !out_bytes)
            return;

        std::memset(out_bytes, 0, bytes_);

        for (size_t b = 0; b < bits_; ++b)
        {
            const float *row = &proj_[b * dim_];
            bool bit = dot_sign(vec, row);
            if (bit)
            {
                size_t byte_idx = b >> 3; // b / 8
                uint8_t bit_mask = static_cast<uint8_t>(1u << (b & 7));
                out_bytes[byte_idx] |= bit_mask;
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
        size_t needed = (bits_ + 63) / 64;
        if (word_count < needed)
            throw std::invalid_argument("SimHash::compute_words: insufficient word_count");

        for (size_t i = 0; i < word_count; ++i)
            out_words[i] = 0ULL;

        for (size_t b = 0; b < bits_; ++b)
        {
            const float *row = &proj_[b * dim_];
            bool bit = dot_sign(vec, row);
            if (bit)
            {
                size_t word_idx = b >> 6; // b / 64
                unsigned pos = static_cast<unsigned>(b & 63);
                out_words[word_idx] |= (1ULL << pos);
            }
        }
    }

} // namespace pomai::ai