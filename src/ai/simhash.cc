/*
 * src/ai/simhash.cc
 *
 * Implementation for SimHash (dense random projections).
 *
 * Clean, commented, straightforward C++ implementation prioritized for
 * readability and correctness. Optimizations (SIMD, blocking, sparse
 * projections) are left for later stages.
 */

#include "src/ai/simhash.h"

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
        // Use normal distribution with mean 0, std 1
        std::normal_distribution<float> nd(0.0f, 1.0f);

        // Fill projection matrix row by row
        for (size_t b = 0; b < bits_; ++b)
        {
            float *row = &proj_[b * dim_];
            // Draw dim_ floats for this row
            for (size_t d = 0; d < dim_; ++d)
                row[d] = nd(rng);
        }
    }

    SimHash::~SimHash() = default;

    inline bool SimHash::dot_sign(const float *vec, const float *proj_row) const
    {
        // Compute dot product; we only need sign, so we can accumulate in double for precision.
        double acc = 0.0;
        for (size_t i = 0; i < dim_; ++i)
            acc += static_cast<double>(vec[i]) * static_cast<double>(proj_row[i]);
        return acc >= 0.0;
    }

    void SimHash::compute(const float *vec, uint8_t *out_bytes) const
    {
        if (!vec || !out_bytes)
            return;

        // Zero output bytes initially
        std::memset(out_bytes, 0, bytes_);

        // For each bit projection, compute sign and set the corresponding output bit.
        for (size_t b = 0; b < bits_; ++b)
        {
            const float *row = &proj_[b * dim_];
            bool bit = dot_sign(vec, row);
            if (bit)
            {
                size_t byte_idx = b >> 3;                               // b / 8
                uint8_t bit_mask = static_cast<uint8_t>(1u << (b & 7)); // LSB = bit 0
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

        // Zero words
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