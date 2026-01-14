/*
 * src/ai/simhash.cc
 *
 * Vectorized SimHash implementation — AVX2 accelerated dot-products with safe scalar fallback.
 *
 * Notes:
 *  - The hot-path dot product uses the adaptive kernel from cpu_kernels.h via pomai_dot.
 *  - hamming_dist moved here to allow an efficient implementation over 64-bit chunks
 *    using __builtin_popcountll. This is portable and much faster than per-byte loops.
 */

#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <immintrin.h> // ok to include; used conditionally

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

    // Helper: compute sign for one projection row (dot product)
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

    // Efficient hamming distance: operate on 64-bit chunks and use builtin popcount.
    // This is fast and portable; for even more speed a platform-specific AVX2/AVX512
    // vectorized popcount can be added later behind a runtime check.
    uint32_t SimHash::hamming_dist(const uint8_t *a, const uint8_t *b, size_t bytes)
    {
        if (!a || !b || bytes == 0)
            return 0;

        size_t i = 0;
        uint32_t dist = 0;

        // process 8-byte blocks
        const size_t BLOCK = sizeof(uint64_t);
        size_t nblocks = bytes / BLOCK;
        const uint64_t *pa = reinterpret_cast<const uint64_t *>(a);
        const uint64_t *pb = reinterpret_cast<const uint64_t *>(b);

        for (size_t k = 0; k < nblocks; ++k)
        {
            uint64_t xa = pa[k];
            uint64_t xb = pb[k];
            dist += POPCOUNT64(xa ^ xb);
        }

        // tail bytes
        i = nblocks * BLOCK;
        for (; i < bytes; ++i)
        {
            uint8_t xa = a[i];
            uint8_t xb = b[i];
            dist += POPCOUNT32(static_cast<unsigned>(xa ^ xb));
        }

        return dist;
    }

} // namespace pomai::ai