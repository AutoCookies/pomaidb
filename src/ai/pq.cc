/*
 * src/ai/pq.cc
 *
 * Implementation for ProductQuantizer.
 *
 * This file provides:
 *  - per-subquantizer k-means training
 *  - encode() to produce m-byte PQ codes
 *  - pack4/unpack4 helpers for 4-bit on-disk storage
 *  - simple save/load for codebooks (raw binary)
 *  - compute_distance_tables for query-time PQ table computation
 *
 * The k-means implementation is intentionally simple and robust. It is
 * adequate for offline training in the prototype/PoC phase.
 */

#include "src/ai/pq.h"
#include "src/core/cpu_kernels.h" // <- added to reuse l2sq kernel

#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <fstream>
#include <stdexcept>

namespace pomai::ai
{

    ProductQuantizer::ProductQuantizer(size_t dim, size_t m, size_t k)
        : dim_(dim), m_(m), k_(k)
    {
        if (dim_ == 0)
            throw std::invalid_argument("ProductQuantizer: dim must be > 0");
        if (m_ == 0)
            throw std::invalid_argument("ProductQuantizer: m must be > 0");
        if (k_ < 2)
            throw std::invalid_argument("ProductQuantizer: k must be >= 2");

        // Compute subdim; we distribute dims evenly across m subquantizers.
        subdim_ = dim_ / m_;
        if (subdim_ == 0)
            subdim_ = 1;

        // Allocate codebooks: m * k * subdim floats
        codebooks_.assign(m_ * k_ * subdim_, 0.0f);
    }

    ProductQuantizer::~ProductQuantizer() = default;

    void ProductQuantizer::train(const float *samples, size_t n_samples, size_t max_iters)
    {
        if (!samples || n_samples == 0)
            throw std::invalid_argument("ProductQuantizer::train: samples empty");

        // RNG for centroid initialization
        std::mt19937_64 rng(1234567ULL);

        for (size_t sub = 0; sub < m_; ++sub)
        {
            kmeans_per_sub(samples, n_samples, sub, max_iters, rng);
        }
    }

    void ProductQuantizer::kmeans_per_sub(const float *samples, size_t n_samples, size_t sub, size_t max_iters, std::mt19937_64 &rng)
    {
        size_t sub_off = sub * subdim_;
        // Effective length of this subvector for last subquantizer may include remainder
        size_t sub_len = subdim_;
        // If dim not divisible evenly, extend last subquantizer to include remainder
        if (sub == m_ - 1)
        {
            size_t used = subdim_ * (m_ - 1);
            if (dim_ > used)
                sub_len = dim_ - used;
        }

        if (sub_len == 0)
            return;

        // Collect pointers to subvector start for each sample to avoid repeated pointer arithmetic
        std::vector<const float *> subs;
        subs.reserve(n_samples);
        for (size_t i = 0; i < n_samples; ++i)
        {
            subs.push_back(samples + i * dim_ + sub_off);
        }

        // Initialize centroids by picking k distinct random samples (or repeating if fewer)
        std::uniform_int_distribution<size_t> ud(0, n_samples - 1);
        std::vector<float> centroids(k_ * sub_len);
        for (size_t c = 0; c < k_; ++c)
        {
            size_t idx = ud(rng);
            const float *src = subs[idx];
            for (size_t d = 0; d < sub_len; ++d)
                centroids[c * sub_len + d] = src[d];
        }

        std::vector<int> assignments(n_samples, 0);
        for (size_t iter = 0; iter < max_iters; ++iter)
        {
            bool changed = false;

            // Assignment step
            for (size_t i = 0; i < n_samples; ++i)
            {
                const float *sv = subs[i];
                // find nearest centroid
                float bestd = std::numeric_limits<float>::infinity();
                int bestc = 0;
                for (size_t c = 0; c < k_; ++c)
                {
                    const float *cb = &centroids[c * sub_len];
                    double sum = 0.0;
                    for (size_t d = 0; d < sub_len; ++d)
                    {
                        double diff = static_cast<double>(sv[d]) - static_cast<double>(cb[d]);
                        sum += diff * diff;
                    }
                    float dist = static_cast<float>(sum);
                    if (dist < bestd)
                    {
                        bestd = dist;
                        bestc = static_cast<int>(c);
                    }
                }
                if (assignments[i] != bestc)
                {
                    assignments[i] = bestc;
                    changed = true;
                }
            }

            // If no assignment changed, converged
            if (!changed)
                break;

            // Update centroids: compute sums and counts
            std::vector<size_t> counts(k_, 0);
            std::vector<double> sums(k_ * sub_len, 0.0);

            for (size_t i = 0; i < n_samples; ++i)
            {
                int c = assignments[i];
                const float *sv = subs[i];
                for (size_t d = 0; d < sub_len; ++d)
                    sums[c * sub_len + d] += static_cast<double>(sv[d]);
                counts[c]++;
            }

            for (size_t c = 0; c < k_; ++c)
            {
                if (counts[c] == 0)
                {
                    // Reinitialize centroid from a random sample
                    size_t idx = ud(rng);
                    const float *src = subs[idx];
                    for (size_t d = 0; d < sub_len; ++d)
                        centroids[c * sub_len + d] = src[d];
                }
                else
                {
                    for (size_t d = 0; d < sub_len; ++d)
                        centroids[c * sub_len + d] = static_cast<float>(sums[c * sub_len + d] / static_cast<double>(counts[c]));
                }
            }
        }

        // Copy centroids into global codebooks_ storage
        // codebooks_[sub * k * subdim + c * subdim + d] = centroids[c * sub_len + d]
        for (size_t c = 0; c < k_; ++c)
        {
            for (size_t d = 0; d < sub_len; ++d)
            {
                size_t dst_idx = sub * (k_ * subdim_) + c * subdim_ + d;
                codebooks_[dst_idx] = centroids[c * sub_len + d];
            }
            // If sub_len < subdim_ (should not happen except for last), zero pad
            for (size_t d = sub_len; d < subdim_; ++d)
            {
                size_t dst_idx = sub * (k_ * subdim_) + c * subdim_ + d;
                codebooks_[dst_idx] = 0.0f;
            }
        }
    }

    void ProductQuantizer::encode(const float *vec, uint8_t *out_codes) const
    {
        if (!vec || !out_codes)
            return;

        for (size_t sub = 0; sub < m_; ++sub)
        {
            size_t sub_off = sub * subdim_;
            size_t sub_len = subdim_;
            if (sub == m_ - 1)
            {
                size_t used = subdim_ * (m_ - 1);
                if (dim_ > used)
                    sub_len = dim_ - used;
            }

            // Find nearest centroid index
            int bestc = 0;
            double bestd = std::numeric_limits<double>::infinity();

            for (size_t c = 0; c < k_; ++c)
            {
                const float *cb = &codebooks_[sub * (k_ * subdim_) + c * subdim_];
                double sum = 0.0;
                for (size_t d = 0; d < sub_len; ++d)
                {
                    double diff = static_cast<double>(vec[sub_off + d]) - static_cast<double>(cb[d]);
                    sum += diff * diff;
                }
                if (sum < bestd)
                {
                    bestd = sum;
                    bestc = static_cast<int>(c);
                }
            }
            out_codes[sub] = static_cast<uint8_t>(bestc & 0xFFu);
        }
    }

    std::vector<uint8_t> ProductQuantizer::encode_vec(const float *vec) const
    {
        std::vector<uint8_t> out(m_);
        encode(vec, out.data());
        return out;
    }

    void ProductQuantizer::pack4From8(const uint8_t *src8, uint8_t *dst_nibbles, size_t m)
    {
        size_t bi = 0;
        for (size_t i = 0; i < m; i += 2)
        {
            uint8_t lo = src8[i] & 0x0F;
            uint8_t hi = 0;
            if (i + 1 < m)
                hi = src8[i + 1] & 0x0F;
            dst_nibbles[bi++] = static_cast<uint8_t>((hi << 4) | lo);
        }
        // zero pad rest if any (shouldn't be necessary but defensive)
        size_t expected = packed4BytesPerVec(m);
        for (; bi < expected; ++bi)
            dst_nibbles[bi] = 0;
    }

    void ProductQuantizer::unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8, size_t m)
    {
        size_t bi = 0;
        for (size_t i = 0; i < m; i += 2)
        {
            uint8_t v = src_nibbles[bi++];
            dst8[i] = v & 0x0F;
            if (i + 1 < m)
                dst8[i + 1] = (v >> 4) & 0x0F;
        }
    }

    bool ProductQuantizer::save_codebooks(const std::string &path) const
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f)
            return false;
        // write header: dim,m,k,subdim as uint64/uint32 for robustness
        uint64_t dim64 = static_cast<uint64_t>(dim_);
        uint64_t m64 = static_cast<uint64_t>(m_);
        uint64_t k64 = static_cast<uint64_t>(k_);
        uint64_t subdim64 = static_cast<uint64_t>(subdim_);
        f.write(reinterpret_cast<const char *>(&dim64), sizeof(dim64));
        f.write(reinterpret_cast<const char *>(&m64), sizeof(m64));
        f.write(reinterpret_cast<const char *>(&k64), sizeof(k64));
        f.write(reinterpret_cast<const char *>(&subdim64), sizeof(subdim64));
        if (!f)
            return false;
        // write raw floats
        f.write(reinterpret_cast<const char *>(codebooks_.data()), static_cast<std::streamsize>(codebooks_.size() * sizeof(float)));
        return f.good();
    }

    bool ProductQuantizer::load_codebooks(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return false;
        uint64_t dim64 = 0, m64 = 0, k64 = 0, subdim64 = 0;
        f.read(reinterpret_cast<char *>(&dim64), sizeof(dim64));
        f.read(reinterpret_cast<char *>(&m64), sizeof(m64));
        f.read(reinterpret_cast<char *>(&k64), sizeof(k64));
        f.read(reinterpret_cast<char *>(&subdim64), sizeof(subdim64));
        if (!f)
            return false;
        if (static_cast<size_t>(dim64) != dim_ || static_cast<size_t>(m64) != m_ || static_cast<size_t>(k64) != k_ || static_cast<size_t>(subdim64) != subdim_)
            return false;
        size_t expected = m_ * k_ * subdim_;
        codebooks_.resize(expected);
        f.read(reinterpret_cast<char *>(codebooks_.data()), static_cast<std::streamsize>(expected * sizeof(float)));
        return f.good();
    }

    void ProductQuantizer::compute_distance_tables(const float *query, float *out_tables) const
    {
        if (!query || !out_tables)
            return;

        // for each subquantizer
        for (size_t sub = 0; sub < m_; ++sub)
        {
            size_t sub_off = sub * subdim_;
            size_t sub_len = subdim_;
            if (sub == m_ - 1)
            {
                size_t used = subdim_ * (m_ - 1);
                if (dim_ > used)
                    sub_len = dim_ - used;
            }

            const float *centroids_base = &codebooks_[sub * (k_ * subdim_)];
            for (size_t c = 0; c < k_; ++c)
            {
                const float *centroid = centroids_base + c * subdim_;
                // Use optimized L2 kernel for subvector distance
                float dist = l2sq(query + sub_off, centroid, sub_len);
                out_tables[sub * k_ + c] = dist;
            }
        }
    }

    bool ProductQuantizer::load_codebooks_from_buffer(const float *src, size_t float_count)
    {
        if (!src)
            return false;
        size_t expected = m_ * k_ * subdim_;
        if (float_count != expected)
            return false;
        codebooks_.assign(src, src + float_count);
        return true;
    }

} // namespace pomai::ai