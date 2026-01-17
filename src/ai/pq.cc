#include "src/ai/pq.h"
#include "src/core/cpu_kernels.h"

#include <random>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <fstream>
#include <stdexcept>

namespace pomai::ai
{

    ProductQuantizer::ProductQuantizer(size_t dim, const pomai::config::PQConfig &cfg)
        : dim_(dim), cfg_(cfg)
    {
        m_ = static_cast<size_t>(cfg_.m_subquantizers);
        k_ = static_cast<size_t>(cfg_.k_centroids);

        if (dim_ == 0)
            throw std::invalid_argument("ProductQuantizer: dim must be > 0");
        if (m_ == 0)
            throw std::invalid_argument("ProductQuantizer: m must be > 0");
        if (k_ < 2)
            throw std::invalid_argument("ProductQuantizer: k must be >= 2");
        if (dim_ < m_)
            throw std::invalid_argument("ProductQuantizer: dim must be >= m");

        subdim_ = dim_ / m_;
    }

    void ProductQuantizer::train(const float *samples, size_t n_samples)
    {
        if (!samples || n_samples == 0)
            return;

        size_t max_iters = static_cast<size_t>(cfg_.train_iters);
        if (max_iters == 0)
            max_iters = 1;

        size_t total_floats = 0;
        for (size_t i = 0; i < m_; ++i)
        {
            size_t sd = subdim_;
            if (i == m_ - 1)
                sd = dim_ - (subdim_ * (m_ - 1));
            total_floats += k_ * sd;
        }
        codebooks_.resize(total_floats);

        std::mt19937 rng(static_cast<unsigned int>(cfg_.seed));

        size_t codebook_offset = 0;
        for (size_t sub = 0; sub < m_; ++sub)
        {
            size_t current_subdim = subdim_;
            if (sub == m_ - 1)
                current_subdim = dim_ - (subdim_ * (m_ - 1));

            float *sub_centroids = &codebooks_[codebook_offset];

            std::vector<size_t> indices(n_samples);
            for (size_t i = 0; i < n_samples; ++i)
                indices[i] = i;
            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t c = 0; c < k_; ++c)
            {
                size_t sample_idx = indices[c % n_samples];
                size_t vec_offset_bytes = sub * subdim_;
                std::memcpy(sub_centroids + c * current_subdim,
                            samples + sample_idx * dim_ + vec_offset_bytes,
                            current_subdim * sizeof(float));
            }

            std::vector<int> counts(k_);
            std::vector<float> next_centroids(k_ * current_subdim);

            for (size_t iter = 0; iter < max_iters; ++iter)
            {
                std::fill(counts.begin(), counts.end(), 0);
                std::fill(next_centroids.begin(), next_centroids.end(), 0.0f);

                for (size_t i = 0; i < n_samples; ++i)
                {
                    size_t vec_offset_bytes = sub * subdim_;
                    const float *vec_sub = samples + i * dim_ + vec_offset_bytes;

                    size_t best_c = 0;
                    float min_dist = std::numeric_limits<float>::max();

                    for (size_t c = 0; c < k_; ++c)
                    {
                        float d = l2sq(vec_sub, sub_centroids + c * current_subdim, current_subdim);
                        if (d < min_dist)
                        {
                            min_dist = d;
                            best_c = c;
                        }
                    }

                    counts[best_c]++;
                    for (size_t d = 0; d < current_subdim; ++d)
                    {
                        next_centroids[best_c * current_subdim + d] += vec_sub[d];
                    }
                }

                for (size_t c = 0; c < k_; ++c)
                {
                    if (counts[c] > 0)
                    {
                        float scale = 1.0f / counts[c];
                        for (size_t d = 0; d < current_subdim; ++d)
                        {
                            sub_centroids[c * current_subdim + d] = next_centroids[c * current_subdim + d] * scale;
                        }
                    }
                }
            }
            codebook_offset += k_ * current_subdim;
        }
    }

    void ProductQuantizer::encode(const float *vec, uint8_t *out_codes) const
    {
        size_t codebook_offset = 0;
        for (size_t sub = 0; sub < m_; ++sub)
        {
            size_t current_subdim = subdim_;
            if (sub == m_ - 1)
                current_subdim = dim_ - (subdim_ * (m_ - 1));

            size_t vec_offset = sub * subdim_;
            const float *vec_sub = vec + vec_offset;
            const float *sub_centroids = &codebooks_[codebook_offset];

            size_t best_c = 0;
            float min_dist = std::numeric_limits<float>::max();

            for (size_t c = 0; c < k_; ++c)
            {
                float d = l2sq(vec_sub, sub_centroids + c * current_subdim, current_subdim);
                if (d < min_dist)
                {
                    min_dist = d;
                    best_c = c;
                }
            }
            out_codes[sub] = static_cast<uint8_t>(best_c);
            codebook_offset += k_ * current_subdim;
        }
    }

    std::vector<uint8_t> ProductQuantizer::encode_vec(const float *vec) const
    {
        std::vector<uint8_t> codes(m_);
        encode(vec, codes.data());
        return codes;
    }

    void ProductQuantizer::pack4From8(const uint8_t *src8, uint8_t *dst_nibbles, size_t m)
    {
        for (size_t i = 0; i < m; i += 2)
        {
            uint8_t low = src8[i] & 0x0F;
            uint8_t high = (i + 1 < m) ? (src8[i + 1] & 0x0F) : 0;
            dst_nibbles[i / 2] = low | (high << 4);
        }
    }

    void ProductQuantizer::unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8, size_t m)
    {
        for (size_t i = 0; i < m; i += 2)
        {
            uint8_t byte = src_nibbles[i / 2];
            dst8[i] = byte & 0x0F;
            if (i + 1 < m)
                dst8[i + 1] = (byte >> 4) & 0x0F;
        }
    }

    bool ProductQuantizer::save_codebooks(const std::string &path) const
    {
        std::ofstream f(path, std::ios::binary);
        if (!f)
            return false;
        f.write(reinterpret_cast<const char *>(codebooks_.data()), static_cast<std::streamsize>(codebooks_.size() * sizeof(float)));
        return f.good();
    }

    bool ProductQuantizer::load_codebooks(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return false;

        f.seekg(0, std::ios::end);
        size_t bytes = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);

        size_t floats = bytes / sizeof(float);
        codebooks_.resize(floats);
        f.read(reinterpret_cast<char *>(codebooks_.data()), static_cast<std::streamsize>(bytes));
        return f.good();
    }

    void ProductQuantizer::compute_distance_tables(const float *query, float *out_tables) const
    {
        if (!query || !out_tables)
            return;

        size_t codebook_offset = 0;
        for (size_t sub = 0; sub < m_; ++sub)
        {
            size_t current_subdim = subdim_;
            if (sub == m_ - 1)
                current_subdim = dim_ - (subdim_ * (m_ - 1));

            size_t vec_offset = sub * subdim_;
            const float *centroids_base = &codebooks_[codebook_offset];

            for (size_t c = 0; c < k_; ++c)
            {
                const float *centroid = centroids_base + c * current_subdim;
                out_tables[sub * k_ + c] = l2sq(query + vec_offset, centroid, current_subdim);
            }
            codebook_offset += k_ * current_subdim;
        }
    }

    bool ProductQuantizer::load_codebooks_from_buffer(const float *src, size_t float_count)
    {
        if (!src)
            return false;
        codebooks_.assign(src, src + float_count);
        return true;
    }

} // namespace pomai::ai