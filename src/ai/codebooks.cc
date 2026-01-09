/*
 * src/ai/codebooks.cc
 *
 * Implementation of Codebooks helpers.
 *
 * compute_distance_tables computes squared L2 distances sub-vector wise.
 * The implementation is written for clarity and correctness; it is easy to
 * optimize with blocking / SIMD later.
 */

#include "src/ai/codebooks.h"

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cassert>

namespace pomai::ai
{

    Codebooks::Codebooks(size_t dim, size_t m, size_t k)
        : dim_(dim), m_(m), k_(k)
    {
        if (dim_ == 0 || m_ == 0 || k_ == 0)
            throw std::invalid_argument("Codebooks: dim/m/k must be > 0");

        subdim_ = dim_ / m_;
        if (subdim_ == 0)
            subdim_ = 1; // fallback, last sub may be larger

        codebooks_.assign(m_ * k_ * subdim_, 0.0f);
    }

    Codebooks::Codebooks(size_t dim, size_t m, size_t k, const std::vector<float> &raw)
        : dim_(dim), m_(m), k_(k), codebooks_(raw)
    {
        if (dim_ == 0 || m_ == 0 || k_ == 0)
            throw std::invalid_argument("Codebooks: dim/m/k must be > 0");
        subdim_ = dim_ / m_;
        if (subdim_ == 0)
            subdim_ = 1;
        if (codebooks_.size() != m_ * k_ * subdim_)
            throw std::invalid_argument("Codebooks: raw vector size mismatch");
    }

    void Codebooks::set_codebooks_from_raw(size_t dim, size_t m, size_t k, const std::vector<float> &raw)
    {
        dim_ = dim;
        m_ = m;
        k_ = k;
        subdim_ = dim_ / m_;
        if (subdim_ == 0)
            subdim_ = 1;
        if (raw.size() != m_ * k_ * subdim_)
            throw std::invalid_argument("Codebooks::set_codebooks_from_raw: raw size mismatch");
        codebooks_ = raw;
    }

    bool Codebooks::load_from_file(const std::string &path)
    {
        // Reuse ProductQuantizer binary layout: header then raw floats.
        // The load routine must match ProductQuantizer::save_codebooks format.
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

        size_t dim = static_cast<size_t>(dim64);
        size_t m = static_cast<size_t>(m64);
        size_t k = static_cast<size_t>(k64);
        size_t subdim = static_cast<size_t>(subdim64);

        // read raw floats
        size_t expected = m * k * subdim;
        std::vector<float> buf(expected);
        f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(expected * sizeof(float)));
        if (!f)
            return false;

        // populate
        dim_ = dim;
        m_ = m;
        k_ = k;
        subdim_ = subdim;
        codebooks_.swap(buf);
        return true;
    }

    bool Codebooks::save_to_file(const std::string &path) const
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f)
            return false;
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
        size_t expected = m_ * k_ * subdim_;
        f.write(reinterpret_cast<const char *>(codebooks_.data()), static_cast<std::streamsize>(expected * sizeof(float)));
        return f.good();
    }

    void Codebooks::compute_distance_tables(const float *query, float *out_tables) const
    {
        if (!query || !out_tables)
            return;
        // For each subquantizer
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

            // pointer to first centroid for this sub
            const float *centroids_base = &codebooks_[sub * (k_ * subdim_)];

            for (size_t c = 0; c < k_; ++c)
            {
                const float *centroid = centroids_base + c * subdim_;
                double sum = 0.0;
                for (size_t d = 0; d < sub_len; ++d)
                {
                    double diff = static_cast<double>(query[sub_off + d]) - static_cast<double>(centroid[d]);
                    sum += diff * diff;
                }
                out_tables[sub * k_ + c] = static_cast<float>(sum);
            }
        }
    }

} // namespace pomai::ai