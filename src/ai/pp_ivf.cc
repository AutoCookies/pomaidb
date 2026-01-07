#include "src/ai/pp_ivf.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>

namespace pomai::ai
{

    PPIVF::PPIVF(size_t dim, size_t k_clusters, size_t m_sub, size_t nbits)
        : dim_(dim), k_(std::max<size_t>(1, k_clusters)), m_sub_(std::max<size_t>(1, m_sub)), nbits_(nbits)
    {
        centroids_.assign(k_ * dim_, 0.0f);
        code_scratch_.resize(m_sub_);
        rng_.seed(123456789);
    }

    PPIVF::~PPIVF() = default;

    bool PPIVF::init_random_seed(uint64_t seed)
    {
        rng_.seed(seed);
        // initialize centroids with random values in [0,1)
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        for (size_t i = 0; i < k_ * dim_; ++i)
            centroids_[i] = ud(rng_);
        return true;
    }

    inline float PPIVF::l2sq_distance(const float *a, const float *b, size_t dim)
    {
        double sum = 0.0;
        for (size_t i = 0; i < dim; ++i)
        {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += d * d;
        }
        return static_cast<float>(sum);
    }

    int PPIVF::assign_cluster(const float *vec) const
    {
        // linear scan centroids (MVP). Complexity O(k*dim) — replace with ANN for production.
        float best = std::numeric_limits<float>::infinity();
        int best_i = 0;
        for (size_t c = 0; c < k_; ++c)
        {
            const float *cent = centroids_.data() + c * dim_;
            float d = l2sq_distance(vec, cent, dim_);
            if (d < best)
            {
                best = d;
                best_i = static_cast<int>(c);
            }
        }
        return best_i;
    }

    const uint8_t *PPIVF::encode_pq(const float *vec)
    {
        // Simple uniform scalar quant per subspace. For nbits==8 produce 0..255 values.
        // Split dim into m_sub_ equal chunks (last chunk may be larger).
        size_t chunk = dim_ / m_sub_;
        if (chunk == 0)
            return nullptr;
        for (size_t s = 0; s < m_sub_; ++s)
        {
            size_t off = s * chunk;
            size_t len = (s + 1 == m_sub_) ? (dim_ - off) : chunk;
            float lo = std::numeric_limits<float>::infinity();
            float hi = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < len; ++i)
            {
                float v = vec[off + i];
                if (v < lo)
                    lo = v;
                if (v > hi)
                    hi = v;
            }
            if (hi <= lo)
            {
                code_scratch_[s] = 0;
                continue;
            }
            // map mean to [0,255]
            float mean = 0.0f;
            for (size_t i = 0; i < len; ++i)
                mean += vec[off + i];
            mean /= static_cast<float>(len);
            float norm = (mean - lo) / (hi - lo);
            uint8_t q = static_cast<uint8_t>(std::lround(std::min(1.0f, std::max(0.0f, norm)) * 255.0f));
            code_scratch_[s] = q;
        }
        return code_scratch_.data();
    }

    void PPIVF::add_label(size_t label, int cluster, const uint8_t *code)
    {
        std::lock_guard<std::mutex> lk(mu_);
        label_to_cluster_[label] = cluster;
        if (code)
            label_to_code_[label] = std::vector<uint8_t>(code, code + m_sub_);
        else
            label_to_code_.erase(label);
    }

    const uint8_t *PPIVF::get_code_for_label(size_t label) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = label_to_code_.find(label);
        if (it == label_to_code_.end())
            return nullptr;
        return it->second.data();
    }

    int PPIVF::get_cluster_for_label(size_t label) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = label_to_cluster_.find(label);
        if (it == label_to_cluster_.end())
            return -1;
        return it->second;
    }

    std::vector<int> PPIVF::probe_clusters(const float *query, size_t probe_k) const
    {
        // linear scan cluster centroids to find nearest probe_k clusters (MVP)
        std::vector<std::pair<float, int>> dist_idx;
        dist_idx.reserve(k_);
        for (size_t c = 0; c < k_; ++c)
        {
            const float *cent = centroids_.data() + c * dim_;
            float d = l2sq_distance(query, cent, dim_);
            dist_idx.emplace_back(d, static_cast<int>(c));
        }
        std::nth_element(dist_idx.begin(), dist_idx.begin() + std::min<size_t>(probe_k, dist_idx.size() - 1), dist_idx.end());
        size_t take = std::min<size_t>(probe_k, dist_idx.size());
        std::vector<int> out;
        out.reserve(take);
        std::sort(dist_idx.begin(), dist_idx.begin() + take);
        for (size_t i = 0; i < take; ++i)
            out.push_back(dist_idx[i].second);
        return out;
    }

} // namespace pomai::ai