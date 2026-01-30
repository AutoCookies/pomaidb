#include <pomai/core/spatial_router.h>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

namespace pomai
{
    std::size_t SpatialRouter::SelectBestIndex(const Vector *centroids,
                                               const std::size_t *indices,
                                               std::size_t count,
                                               const float *query,
                                               std::size_t dim)
    {
        alignas(32) float distances[256];
        float best_d = std::numeric_limits<float>::infinity();
        std::size_t best = 0;

        std::size_t offset = 0;
        while (offset < count)
        {
            std::size_t batch = std::min<std::size_t>(count - offset, 256);
            kernels::L2Sqr8CentroidsAVX2(centroids, indices + offset, batch, query, dim, distances);
            for (std::size_t i = 0; i < batch; ++i)
            {
                if (distances[i] < best_d)
                {
                    best_d = distances[i];
                    best = offset + i;
                }
            }
            offset += batch;
        }
        return best;
    }

    std::size_t SpatialRouter::PickShardForInsert(const Vector &vec) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        if (centroids_.empty())
            throw std::runtime_error("SpatialRouter: empty");

        const std::size_t C = centroids_.size();
        std::size_t best = 0;

        if (C > kHierarchicalThreshold && !master_centroids_.empty())
        {
            std::vector<std::size_t> master_indices(num_master_);
            std::iota(master_indices.begin(), master_indices.end(), 0);
            std::size_t best_master = SelectBestIndex(master_centroids_.data(),
                                                      master_indices.data(),
                                                      num_master_,
                                                      vec.data.data(),
                                                      vec.data.size());

            const auto &leafs = master_to_leaf_[best_master];
            if (!leafs.empty())
            {
                best = leafs[SelectBestIndex(centroids_.data(),
                                             leafs.data(),
                                             leafs.size(),
                                             vec.data.data(),
                                             vec.data.size())];
            }
        }
        else
        {
            float best_d = std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < C; ++i)
            {
                float d = kernels::L2Sqr(vec.data.data(), centroids_[i].data.data(), vec.data.size());
                if (d < best_d)
                {
                    best_d = d;
                    best = i;
                }
            }
        }

        if (centroid_hits_)
            centroid_hits_[best].fetch_add(1, std::memory_order_relaxed);

        return best;
    }

    std::vector<std::size_t> SpatialRouter::CandidateShardsForQuery(const Vector &q, std::size_t P) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        if (centroids_.empty() || P == 0)
            return {};

        const std::size_t C = centroids_.size();
        struct Pair
        {
            float d;
            std::size_t idx;
        };
        std::vector<Pair> v;

        if (C > kHierarchicalThreshold && !master_centroids_.empty())
        {
            std::vector<std::size_t> master_indices(num_master_);
            std::iota(master_indices.begin(), master_indices.end(), 0);
            alignas(32) float master_distances[256];

            kernels::L2Sqr8CentroidsAVX2(master_centroids_.data(),
                                         master_indices.data(),
                                         num_master_,
                                         q.data.data(),
                                         q.data.size(),
                                         master_distances);

            float min_m_d = std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < num_master_; ++i)
                min_m_d = std::min(min_m_d, master_distances[i]);

            float m_radius = min_m_d * kProbeRadiusMultiplier + kProbeRadiusBias;
            std::vector<std::size_t> m_cands;
            for (std::size_t i = 0; i < num_master_; ++i)
            {
                if (master_distances[i] <= m_radius)
                    m_cands.push_back(i);
            }

            if (m_cands.empty())
                m_cands.push_back(SelectBestIndex(master_centroids_.data(), master_indices.data(), num_master_, q.data.data(), q.data.size()));

            for (auto midx : m_cands)
            {
                const auto &leafs = master_to_leaf_[midx];
                if (leafs.empty())
                    continue;

                std::vector<float> dists(leafs.size());
                kernels::L2Sqr8CentroidsAVX2(centroids_.data(),
                                             leafs.data(),
                                             leafs.size(),
                                             q.data.data(),
                                             q.data.size(),
                                             dists.data());
                for (std::size_t i = 0; i < leafs.size(); ++i)
                    v.push_back({dists[i], leafs[i]});
            }
        }
        else
        {
            v.reserve(C);
            for (std::size_t i = 0; i < C; ++i)
            {
                v.push_back({kernels::L2Sqr(q.data.data(), centroids_[i].data.data(), q.data.size()), i});
            }
        }

        if (v.empty())
            return {};

        float min_d = std::numeric_limits<float>::infinity();
        for (const auto &p : v)
            min_d = std::min(min_d, p.d);

        float radius = min_d * kProbeRadiusMultiplier + kProbeRadiusBias;
        std::vector<Pair> within;
        for (const auto &p : v)
        {
            if (p.d <= radius)
                within.push_back(p);
        }

        std::sort(within.begin(), within.end(), [](const Pair &a, const Pair &b)
                  { return a.d < b.d; });

        std::size_t cap = std::min<std::size_t>(C, std::max<std::size_t>(P, P * kProbeCapMultiplier));
        if (within.size() > cap)
            within.resize(cap);

        std::vector<std::size_t> out;
        for (const auto &p : within)
            out.push_back(p.idx);

        if (out.size() < P)
        {
            std::sort(v.begin(), v.end(), [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });
            for (std::size_t i = 0; i < std::min(P, v.size()); ++i)
            {
                if (std::find(out.begin(), out.end(), v[i].idx) == out.end())
                    out.push_back(v[i].idx);
            }
        }

        if (centroid_hits_)
        {
            for (auto idx : out)
                centroid_hits_[idx].fetch_add(1, std::memory_order_relaxed);
        }

        return out;
    }

    void SpatialRouter::ReplaceCentroids(std::vector<Vector> new_centroids)
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        const std::size_t C = new_centroids.size();
        centroids_.swap(new_centroids);
        num_centroids_ = C;
        master_centroids_.clear();
        master_to_leaf_.clear();

        if (C > kHierarchicalThreshold)
        {
            num_master_ = std::min<std::size_t>(64, C / 8 + 1);
            master_centroids_ = BuildKMeans(centroids_, num_master_, 8);
            master_to_leaf_.assign(num_master_, {});

            std::vector<std::size_t> m_indices(num_master_);
            std::iota(m_indices.begin(), m_indices.end(), 0);

            for (std::size_t i = 0; i < C; ++i)
            {
                master_to_leaf_[SelectBestIndex(master_centroids_.data(), m_indices.data(), num_master_, centroids_[i].data.data(), centroids_[i].data.size())].push_back(i);
            }
        }

        if (C > 0)
        {
            centroid_hits_ = std::make_unique<std::atomic<std::uint64_t>[]>(C);
            for (std::size_t i = 0; i < C; ++i)
                centroid_hits_[i].store(0, std::memory_order_relaxed);
        }
        else
        {
            centroid_hits_.reset();
        }
    }

    std::vector<Vector> SpatialRouter::BuildKMeans(const std::vector<Vector> &data, std::size_t k, int iterations)
    {
        if (data.empty() || k == 0)
            throw std::runtime_error("KMeans: invalid input");
        const std::size_t dim = data[0].data.size();
        const std::size_t N = data.size();
        if (k > N)
            k = N;

        std::vector<Vector> centroids;
        std::mt19937_64 rng(std::random_device{}());

        std::uniform_int_distribution<std::size_t> dist(0, N - 1);
        centroids.push_back(data[dist(rng)]);

        for (std::size_t i = 1; i < k; ++i)
        {
            std::vector<double> min_dists(N, std::numeric_limits<double>::max());
            double total_d = 0;
            for (std::size_t j = 0; j < N; ++j)
            {
                for (const auto &c : centroids)
                {
                    min_dists[j] = std::min<double>(min_dists[j], kernels::L2Sqr(data[j].data.data(), c.data.data(), dim));
                }
                total_d += min_dists[j];
            }

            std::uniform_real_distribution<double> pick(0, total_d);
            double target = pick(rng);
            double cumulative = 0;
            for (std::size_t j = 0; j < N; ++j)
            {
                cumulative += min_dists[j];
                if (cumulative >= target)
                {
                    centroids.push_back(data[j]);
                    break;
                }
            }
            if (centroids.size() <= i)
                centroids.push_back(data[dist(rng)]);
        }

        for (int it = 0; it < iterations; ++it)
        {
            std::vector<std::vector<double>> acc(k, std::vector<double>(dim, 0.0));
            std::vector<std::size_t> counts(k, 0);

            for (std::size_t i = 0; i < N; ++i)
            {
                float best_d = std::numeric_limits<float>::infinity();
                std::size_t best = 0;
                for (std::size_t c = 0; c < k; ++c)
                {
                    float d = kernels::L2Sqr(data[i].data.data(), centroids[c].data.data(), dim);
                    if (d < best_d)
                    {
                        best_d = d;
                        best = c;
                    }
                }
                counts[best]++;
                for (std::size_t d = 0; d < dim; ++d)
                    acc[best][d] += data[i].data[d];
            }

            for (std::size_t c = 0; c < k; ++c)
            {
                if (counts[c] == 0)
                    continue;
                for (std::size_t d = 0; d < dim; ++d)
                    centroids[c].data[d] = static_cast<float>(acc[c][d] / counts[c]);
            }
        }
        return centroids;
    }

    std::vector<Vector> SpatialRouter::SnapshotCentroids() const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        return centroids_;
    }

    std::optional<SpatialRouter::HotspotInfo> SpatialRouter::DetectHotspot(double threshold_ratio) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        if (num_centroids_ == 0 || !centroid_hits_)
            return std::nullopt;

        std::uint64_t total = 0;
        std::size_t b_idx = 0;
        std::uint64_t b_hits = 0;

        for (std::size_t i = 0; i < num_centroids_; ++i)
        {
            std::uint64_t h = centroid_hits_[i].load(std::memory_order_relaxed);
            total += h;
            if (h > b_hits)
            {
                b_hits = h;
                b_idx = i;
            }
        }

        if (total == 0)
            return std::nullopt;
        double avg = static_cast<double>(total) / num_centroids_;
        double r = (avg > 0) ? (static_cast<double>(b_hits) / avg) : 0;

        if (r < threshold_ratio)
            return std::nullopt;
        return HotspotInfo{b_idx, r, static_cast<std::size_t>(total), avg};
    }
}
