#include "spatial_router.h"
#include <cmath>
#include <numeric>

namespace pomai
{
    namespace
    {
        void L2Sqr8CentroidsAVX2(const Vector *centroids,
                                 const std::size_t *indices,
                                 std::size_t count,
                                 const float *query,
                                 std::size_t dim,
                                 float *out)
        {
            std::size_t i = 0;
            for (; i + 8 <= count; i += 8)
            {
                __m256 sum = _mm256_setzero_ps();
                for (std::size_t d = 0; d < dim; ++d)
                {
                    float qv = query[d];
                    __m256 c = _mm256_set_ps(
                        centroids[indices[i + 7]].data[d],
                        centroids[indices[i + 6]].data[d],
                        centroids[indices[i + 5]].data[d],
                        centroids[indices[i + 4]].data[d],
                        centroids[indices[i + 3]].data[d],
                        centroids[indices[i + 2]].data[d],
                        centroids[indices[i + 1]].data[d],
                        centroids[indices[i + 0]].data[d]);
                    __m256 q = _mm256_set1_ps(qv);
                    __m256 diff = _mm256_sub_ps(c, q);
                    sum = _mm256_fmadd_ps(diff, diff, sum);
                }
                _mm256_storeu_ps(out + i, sum);
            }

            for (; i < count; ++i)
            {
                const auto &v = centroids[indices[i]];
                out[i] = kernels::L2Sqr(query, v.data.data(), dim);
            }
        }

        std::size_t SelectBestIndex(const Vector *centroids,
                                    const std::size_t *indices,
                                    std::size_t count,
                                    const float *query,
                                    std::size_t dim)
        {
            std::vector<float> distances(count, 0.0f);
            L2Sqr8CentroidsAVX2(centroids, indices, count, query, dim, distances.data());
            float best_d = std::numeric_limits<float>::infinity();
            std::size_t best = 0;
            for (std::size_t i = 0; i < count; ++i)
            {
                if (distances[i] < best_d)
                {
                    best_d = distances[i];
                    best = i;
                }
            }
            return best;
        }
    }

    std::size_t SpatialRouter::PickShardForInsert(const Vector &vec) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        if (centroids_.empty())
            throw std::runtime_error("SpatialRouter: no centroids configured");

        const std::size_t C = centroids_.size();
        std::size_t best = 0;

        if (C > kHierarchicalThreshold && !master_centroids_.empty() && !master_to_leaf_.empty())
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

        // Tăng bộ đếm hotspot với memory_order_relaxed để đạt hiệu suất tối đa
        if (centroid_hits_)
            centroid_hits_[best].fetch_add(1, std::memory_order_relaxed);

        return best;
    }

    std::vector<std::size_t> SpatialRouter::CandidateShardsForQuery(const Vector &q, std::size_t P) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        if (centroids_.empty())
            return {};
        if (P == 0)
            return {};

        const std::size_t C = centroids_.size();
        struct Pair
        {
            float d;
            std::size_t idx;
        };
        std::vector<Pair> v;

        if (C > kHierarchicalThreshold && !master_centroids_.empty() && !master_to_leaf_.empty())
        {
            std::vector<std::size_t> master_indices(num_master_);
            std::iota(master_indices.begin(), master_indices.end(), 0);
            std::vector<float> master_distances(num_master_, 0.0f);
            L2Sqr8CentroidsAVX2(master_centroids_.data(),
                               master_indices.data(),
                               num_master_,
                               q.data.data(),
                               q.data.size(),
                               master_distances.data());

            float best_master_d = std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < num_master_; ++i)
                best_master_d = std::min(best_master_d, master_distances[i]);

            const float master_radius = best_master_d * kProbeRadiusMultiplier + kProbeRadiusBias;
            std::vector<std::size_t> master_candidates;
            master_candidates.reserve(num_master_);
            for (std::size_t i = 0; i < num_master_; ++i)
            {
                if (master_distances[i] <= master_radius)
                    master_candidates.push_back(i);
            }
            if (master_candidates.empty())
            {
                std::size_t best_master = SelectBestIndex(master_centroids_.data(),
                                                          master_indices.data(),
                                                          num_master_,
                                                          q.data.data(),
                                                          q.data.size());
                master_candidates.push_back(best_master);
            }

            std::size_t total_leafs = 0;
            for (auto midx : master_candidates)
                total_leafs += master_to_leaf_[midx].size();
            v.reserve(total_leafs);
            for (auto midx : master_candidates)
            {
                const auto &leafs = master_to_leaf_[midx];
                if (leafs.empty())
                    continue;
                std::vector<float> distances(leafs.size(), 0.0f);
                L2Sqr8CentroidsAVX2(centroids_.data(),
                                   leafs.data(),
                                   leafs.size(),
                                   q.data.data(),
                                   q.data.size(),
                                   distances.data());
                for (std::size_t i = 0; i < leafs.size(); ++i)
                    v.push_back({distances[i], leafs[i]});
            }
        }
        else
        {
            v.reserve(C);
            for (std::size_t i = 0; i < C; ++i)
            {
                float d = kernels::L2Sqr(q.data.data(), centroids_[i].data.data(), q.data.size());
                v.push_back({d, i});
            }
        }

        std::vector<std::size_t> out;
        if (v.empty())
            return out;

        float best_d = std::numeric_limits<float>::infinity();
        for (const auto &p : v)
            best_d = std::min(best_d, p.d);

        const float radius = best_d * kProbeRadiusMultiplier + kProbeRadiusBias;
        std::vector<Pair> within_radius;
        within_radius.reserve(v.size());
        for (const auto &p : v)
        {
            if (p.d <= radius)
                within_radius.push_back(p);
        }

        const std::size_t cap = std::min<std::size_t>(C, std::max<std::size_t>(P, P * kProbeCapMultiplier));
        if (!within_radius.empty())
        {
            std::sort(within_radius.begin(), within_radius.end(), [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });
            if (within_radius.size() > cap)
                within_radius.resize(cap);
            out.reserve(within_radius.size());
            for (const auto &p : within_radius)
                out.push_back(p.idx);
        }

        if (out.size() < P)
        {
            std::nth_element(v.begin(), v.begin() + std::min<std::size_t>(P, v.size() - 1), v.end(),
                             [](const Pair &a, const Pair &b)
                             { return a.d < b.d; });
            std::size_t take = std::min<std::size_t>(P, v.size());
            std::sort(v.begin(), v.begin() + take, [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });
            for (std::size_t i = 0; i < take; ++i)
            {
                if (std::find(out.begin(), out.end(), v[i].idx) == out.end())
                    out.push_back(v[i].idx);
            }
        }

        // Cập nhật bộ đếm cho các shard tiềm năng
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
        num_master_ = 0;

        if (C > kHierarchicalThreshold)
        {
            num_master_ = static_cast<std::size_t>(std::max<std::size_t>(1, std::sqrt(static_cast<double>(C))));
            master_centroids_ = BuildKMeans(centroids_, num_master_, 8);
            master_to_leaf_.assign(num_master_, {});
            for (std::size_t i = 0; i < C; ++i)
            {
                float best_d = std::numeric_limits<float>::infinity();
                std::size_t best = 0;
                for (std::size_t m = 0; m < num_master_; ++m)
                {
                    float d = kernels::L2Sqr(centroids_[i].data.data(),
                                            master_centroids_[m].data.data(),
                                            centroids_[i].data.size());
                    if (d < best_d)
                    {
                        best_d = d;
                        best = m;
                    }
                }
                master_to_leaf_[best].push_back(i);
            }
        }

        if (C > 0)
        {
            // Cấp phát mảng nguyên tử mới, tránh việc copy/move gây lỗi
            centroid_hits_ = std::make_unique<std::atomic<std::uint64_t>[]>(C);
            for (std::size_t i = 0; i < C; ++i)
                centroid_hits_[i].store(0, std::memory_order_relaxed);
        }
        else
        {
            centroid_hits_.reset();
        }
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

        std::uint64_t total_hits = 0;
        std::size_t best_idx = 0;
        std::uint64_t best_hits = 0;

        for (std::size_t i = 0; i < num_centroids_; ++i)
        {
            std::uint64_t hits = centroid_hits_[i].load(std::memory_order_relaxed);
            total_hits += hits;
            if (hits > best_hits)
            {
                best_hits = hits;
                best_idx = i;
            }
        }

        if (total_hits == 0)
            return std::nullopt;

        double avg = static_cast<double>(total_hits) / num_centroids_;
        double ratio = (avg > 0) ? (static_cast<double>(best_hits) / avg) : 0;

        if (ratio < threshold_ratio)
            return std::nullopt;

        return HotspotInfo{best_idx, ratio, static_cast<std::size_t>(total_hits), avg};
    }

    // K-Means giữ nguyên logic xử lý dữ liệu của bạn nhưng tối ưu hóa việc di chuyển vector
    std::vector<Vector> SpatialRouter::BuildKMeans(const std::vector<Vector> &data, std::size_t k, int iterations)
    {
        if (data.empty() || k == 0)
            throw std::runtime_error("BuildKMeans: Invalid input");

        const std::size_t dim = data[0].data.size();
        const std::size_t N = data.size();
        if (k > N)
            throw std::runtime_error("BuildKMeans: k > N");

        std::vector<Vector> centroids;
        std::mt19937_64 rng(std::random_device{}());
        std::vector<std::size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (std::size_t i = 0; i < k; ++i)
            centroids.push_back(data[indices[i]]);

        std::vector<std::size_t> assignments(N);
        for (int it = 0; it < iterations; ++it)
        {
            std::vector<std::vector<double>> acc(k, std::vector<double>(dim, 0.0));
            std::vector<std::size_t> counts(k, 0);

            for (std::size_t i = 0; i < N; ++i)
            {
                float bestd = std::numeric_limits<float>::infinity();
                std::size_t best = 0;
                for (std::size_t c = 0; c < k; ++c)
                {
                    float d = kernels::L2Sqr(data[i].data.data(), centroids[c].data.data(), dim);
                    if (d < bestd)
                    {
                        bestd = d;
                        best = c;
                    }
                }
                assignments[i] = best;
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
}
