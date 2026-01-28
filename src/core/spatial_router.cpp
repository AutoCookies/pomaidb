#include "spatial_router.h"
#include <cmath>
#include <numeric>

namespace pomai
{
    std::size_t SpatialRouter::PickShardForInsert(const Vector &vec) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        if (centroids_.empty())
            throw std::runtime_error("SpatialRouter: no centroids configured");

        const std::size_t C = centroids_.size();
        float best_d = std::numeric_limits<float>::infinity();
        std::size_t best = 0;

        for (std::size_t i = 0; i < C; ++i)
        {
            float d = kernels::L2Sqr(vec.data.data(), centroids_[i].data.data(), vec.data.size());
            if (d < best_d)
            {
                best_d = d;
                best = i;
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

        const std::size_t C = centroids_.size();
        struct Pair
        {
            float d;
            std::size_t idx;
        };
        std::vector<Pair> v;
        v.reserve(C);

        for (std::size_t i = 0; i < C; ++i)
        {
            float d = kernels::L2Sqr(q.data.data(), centroids_[i].data.data(), q.data.size());
            v.push_back({d, i});
        }

        std::vector<std::size_t> out;
        if (P >= C)
        {
            std::sort(v.begin(), v.end(), [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });
            for (auto &p : v)
                out.push_back(p.idx);
        }
        else
        {
            std::nth_element(v.begin(), v.begin() + P, v.end(), [](const Pair &a, const Pair &b)
                             { return a.d < b.d; });
            std::sort(v.begin(), v.begin() + P, [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });
            for (std::size_t i = 0; i < P; ++i)
                out.push_back(v[i].idx);
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