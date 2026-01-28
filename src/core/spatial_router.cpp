#include "spatial_router.h"

#include <cmath>
#include <iostream>
#include <random>
#include <limits>
#include <atomic>
#include <algorithm>
#include <numeric> // for std::iota

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
        return best;
    }

    std::vector<std::size_t> SpatialRouter::CandidateShardsForQuery(const Vector &q, std::size_t P) const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        std::vector<std::size_t> out;
        if (centroids_.empty())
            return out;

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
            v.push_back(Pair{d, i});
        }

        // If P >= C, return all centroids sorted.
        if (P >= C)
        {
            std::sort(v.begin(), v.end(), [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });
            out.reserve(C);
            for (auto &p : v)
                out.push_back(p.idx);
            return out;
        }

        // partial sort: get top-P without fully sorting
        std::nth_element(v.begin(), v.begin() + P, v.end(), [](const Pair &a, const Pair &b)
                         { return a.d < b.d; });
        std::sort(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(P), [](const Pair &a, const Pair &b)
                  { return a.d < b.d; });

        out.reserve(P);
        for (size_t i = 0; i < P; ++i)
            out.push_back(v[i].idx);

        return out;
    }

    void SpatialRouter::ReplaceCentroids(std::vector<Vector> new_centroids)
    {
        std::unique_lock<std::shared_mutex> lk(mu_);
        centroids_.swap(new_centroids);
    }

    std::vector<Vector> SpatialRouter::SnapshotCentroids() const
    {
        std::shared_lock<std::shared_mutex> lk(mu_);
        return centroids_;
    }

    // -------------------- Simple Lloyd's k-means --------------------
    //
    // A lightweight k-means implementation for offline centroid computation.
    // Not optimized for huge data; intended for sampling-based centroid computation.
    //
    // Data assumptions:
    // - All vectors in 'data' share the same dimension
    //
    std::vector<Vector> SpatialRouter::BuildKMeans(const std::vector<Vector> &data, std::size_t k, int iterations)
    {
        if (data.empty())
            throw std::runtime_error("BuildKMeans: empty input data");
        if (k == 0)
            throw std::runtime_error("BuildKMeans: k must be > 0");
        const std::size_t dim = data[0].data.size();

        // Validate dims
        for (const auto &v : data)
        {
            if (v.data.size() != dim)
                throw std::runtime_error("BuildKMeans: inconsistent vector dimension in sample");
        }

        const std::size_t N = data.size();
        if (k > N)
            throw std::runtime_error("BuildKMeans: k cannot be greater than number of samples");

        std::vector<Vector> centroids;
        centroids.reserve(k);

        // Initialize centroids by selecting k distinct random samples using a partial shuffle.
        std::mt19937_64 rng(std::random_device{}());
        std::vector<std::size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);

        // Fisher-Yates style: shuffle first k elements (partial shuffle)
        for (std::size_t i = 0; i < k; ++i)
        {
            std::uniform_int_distribution<std::size_t> dist(i, N - 1);
            std::size_t j = dist(rng);
            std::swap(indices[i], indices[j]);
            centroids.push_back(data[indices[i]]); // copy initial centroid
        }

        std::vector<std::size_t> assignments(N, 0);
        std::vector<double> counts(k, 0.0);
        std::vector<std::vector<double>> acc(k, std::vector<double>(dim, 0.0));

        for (int it = 0; it < iterations; ++it)
        {
            // assignment step
            for (std::size_t i = 0; i < N; ++i)
            {
                float bestd = std::numeric_limits<float>::infinity();
                std::size_t best = 0;
                const auto &v = data[i];
                for (std::size_t c = 0; c < k; ++c)
                {
                    float d = kernels::L2Sqr(v.data.data(), centroids[c].data.data(), dim);
                    if (d < bestd)
                    {
                        bestd = d;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // reset accumulators
            for (std::size_t c = 0; c < k; ++c)
            {
                counts[c] = 0.0;
                std::fill(acc[c].begin(), acc[c].end(), 0.0);
            }

            // accumulate
            for (std::size_t i = 0; i < N; ++i)
            {
                std::size_t c = assignments[i];
                counts[c] += 1.0;
                const auto &src = data[i].data;
                auto &dst = acc[c];
                for (std::size_t d = 0; d < dim; ++d)
                    dst[d] += static_cast<double>(src[d]);
            }

            // update centroids
            for (std::size_t c = 0; c < k; ++c)
            {
                if (counts[c] == 0.0)
                {
                    // reinitialize empty cluster to a random point
                    std::uniform_int_distribution<std::size_t> dist(0, N - 1);
                    centroids[c] = data[dist(rng)];
                }
                else
                {
                    Vector cv;
                    cv.data.resize(dim);
                    double inv = 1.0 / counts[c];
                    for (std::size_t d = 0; d < dim; ++d)
                        cv.data[d] = static_cast<float>(acc[c][d] * inv);
                    centroids[c] = std::move(cv);
                }
            }
        }

        return centroids;
    }

} // namespace pomai