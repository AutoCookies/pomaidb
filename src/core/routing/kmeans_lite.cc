#include "core/routing/kmeans_lite.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace pomai::core::routing {

namespace {
float DistSq(std::span<const float> a, std::span<const float> b) {
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
} // namespace

RoutingTable BuildInitialTable(std::span<const float> samples,
                               std::uint32_t sample_count,
                               std::uint32_t dim,
                               std::uint32_t k,
                               std::uint32_t shard_count,
                               std::uint32_t lloyd_iters,
                               std::uint64_t seed) {
    RoutingTable t;
    t.epoch = 1;
    t.k = std::max(1u, std::min(k, sample_count));
    t.dim = dim;
    t.centroids.resize(static_cast<std::size_t>(t.k) * dim, 0.0f);
    t.owner_shard.resize(t.k, 0);
    t.counts.resize(t.k, 1);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint32_t> uni(0, sample_count - 1);

    auto sample_span = [&](std::uint32_t idx) {
        const std::size_t base = static_cast<std::size_t>(idx) * dim;
        return std::span<const float>(samples.data() + base, dim);
    };

    std::vector<std::uint32_t> chosen;
    chosen.reserve(t.k);
    chosen.push_back(uni(rng));

    for (std::uint32_t c = 1; c < t.k; ++c) {
        std::vector<double> d2(sample_count, 0.0);
        double total = 0.0;
        for (std::uint32_t i = 0; i < sample_count; ++i) {
            float best = std::numeric_limits<float>::max();
            auto v = sample_span(i);
            for (std::uint32_t p : chosen) {
                const float d = DistSq(v, sample_span(p));
                if (d < best) best = d;
            }
            d2[i] = static_cast<double>(best);
            total += d2[i];
        }
        if (total <= 0.0) {
            chosen.push_back(uni(rng));
            continue;
        }
        std::uniform_real_distribution<double> pick(0.0, total);
        double r = pick(rng);
        std::uint32_t next = 0;
        for (std::uint32_t i = 0; i < sample_count; ++i) {
            r -= d2[i];
            if (r <= 0.0) {
                next = i;
                break;
            }
        }
        chosen.push_back(next);
    }

    for (std::uint32_t c = 0; c < t.k; ++c) {
        const auto v = sample_span(chosen[c]);
        std::copy(v.begin(), v.end(), t.centroids.begin() + static_cast<std::size_t>(c) * dim);
    }

    std::vector<std::uint32_t> assign(sample_count, 0);
    for (std::uint32_t iter = 0; iter < lloyd_iters; ++iter) {
        std::vector<double> sums(static_cast<std::size_t>(t.k) * dim, 0.0);
        std::vector<std::uint64_t> cnt(t.k, 0);

        for (std::uint32_t i = 0; i < sample_count; ++i) {
            auto v = sample_span(i);
            std::uint32_t best_id = 0;
            float best_d = std::numeric_limits<float>::max();
            for (std::uint32_t c = 0; c < t.k; ++c) {
                const std::size_t base = static_cast<std::size_t>(c) * dim;
                const float d = DistSq(v, std::span<const float>(t.centroids.data() + base, dim));
                if (d < best_d) {
                    best_d = d;
                    best_id = c;
                }
            }
            assign[i] = best_id;
            cnt[best_id]++;
            const std::size_t base = static_cast<std::size_t>(best_id) * dim;
            for (std::uint32_t d = 0; d < dim; ++d) sums[base + d] += static_cast<double>(v[d]);
        }

        for (std::uint32_t c = 0; c < t.k; ++c) {
            if (cnt[c] == 0) continue;
            const std::size_t base = static_cast<std::size_t>(c) * dim;
            for (std::uint32_t d = 0; d < dim; ++d) {
                t.centroids[base + d] = static_cast<float>(sums[base + d] / static_cast<double>(cnt[c]));
            }
            t.counts[c] = cnt[c];
        }
    }

    for (std::uint32_t c = 0; c < t.k; ++c) {
        t.owner_shard[c] = shard_count == 0 ? 0 : (c % shard_count);
        if (t.counts[c] == 0) t.counts[c] = 1;
    }
    return t;
}

void OnlineUpdate(RoutingTable* table, std::span<const float> vec) {
    if (!table || !table->Valid()) return;
    std::uint32_t best_id = 0;
    float best_d = std::numeric_limits<float>::max();
    for (std::uint32_t c = 0; c < table->k; ++c) {
        const float d = table->DistanceSq(vec, c);
        if (d < best_d) {
            best_d = d;
            best_id = c;
        }
    }
    const std::size_t base = static_cast<std::size_t>(best_id) * table->dim;
    std::uint64_t next = table->counts[best_id] + 1;
    table->counts[best_id] = next;
    const float inv = 1.0f / static_cast<float>(next);
    for (std::uint32_t d = 0; d < table->dim; ++d) {
        table->centroids[base + d] += (vec[d] - table->centroids[base + d]) * inv;
    }
}

} // namespace pomai::core::routing
