#include "core/routing/routing_table.h"

#include <algorithm>
#include <limits>

namespace pomai::core::routing {

namespace {
float L2Sq(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
} // namespace

bool RoutingTable::Valid() const {
    return k > 0 && dim > 0 && centroids.size() == static_cast<std::size_t>(k) * dim &&
           owner_shard.size() == k && counts.size() == k;
}

float RoutingTable::DistanceSq(std::span<const float> vec, std::uint32_t centroid_id) const {
    const std::size_t base = static_cast<std::size_t>(centroid_id) * dim;
    return L2Sq(vec, std::span<const float>(centroids.data() + base, dim));
}

std::uint32_t RoutingTable::RouteVector(std::span<const float> vec) const {
    std::uint32_t best = 0;
    float best_d = std::numeric_limits<float>::max();
    for (std::uint32_t i = 0; i < k; ++i) {
        const float d = DistanceSq(vec, i);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return owner_shard[best];
}

std::vector<std::uint32_t> RoutingTable::ClosestCentroids(std::span<const float> vec, std::uint32_t n) const {
    const std::uint32_t take = std::min(k, n);
    std::vector<std::pair<float, std::uint32_t>> all;
    all.reserve(k);
    for (std::uint32_t i = 0; i < k; ++i) {
        all.push_back({DistanceSq(vec, i), i});
    }
    std::partial_sort(all.begin(), all.begin() + take, all.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<std::uint32_t> out;
    out.reserve(take);
    for (std::uint32_t i = 0; i < take; ++i) {
        out.push_back(all[i].second);
    }
    return out;
}

} // namespace pomai::core::routing
