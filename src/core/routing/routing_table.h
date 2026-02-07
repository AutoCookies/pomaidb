#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace pomai::core::routing {

enum class RoutingMode : std::uint8_t {
    kDisabled = 0,
    kWarmup = 1,
    kReady = 2,
};

struct RoutingTable {
    std::uint64_t epoch = 0;
    std::uint32_t k = 0;
    std::uint32_t dim = 0;
    std::vector<float> centroids;          // size = k*dim
    std::vector<std::uint32_t> owner_shard; // size = k
    std::vector<std::uint64_t> counts;     // size = k

    bool Valid() const;
    std::uint32_t RouteVector(std::span<const float> vec) const;
    float DistanceSq(std::span<const float> vec, std::uint32_t centroid_id) const;
    std::vector<std::uint32_t> ClosestCentroids(std::span<const float> vec, std::uint32_t n) const;
};

using RoutingTablePtr = std::shared_ptr<const RoutingTable>;

} // namespace pomai::core::routing
