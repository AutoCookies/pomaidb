#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai/pomai.h"
#include "pomai/status.h"

namespace pomai::core {

class SpatialEngine {
public:
    SpatialEngine(std::string path, std::size_t max_points);
    Status Open();
    Status Close();
    Status Put(std::uint64_t id, double lat, double lon);
    Status RadiusSearch(double lat, double lon, double radius_m, std::vector<pomai::SpatialPoint>* out) const;
    Status WithinPolygon(const pomai::GeoPolygon& polygon, std::vector<pomai::SpatialPoint>* out) const;
    Status Nearest(double lat, double lon, std::uint32_t topk, std::vector<pomai::SpatialPoint>* out) const;
    void ForEach(const std::function<void(std::uint64_t id, double lat, double lon)>& fn) const;

private:
    static bool PointInPolygon(double lat, double lon, const pomai::GeoPolygon& poly);
    std::string path_;
    std::size_t max_points_;
    std::unordered_map<std::uint64_t, pomai::SpatialPoint> points_;
};

} // namespace pomai::core

