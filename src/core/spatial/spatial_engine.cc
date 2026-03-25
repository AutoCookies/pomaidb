#include "core/spatial/spatial_engine.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "core/simd/simd_dispatch.h"

namespace pomai::core {

SpatialEngine::SpatialEngine(std::string path, std::size_t max_points)
    : path_(std::move(path)), max_points_(max_points) {}

Status SpatialEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("spatial create dir failed");
    std::ifstream in(path_ + "/spatial.log", std::ios::binary);
    if (!in.good()) return Status::Ok();
    std::uint64_t id = 0;
    double lat = 0, lon = 0;
    while (in.read(reinterpret_cast<char*>(&id), sizeof(id))) {
        if (!in.read(reinterpret_cast<char*>(&lat), sizeof(lat))) break;
        if (!in.read(reinterpret_cast<char*>(&lon), sizeof(lon))) break;
        points_[id] = pomai::SpatialPoint{id, lat, lon};
        if (max_points_ > 0 && points_.size() > max_points_) points_.erase(points_.begin());
    }
    return Status::Ok();
}

Status SpatialEngine::Close() { return Status::Ok(); }

Status SpatialEngine::Put(std::uint64_t id, double lat, double lon) {
    if (max_points_ > 0 && points_.size() >= max_points_ && points_.find(id) == points_.end()) {
        points_.erase(points_.begin());
    }
    points_[id] = pomai::SpatialPoint{id, lat, lon};
    std::ofstream out(path_ + "/spatial.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("spatial append failed");
    out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    out.write(reinterpret_cast<const char*>(&lat), sizeof(lat));
    out.write(reinterpret_cast<const char*>(&lon), sizeof(lon));
    return Status::Ok();
}

Status SpatialEngine::RadiusSearch(double lat, double lon, double radius_m, std::vector<pomai::SpatialPoint>* out) const {
    if (!out) return Status::InvalidArgument("spatial out null");
    out->clear();
    for (const auto& kv : points_) {
        double d = simd::HaversineMeters(lat, lon, kv.second.latitude, kv.second.longitude);
        if (d <= radius_m) out->push_back(kv.second);
    }
    return Status::Ok();
}

bool SpatialEngine::PointInPolygon(double lat, double lon, const pomai::GeoPolygon& poly) {
    if (poly.vertices.size() < 3) return false;
    bool inside = false;
    std::size_t j = poly.vertices.size() - 1;
    for (std::size_t i = 0; i < poly.vertices.size(); j = i++) {
        const auto& pi = poly.vertices[i];
        const auto& pj = poly.vertices[j];
        const bool intersect = ((pi.longitude > lon) != (pj.longitude > lon)) &&
            (lat < (pj.latitude - pi.latitude) * (lon - pi.longitude) / ((pj.longitude - pi.longitude) + 1e-12) + pi.latitude);
        if (intersect) inside = !inside;
    }
    return inside;
}

Status SpatialEngine::WithinPolygon(const pomai::GeoPolygon& polygon, std::vector<pomai::SpatialPoint>* out) const {
    if (!out) return Status::InvalidArgument("spatial out null");
    out->clear();
    for (const auto& kv : points_) {
        if (PointInPolygon(kv.second.latitude, kv.second.longitude, polygon)) out->push_back(kv.second);
    }
    return Status::Ok();
}

Status SpatialEngine::Nearest(double lat, double lon, std::uint32_t topk, std::vector<pomai::SpatialPoint>* out) const {
    if (!out) return Status::InvalidArgument("spatial out null");
    out->clear();
    std::vector<std::pair<double, pomai::SpatialPoint>> scored;
    scored.reserve(points_.size());
    for (const auto& kv : points_) {
        scored.emplace_back(simd::HaversineMeters(lat, lon, kv.second.latitude, kv.second.longitude), kv.second);
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const std::size_t n = std::min<std::size_t>(scored.size(), topk);
    for (std::size_t i = 0; i < n; ++i) out->push_back(scored[i].second);
    return Status::Ok();
}

void SpatialEngine::ForEach(const std::function<void(std::uint64_t id, double lat, double lon)>& fn) const {
    for (const auto& [id, pt] : points_) fn(id, pt.latitude, pt.longitude);
}

} // namespace pomai::core

