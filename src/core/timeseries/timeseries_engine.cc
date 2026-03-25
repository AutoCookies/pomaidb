#include "core/timeseries/timeseries_engine.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace pomai::core {

TimeSeriesEngine::TimeSeriesEngine(std::string path, std::size_t max_points_per_series)
    : path_(std::move(path)), max_points_per_series_(max_points_per_series) {}

Status TimeSeriesEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("timeseries create dir failed");
    std::ifstream in(path_ + "/timeseries.log", std::ios::binary);
    if (!in.good()) return Status::Ok();
    std::uint64_t sid = 0, ts = 0;
    double value = 0.0;
    while (in.read(reinterpret_cast<char*>(&sid), sizeof(sid))) {
        if (!in.read(reinterpret_cast<char*>(&ts), sizeof(ts))) break;
        if (!in.read(reinterpret_cast<char*>(&value), sizeof(value))) break;
        auto& vec = data_[sid];
        vec.push_back({ts, value});
        if (max_points_per_series_ > 0 && vec.size() > max_points_per_series_) {
            vec.erase(vec.begin(), vec.begin() + static_cast<long>(vec.size() - max_points_per_series_));
        }
    }
    return Status::Ok();
}

Status TimeSeriesEngine::Close() { return Status::Ok(); }

Status TimeSeriesEngine::Put(std::uint64_t series_id, std::uint64_t ts, double value) {
    auto& vec = data_[series_id];
    if (!vec.empty()) {
        // Keep ordered by time with append-fast path.
        if (ts < vec.back().timestamp) {
            vec.push_back({ts, value});
            std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
        } else {
            vec.push_back({ts, value});
        }
    } else {
        vec.push_back({ts, value});
    }
    if (max_points_per_series_ > 0 && vec.size() > max_points_per_series_) {
        vec.erase(vec.begin(), vec.begin() + static_cast<long>(vec.size() - max_points_per_series_));
    }

    std::ofstream out(path_ + "/timeseries.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("timeseries append failed");
    out.write(reinterpret_cast<const char*>(&series_id), sizeof(series_id));
    out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    return Status::Ok();
}

Status TimeSeriesEngine::Range(std::uint64_t series_id, std::uint64_t start_ts, std::uint64_t end_ts, std::vector<pomai::TimeSeriesPoint>* out) const {
    if (!out) return Status::InvalidArgument("timeseries out is null");
    out->clear();
    auto it = data_.find(series_id);
    if (it == data_.end()) return Status::Ok();
    for (const auto& p : it->second) {
        if (p.timestamp >= start_ts && p.timestamp <= end_ts) out->push_back(p);
    }
    return Status::Ok();
}

void TimeSeriesEngine::ForEach(const std::function<void(std::uint64_t series_id, const pomai::TimeSeriesPoint&)>& fn) const {
    for (const auto& [sid, pts] : data_) {
        for (const auto& p : pts) {
            fn(sid, p);
        }
    }
}

} // namespace pomai::core

