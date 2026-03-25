#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai/pomai.h"
#include "pomai/status.h"

namespace pomai::core {

class TimeSeriesEngine {
public:
    TimeSeriesEngine(std::string path, std::size_t max_points_per_series);
    Status Open();
    Status Close();
    Status Put(std::uint64_t series_id, std::uint64_t ts, double value);
    Status Range(std::uint64_t series_id, std::uint64_t start_ts, std::uint64_t end_ts, std::vector<pomai::TimeSeriesPoint>* out) const;

    void ForEach(const std::function<void(std::uint64_t series_id, const pomai::TimeSeriesPoint&)>& fn) const;

private:
    std::string path_;
    std::size_t max_points_per_series_;
    std::unordered_map<std::uint64_t, std::vector<pomai::TimeSeriesPoint>> data_;
};

} // namespace pomai::core

