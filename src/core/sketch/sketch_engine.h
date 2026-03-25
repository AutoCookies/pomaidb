#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "pomai/status.h"

namespace pomai::core {

class SketchEngine {
public:
    explicit SketchEngine(std::size_t max_entries) : max_entries_(max_entries) {}
    Status Add(std::string_view key, uint64_t increment);
    Status Estimate(std::string_view key, uint64_t* out) const;
    Status Seen(std::string_view key, bool* out) const;
    Status UniqueEstimate(uint64_t* out) const;
    void ForEach(const std::function<void(std::string_view key, uint64_t count)>& fn) const;

private:
    void EvictIfNeeded();
    std::size_t max_entries_ = 20000;
    std::unordered_map<std::string, uint64_t> counts_;
    std::unordered_set<std::string> seen_;
};

} // namespace pomai::core

