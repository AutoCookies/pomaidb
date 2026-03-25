#include "core/sketch/sketch_engine.h"

namespace pomai::core {

Status SketchEngine::Add(std::string_view key, uint64_t increment) {
    if (key.empty()) return Status::InvalidArgument("sketch key empty");
    EvictIfNeeded();
    auto& v = counts_[std::string(key)];
    v += increment;
    seen_.insert(std::string(key));
    return Status::Ok();
}

Status SketchEngine::Estimate(std::string_view key, uint64_t* out) const {
    if (!out) return Status::InvalidArgument("sketch out null");
    auto it = counts_.find(std::string(key));
    *out = (it == counts_.end()) ? 0 : it->second;
    return Status::Ok();
}

Status SketchEngine::Seen(std::string_view key, bool* out) const {
    if (!out) return Status::InvalidArgument("sketch out null");
    *out = seen_.find(std::string(key)) != seen_.end();
    return Status::Ok();
}

Status SketchEngine::UniqueEstimate(uint64_t* out) const {
    if (!out) return Status::InvalidArgument("sketch out null");
    *out = static_cast<uint64_t>(seen_.size());
    return Status::Ok();
}

void SketchEngine::ForEach(const std::function<void(std::string_view key, uint64_t count)>& fn) const {
    for (const auto& [k, c] : counts_) fn(k, c);
}

void SketchEngine::EvictIfNeeded() {
    if (max_entries_ == 0) return;
    if (counts_.size() < max_entries_) return;
    auto it = counts_.begin();
    if (it != counts_.end()) {
        seen_.erase(it->first);
        counts_.erase(it);
    }
}

} // namespace pomai::core

