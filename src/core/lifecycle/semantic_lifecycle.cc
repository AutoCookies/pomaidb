#include "core/lifecycle/semantic_lifecycle.h"

namespace pomai::core {

void SemanticLifecycle::OnRead(VectorId id) {
    if (max_entries_ == 0) return;
    EvictIfNeeded();
    table_[id].reads++;
}

void SemanticLifecycle::OnWrite(VectorId id) {
    if (max_entries_ == 0) return;
    EvictIfNeeded();
    table_[id].writes++;
}

void SemanticLifecycle::OnDelete(VectorId id) {
    table_.erase(id);
}

DataTemperature SemanticLifecycle::Classify(VectorId id) const {
    auto it = table_.find(id);
    if (it == table_.end()) return DataTemperature::kCold;
    const auto score = it->second.reads * 3 + it->second.writes;
    if (score >= 20) return DataTemperature::kHot;
    if (score >= 5) return DataTemperature::kWarm;
    return DataTemperature::kCold;
}

std::size_t SemanticLifecycle::CountHot() const {
    std::size_t n = 0;
    for (const auto& kv : table_) if (Classify(kv.first) == DataTemperature::kHot) ++n;
    return n;
}
std::size_t SemanticLifecycle::CountWarm() const {
    std::size_t n = 0;
    for (const auto& kv : table_) if (Classify(kv.first) == DataTemperature::kWarm) ++n;
    return n;
}
std::size_t SemanticLifecycle::CountCold() const {
    std::size_t n = 0;
    for (const auto& kv : table_) if (Classify(kv.first) == DataTemperature::kCold) ++n;
    return n;
}

void SemanticLifecycle::EvictIfNeeded() {
    if (max_entries_ == 0) return;
    if (table_.size() < max_entries_) return;
    // O(1) best-effort eviction to bound RAM strictly.
    table_.erase(table_.begin());
}

} // namespace pomai::core

