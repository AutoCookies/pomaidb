#pragma once
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace pomai::core::metrics {

    enum class MetricType {
        kCounter,
        kGauge
    };

    /**
     * Centralized Metrics Registry for PomaiDB.
     * Optimized for Single-Threaded Edge: Zero-Lock and Zero-Atomic.
     */
    class MetricsRegistry {
    public:
        static MetricsRegistry& Instance() {
            static MetricsRegistry instance;
            return instance;
        }

        void Increment(const std::string& name, uint64_t count = 1) {
            counters_[name] += count;
        }

        void SetGauge(const std::string& name, uint64_t value) {
            gauges_[name] = value;
        }

        uint64_t GetCounter(const std::string& name) const {
            auto it = counters_.find(name);
            return (it != counters_.end()) ? it->second : 0;
        }

        uint64_t GetGauge(const std::string& name) const {
            auto it = gauges_.find(name);
            return (it != gauges_.end()) ? it->second : 0;
        }

        std::map<std::string, uint64_t> SnapshotCounters() const {
            return counters_;
        }

    private:
        MetricsRegistry() = default;
        mutable std::map<std::string, uint64_t> counters_;
        mutable std::map<std::string, uint64_t> gauges_;
    };

} // namespace pomai::core::metrics
