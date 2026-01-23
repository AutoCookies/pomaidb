#pragma once

#include <cstdint>
#include <atomic>
#include "src/core/config.h"

namespace pomai::ai
{
    struct Budget
    {
        uint32_t ops_budget = 0;
        uint32_t bucket_budget = 0;
        bool allow_exact_refine = false;
    };

    class WhisperGrain
    {
    public:
        explicit WhisperGrain(const pomai::config::WhisperConfig &cfg);

        void observe_latency(float latency_ms);
        void set_cpu_load(float cpu_percent);
        Budget compute_budget(bool is_hot = false) const;

        float latency_ema() const noexcept { return latency_ema_.load(std::memory_order_acquire); }
        float cpu_load() const noexcept { return cpu_load_.load(std::memory_order_relaxed); }
        uint64_t observations() const noexcept { return observations_.load(std::memory_order_relaxed); }

    private:
        pomai::config::WhisperConfig cfg_;
        std::atomic<float> latency_ema_;
        std::atomic<float> cpu_load_;
        std::atomic<uint64_t> observations_;
    };
}