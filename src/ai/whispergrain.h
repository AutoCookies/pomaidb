#pragma once
/*
 * src/ai/whispergrain.h
 *
 * WhisperGrain v2.1 - Budget controller for Pomai
 * Refactored to use pomai::config::WhisperConfig.
 */

#include <cstdint>
#include <atomic>
#include <mutex>
#include <chrono>
#include "src/core/config.h" // [CHANGED] Include Config

namespace pomai::ai
{
    // Runtime Budget Result (Output of the controller, not config)
    struct Budget
    {
        uint32_t ops_budget = 0;         // total Op budget
        uint32_t bucket_budget = 0;      // how many buckets (spatial budget) allowed
        bool allow_exact_refine = false; // allow the expensive exact refine (IO)
    };

    class WhisperGrain
    {
    public:
        explicit WhisperGrain(const pomai::config::WhisperConfig &cfg);

        // Update controllers with observed latency (ms) and current cpu percent (0..100).
        void observe_latency(float latency_ms);
        void set_cpu_load(float cpu_percent);

        // Compute a Budget for a new query.
        Budget compute_budget(bool is_hot = false) const;

        // Expose runtime info for diagnostics
        float latency_ema() const { return latency_ema_.load(); }
        float cpu_load() const { return cpu_load_.load(); }

    private:
        pomai::config::WhisperConfig cfg_; // Store config copy

        // EMA state
        std::atomic<float> latency_ema_; // ms
        std::atomic<float> cpu_load_;    // %
    };

} // namespace pomai::ai