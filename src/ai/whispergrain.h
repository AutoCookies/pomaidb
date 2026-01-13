#pragma once
/*
 * src/ai/whispergrain.h
 *
 * WhisperGrain v2.1 - Budget controller for Pomai
 *
 * Provides:
 *  - Configurable cost model (Ops per action)
 *  - EMA latency tracking + CPU safety gating
 *  - compute_budget(...) to return an execution budget for a query
 *
 * This is a standalone controller. It is lightweight and thread-safe for read-mostly use.
 */

#include <cstdint>
#include <atomic>
#include <mutex>
#include <chrono>

namespace pomai::ai
{

    struct WhisperConfig
    {
        // Cost model (ops)
        uint32_t cost_check = 1;       // cheap checks
        uint32_t cost_echo_decode = 5; // EternalEcho decode + distance
        uint32_t cost_exact = 100;     // disk / full vector exact cost

        // Budget defaults
        uint32_t base_budget_ops = 5000; // default per-query budget
        uint32_t min_budget_ops = 250;   // minimum budget allowed for normal queries
        uint32_t hot_query_floor = 2000; // guaranteed floor for hot queries
        float budget_headroom = 1.2f;    // allow up to +20% if idle

        // Latency target for EMA (ms)
        float latency_target_ms = 50.0f;

        // EMA alpha (0..1) for latency smoothing (higher -> more responsive)
        float latency_ema_alpha = 0.15f;

        // CPU thresholds (percent)
        float cpu_soft_threshold = 75.0f; // mild penalty
        float cpu_hard_threshold = 90.0f; // heavy penalty

        // Hysteresis windows (ms) for refine toggling
        uint32_t refine_enable_margin_ms = 20;
    };

    struct Budget
    {
        uint32_t ops_budget = 0;         // total Op budget
        uint32_t bucket_budget = 0;      // how many buckets (spatial budget) allowed
        bool allow_exact_refine = false; // allow the expensive exact refine (IO)
    };

    class WhisperGrain
    {
    public:
        WhisperGrain(const WhisperConfig &cfg = WhisperConfig());

        // Update controllers with observed latency (ms) and current cpu percent (0..100).
        // Called by server for each completed query or periodically.
        void observe_latency(float latency_ms);
        void set_cpu_load(float cpu_percent);

        // Compute a Budget for a new query.
        // is_hot: whether this query was seen frequently (hot query)
        Budget compute_budget(bool is_hot = false) const;

        // Expose runtime info for diagnostics
        float latency_ema() const { return latency_ema_.load(); }
        float cpu_load() const { return cpu_load_.load(); }

    private:
        WhisperConfig cfg_;

        // EMA state
        std::atomic<float> latency_ema_; // ms

        // CPU load snapshot
        std::atomic<float> cpu_load_; // %

        // internal mutex for safe updates where needed
        mutable std::mutex mu_;
    };

} // namespace pomai::ai