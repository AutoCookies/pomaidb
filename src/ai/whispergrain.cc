#include "src/ai/whispergrain.h"
#include "src/core/metrics.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <atomic>
#include <iostream>

namespace pomai::ai
{
    WhisperGrain::WhisperGrain(const pomai::config::WhisperConfig &cfg)
        : cfg_(cfg), latency_ema_(-1.0f), cpu_load_(0.0f)
    {
    }

    void WhisperGrain::observe_latency(float latency_ms)
    {
        PomaiMetrics::total_searches.fetch_add(1, std::memory_order_relaxed);
        if (latency_ms < 0.001f)
        {
            PomaiMetrics::searches_fast_miss.fetch_add(1, std::memory_order_relaxed);
            latency_ms = 0.001f;
        }
        float alpha = cfg_.latency_ema_alpha;
        float old_val = latency_ema_.load(std::memory_order_acquire);
        if (old_val <= 0.0f)
        {
            latency_ema_.store(latency_ms, std::memory_order_release);
            return;
        }
        float new_val = alpha * latency_ms + (1.0f - alpha) * old_val;
        latency_ema_.store(new_val, std::memory_order_release);
    }

    void WhisperGrain::set_cpu_load(float cpu_percent)
    {
        cpu_load_.store(cpu_percent, std::memory_order_relaxed);
    }

    Budget WhisperGrain::compute_budget(bool is_hot) const
    {
        Budget b;
        float ema = latency_ema_.load(std::memory_order_acquire);
        float cpu = cpu_load_.load(std::memory_order_relaxed);
        float base = static_cast<float>(cfg_.base_budget_ops);
        if (ema <= 0.0f)
            ema = cfg_.latency_target_ms;
        float scale = 1.0f;
        float target = cfg_.latency_target_ms;
        if (ema > target)
            scale = std::sqrt(target / ema);
        else
            scale = cfg_.budget_headroom;
        if (cpu >= cfg_.cpu_hard_threshold)
            scale *= 0.25f;
        else if (cpu >= cfg_.cpu_soft_threshold)
            scale *= 0.6f;
        float ops = base * scale;
        float min_ops = static_cast<float>(is_hot ? cfg_.hot_query_floor : cfg_.min_budget_ops);
        if (ops < min_ops)
            ops = min_ops;
        float max_ops = base * cfg_.budget_headroom;
        if (ops > max_ops)
            ops = max_ops;
        b.ops_budget = static_cast<uint32_t>(ops);
        constexpr uint32_t OPS_PER_BUCKET = 500;
        b.bucket_budget = std::max<uint32_t>(16, b.ops_budget / OPS_PER_BUCKET);
        b.allow_exact_refine = (ema < (target - static_cast<float>(cfg_.refine_enable_margin_ms)) &&
                                cpu < cfg_.cpu_soft_threshold);
        return b;
    }
} // namespace pomai::ai