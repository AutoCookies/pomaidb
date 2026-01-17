/*
 * src/ai/whispergrain.cc
 *
 * Implementation of WhisperGrain (Adaptive Budget Controller).
 */

#include "src/ai/whispergrain.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <atomic>

namespace pomai::ai
{

    WhisperGrain::WhisperGrain(const pomai::config::WhisperConfig &cfg)
        : cfg_(cfg), latency_ema_(-1.0f), cpu_load_(0.0f)
    {
    }

    void WhisperGrain::observe_latency(float latency_ms)
    {
        float alpha = cfg_.latency_ema_alpha;
        float old_val = latency_ema_.load(std::memory_order_relaxed);

        // Fast Start: If first observation, snap directly.
        if (old_val < 0.0f)
        {
            if (latency_ema_.compare_exchange_strong(old_val, latency_ms,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed))
            {
                return;
            }
            // If CAS failed, someone else initialized it, proceed to blend.
            old_val = latency_ema_.load(std::memory_order_relaxed);
        }

        // Standard EMA update: New = Alpha * Sample + (1-Alpha) * Old
        // CAS loop for thread safety
        float new_val = old_val;
        do
        {
            new_val = alpha * latency_ms + (1.0f - alpha) * old_val;
        } while (!latency_ema_.compare_exchange_weak(old_val, new_val,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed));
    }

    void WhisperGrain::set_cpu_load(float cpu_percent)
    {
        cpu_load_.store(cpu_percent, std::memory_order_relaxed);
    }

    Budget WhisperGrain::compute_budget(bool is_hot) const
    {
        Budget b;
        float ema = latency_ema_.load(std::memory_order_relaxed);
        float cpu = cpu_load_.load(std::memory_order_relaxed);

        // 1. Base Budget
        float base = static_cast<float>(cfg_.base_budget_ops);

        // 2. Latency Feedback Control
        // If EMA < Target: We are fast -> Scale UP (Relax budget)
        // If EMA > Target: We are slow -> Scale DOWN (Tighten budget)

        // Handle uninitialized state (ema < 0) -> treat as meeting target
        if (ema < 0.0f)
            ema = cfg_.latency_target_ms;

        float scale = 1.0f;
        float target = cfg_.latency_target_ms;

        if (ema > target && ema > 0.001f)
        {
            // Sqrt damping for smoother degradation
            scale = std::sqrt(target / ema);
        }
        else
        {
            // System idle -> Allow boost (Headroom)
            scale = cfg_.budget_headroom;
        }

        // 3. CPU Safety Rail (Thermal Control)
        if (cpu >= cfg_.cpu_hard_threshold)
        {
            scale *= 0.25f; // Hard throttling
        }
        else if (cpu >= cfg_.cpu_soft_threshold)
        {
            scale *= 0.6f; // Soft backpressure
        }

        // 4. Calculate Final OPS
        float ops = base * scale;

        // Apply floors (Min QoS)
        float min_ops = static_cast<float>(is_hot ? cfg_.hot_query_floor : cfg_.min_budget_ops);
        if (ops < min_ops)
            ops = min_ops;

        // Apply ceilings (Max Burst)
        float max_ops = base * cfg_.budget_headroom;
        if (ops > max_ops)
            ops = max_ops;

        b.ops_budget = static_cast<uint32_t>(ops);

        // 5. Spatial Budget Heuristic
        constexpr uint32_t COST_PER_BUCKET = 10;
        b.bucket_budget = std::max<uint32_t>(1, b.ops_budget / COST_PER_BUCKET);

        // 6. Refine Policy (IO permission)
        // Only allow exact refine if we have plenty of latency margin
        if (ema < (target - static_cast<float>(cfg_.refine_enable_margin_ms)) &&
            cpu < cfg_.cpu_soft_threshold)
        {
            b.allow_exact_refine = true;
        }
        else
        {
            b.allow_exact_refine = false;
        }

        return b;
    }

} // namespace pomai::ai