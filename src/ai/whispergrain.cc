/*
 * src/ai/whispergrain.cc
 *
 * Implementation of WhisperGrain (Adaptive Budget Controller).
 *
 * Improvements:
 * - Thread-Safety: Atomic CAS loop for EMA updates.
 * - Fast Start: First latency sample initializes EMA directly (Jump) 
 * to avoid slow convergence from the target baseline (preventing initial over-throttling).
 */

#include "src/ai/whispergrain.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <atomic>

namespace pomai::ai
{

    WhisperGrain::WhisperGrain(const WhisperConfig &cfg)
        : cfg_(cfg), latency_ema_(-1.0f), cpu_load_(0.0f) // [FIX] Init to sentinel -1.0
    {
    }

    void WhisperGrain::observe_latency(float latency_ms)
    {
        float alpha = cfg_.latency_ema_alpha;
        float old_val = latency_ema_.load(std::memory_order_relaxed);

        // [FIX] Fast Start: If this is the first observation (value is -1), 
        // snap directly to the sample instead of blending.
        if (old_val < 0.0f) {
            // Try to set the first value (Jump Start). 
            // compare_exchange_strong updates old_val if it fails.
            if (latency_ema_.compare_exchange_strong(old_val, latency_ms, 
                                                     std::memory_order_release, 
                                                     std::memory_order_relaxed)) {
                return; // Successfully initialized
            }
            // If CAS failed, old_val is now the valid value set by another thread.
            // Fall through to normal EMA logic.
        }

        // Standard EMA CAS Loop (Thread-safe update)
        // EMA_new = alpha * sample + (1-alpha) * EMA_old
        float new_val;
        do {
            new_val = alpha * latency_ms + (1.0f - alpha) * old_val;
        } while (!latency_ema_.compare_exchange_weak(old_val, new_val, 
                                                     std::memory_order_release, 
                                                     std::memory_order_relaxed));
    }

    void WhisperGrain::set_cpu_load(float cpu_percent)
    {
        // Simple store is fine for CPU load as it's a "latest value" metric
        cpu_load_.store(cpu_percent, std::memory_order_release);
    }

    Budget WhisperGrain::compute_budget(bool is_hot) const
    {
        Budget b;
        
        // 1. Snapshot atomic metrics (Consistency View)
        float ema = latency_ema_.load(std::memory_order_acquire);
        float cpu = cpu_load_.load(std::memory_order_acquire);
        
        // [FIX] Handle uninitialized state (before first query)
        // Treat as if we are exactly at target latency (neutral scale = 1.0)
        if (ema < 0.0f) {
            ema = cfg_.latency_target_ms;
        }

        // Base ops budget
        float base = static_cast<float>(cfg_.base_budget_ops);

        // 2. Latency damping: Scale down if system is slow
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
        if (ops < min_ops) ops = min_ops;

        // Apply ceilings (Max Burst)
        float max_ops = base * cfg_.budget_headroom;
        if (ops > max_ops) ops = max_ops;

        b.ops_budget = static_cast<uint32_t>(ops);

        // 5. Spatial Budget Heuristic
        constexpr uint32_t COST_PER_BUCKET = 10;
        b.bucket_budget = std::max<uint32_t>(1, b.ops_budget / COST_PER_BUCKET);

        // 6. Refinement Policy
        bool cpu_ok = cpu < (cfg_.cpu_soft_threshold - 5.0f);
        bool budget_ok = b.ops_budget > (base * 0.6f);
        
        b.allow_exact_refine = (cpu_ok && budget_ok);

        return b;
    }

} // namespace pomai::ai