#include "src/ai/whispergrain.h"

#include <algorithm>
#include <cmath>
#include <chrono>

namespace pomai::ai
{

    WhisperGrain::WhisperGrain(const WhisperConfig &cfg)
        : cfg_(cfg), latency_ema_(cfg.latency_target_ms), cpu_load_(0.0f)
    {
        // start latency_ema_ initialized at target as neutral baseline
    }

    void WhisperGrain::observe_latency(float latency_ms)
    {
        // Exponential moving average: EMA_new = alpha * sample + (1-alpha) * EMA_old
        float alpha = cfg_.latency_ema_alpha;
        float old = latency_ema_.load();
        float nxt = alpha * latency_ms + (1.0f - alpha) * old;
        latency_ema_.store(nxt);
    }

    void WhisperGrain::set_cpu_load(float cpu_percent)
    {
        cpu_load_.store(cpu_percent);
    }

    Budget WhisperGrain::compute_budget(bool is_hot) const
    {
        Budget b;
        // base ops
        float base = static_cast<float>(cfg_.base_budget_ops);

        // Latency damping: if EMA > target, scale down. Use sqrt damping:
        // scale = sqrt(target / ema)  (if ema > target) else 1.0 (or slightly >1 allow headroom)
        float ema = latency_ema_.load();
        float target = cfg_.latency_target_ms;
        float scale = 1.0f;
        if (ema > target && ema > 0.0f)
        {
            scale = std::sqrt(target / ema); // in (0,1]
        }
        else
        {
            // if system is idle relative to target, allow headroom
            scale = cfg_.budget_headroom;
        }

        // CPU safety rail
        float cpu = cpu_load_.load();
        if (cpu >= cfg_.cpu_hard_threshold)
        {
            scale *= 0.25f; // heavy cut
        }
        else if (cpu >= cfg_.cpu_soft_threshold)
        {
            scale *= 0.6f; // mild cut
        }

        // Apply hot-query floor
        float min_ops = static_cast<float>(is_hot ? cfg_.hot_query_floor : cfg_.min_budget_ops);

        float ops = base * scale;
        if (ops < min_ops)
            ops = min_ops;

        // clamp and allow small headroom
        ops = std::clamp(ops, static_cast<float>(cfg_.min_budget_ops), base * cfg_.budget_headroom);

        b.ops_budget = static_cast<uint32_t>(std::floor(ops + 0.5f));

        // Spatial budget heuristic: bucket_budget proportional to ops
        // e.g. assume each bucket costs ~ (10 ops) on average
        uint32_t per_bucket_ops = 10;
        b.bucket_budget = std::max<uint32_t>(1, b.ops_budget / per_bucket_ops);

        // allow exact refine only if ops budget is comfortably above base*0.5 and cpu is low
        if (b.ops_budget > (base * 0.6f) && cpu < (cfg_.cpu_soft_threshold - 5.0f))
            b.allow_exact_refine = true;
        else
            b.allow_exact_refine = false;

        return b;
    }

} // namespace pomai::ai