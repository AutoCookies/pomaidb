#include "src/ai/whispergrain.h"

#include <iostream>
#include <cmath>

using namespace pomai::ai;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

struct Runner
{
    void expect(bool cond, const char *name)
    {
        if (cond)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << name << "\n";
            passed++;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << "\n";
            failed++;
        }
    }

    int summary()
    {
        std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
        return failed == 0 ? 0 : 1;
    }

    int passed = 0;
    int failed = 0;
};

int main()
{
    Runner r;

    pomai::config::WhisperConfig cfg;
    // tune config for deterministic behavior in tests
    cfg.latency_ema_alpha = 0.5f;
    cfg.latency_target_ms = 100.0f;
    cfg.base_budget_ops = 100;
    cfg.budget_headroom = 2.0f;
    cfg.cpu_hard_threshold = 90.0f;
    cfg.cpu_soft_threshold = 70.0f;
    cfg.hot_query_floor = 10;
    cfg.min_budget_ops = 5;
    cfg.refine_enable_margin_ms = 10;

    WhisperGrain wg(cfg);

    // Initial state: latency_ema uninitialized (-1), cpu 0
    r.expect(wg.latency_ema() < 0.0f, "initial latency_ema is negative (uninitialized)");
    r.expect(std::fabs(wg.cpu_load() - 0.0f) < 1e-6f, "initial cpu_load is 0");

    // 1) compute_budget with uninitialized EMA should treat as meeting target and use headroom
    {
        Budget b = wg.compute_budget(false);
        // ops = base * budget_headroom
        uint32_t expect_ops = static_cast<uint32_t>(cfg.base_budget_ops * cfg.budget_headroom);
        r.expect(b.ops_budget == expect_ops, "compute_budget uses headroom when EMA uninitialized");
        r.expect(b.bucket_budget == std::max<uint32_t>(1, expect_ops / 10), "bucket_budget derived from ops_budget");
        r.expect(b.allow_exact_refine == false, "allow_exact_refine false when EMA uninitialized");
    }

    // 2) Observe a very low latency and low CPU -> allow exact refine, boosted budget
    {
        wg.observe_latency(10.0f); // much lower than target
        wg.set_cpu_load(20.0f);    // low CPU
        float ema = wg.latency_ema();
        r.expect(ema >= 0.0f && ema <= 10.0f + 1e-3f, "latency_ema updated after first observation");

        Budget b = wg.compute_budget(false);
        // since ema < target - margin (10 < 100 - 10) and cpu < soft threshold -> allow refine
        r.expect(b.allow_exact_refine == true, "allow_exact_refine true under low latency and low CPU");

        // ops should be >= base (headroom applied) but not exceed max (= base * headroom)
        uint32_t max_ops = static_cast<uint32_t>(cfg.base_budget_ops * cfg.budget_headroom);
        r.expect(b.ops_budget <= max_ops && b.ops_budget >= cfg.base_budget_ops, "ops_budget within expected boosted range");
    }

    // 3) Observe very high latency and high CPU -> throttled budget and no refine
    {
        wg.observe_latency(500.0f); // high latency
        wg.set_cpu_load(95.0f);     // above hard threshold
        float ema = wg.latency_ema();
        r.expect(ema > 0.0f, "latency_ema remains positive after high observation");

        Budget b = wg.compute_budget(true); // hot query
        // cpu >= hard threshold -> scale multiplied by 0.25, so ops significantly lowered
        uint32_t min_ops = static_cast<uint32_t>(cfg.hot_query_floor);
        r.expect(b.ops_budget >= min_ops, "ops_budget respects hot_query_floor");
        r.expect(b.allow_exact_refine == false, "allow_exact_refine false under high latency or CPU");
    }

    // 4) Stability: multiple observe_latency blends via EMA (approx)
    {
        WhisperGrain w2(cfg);
        w2.observe_latency(100.0f); // first snap
        float e1 = w2.latency_ema();
        r.expect(std::fabs(e1 - 100.0f) < 1e-6f, "first latency observation snaps to value");

        w2.observe_latency(60.0f);
        float e2 = w2.latency_ema();
        // with alpha=0.5, new = 0.5*60 + 0.5*100 = 80
        r.expect(std::fabs(e2 - 80.0f) < 1e-3f, "EMA update matches expected blend (alpha)");
    }

    return r.summary();
}