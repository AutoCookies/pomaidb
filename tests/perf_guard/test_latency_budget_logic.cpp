#include <catch2/catch.hpp>

#include <pomai/index/whispergrain.h>

namespace
{
    using namespace pomai::ai;
    using pomai::WhisperConfig;
}

TEST_CASE("WhisperGrain budget reacts to latency and cpu", "[perf][budget]")
{
    WhisperConfig cfg;
    cfg.latency_target_ms = 10.0f;
    cfg.base_budget_ops = 10000;
    cfg.min_budget_ops = 500;
    cfg.hot_query_floor = 2000;
    cfg.budget_headroom = 2.0f;
    cfg.cpu_soft_threshold = 70.0f;
    cfg.cpu_hard_threshold = 90.0f;
    cfg.refine_enable_margin_ms = 2;

    WhisperGrain grain(cfg);
    grain.observe_latency(5.0f);
    grain.set_cpu_load(20.0f);
    auto budget = grain.compute_budget(false);

    REQUIRE(budget.ops_budget >= cfg.min_budget_ops);
    REQUIRE(budget.allow_exact_refine);

    grain.observe_latency(30.0f);
    grain.set_cpu_load(95.0f);
    auto stressed = grain.compute_budget(true);

    REQUIRE(stressed.ops_budget <= budget.ops_budget);
    REQUIRE_FALSE(stressed.allow_exact_refine);
    REQUIRE(stressed.ops_budget >= cfg.hot_query_floor);
}
