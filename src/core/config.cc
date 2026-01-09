// core/config.cc
#include "src/core/config.h"
#include <cstdlib>
#include <string>
#include <iostream>
#include <thread> // for hardware_concurrency
#include <cstring>

namespace pomai::config
{

    Runtime runtime{}; // defaults defined in struct

    // Initialize runtime defaults (shard_count and max_elements_total) to sensible values.
    // We do this at startup so the process doesn't need to read env vars for these.
    static bool init_runtime_defaults()
    {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 1;
        // Clamp to uint32_t
        runtime.shard_count = static_cast<std::uint32_t>(hc);

        // Default max_elements_total for vector index across all shards.
        // Set to 1,000,000 by default to allow tests with larger vector sets.
        runtime.max_elements_total = 1200000;

        std::cerr << "[config] default shard_count = " << runtime.shard_count
                  << ", default max_elements_total = " << runtime.max_elements_total << "\n";

        std::cerr << "[config] hot_size_limit = " << runtime.hot_size_limit
                  << ", promote_lookahead_ms = " << runtime.promote_lookahead_ms
                  << " ms, demote_threshold_ms = " << runtime.demote_threshold_ms
                  << " ms, demote_async_max_pending = " << runtime.demote_async_max_pending
                  << ", demote_sync_fallback = " << (runtime.demote_sync_fallback ? "true" : "false") << "\n";

        return true;
    }
    static bool runtime_defaults_initialized = init_runtime_defaults();

    static std::optional<std::uint64_t> u64_from_env(const char *name)
    {
        const char *v = std::getenv(name);
        if (!v)
            return std::nullopt;
        try
        {
            return static_cast<std::uint64_t>(std::stoull(v));
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    void init_from_env()
    {
        if (auto v = u64_from_env("POMAI_HARVEST_SAMPLE"))
            runtime.harvest_sample = static_cast<std::uint32_t>(*v);
        if (auto v = u64_from_env("POMAI_HARVEST_MAX_ATTEMPTS"))
            runtime.harvest_max_attempts = static_cast<std::uint32_t>(*v);
        if (auto v = u64_from_env("POMAI_INITIAL_ENTROPY"))
            runtime.initial_entropy = static_cast<std::uint32_t>(*v);
        if (auto v = u64_from_env("POMAI_MAX_ENTROPY"))
            runtime.max_entropy = static_cast<std::uint32_t>(*v);
        if (auto v = u64_from_env("POMAI_ARENA_MB"))
            runtime.arena_mb_per_shard = *v;
        if (auto v = u64_from_env("POMAI_PORT"))
            runtime.default_port = static_cast<std::uint16_t>(*v);
        if (auto v = u64_from_env("POMAI_RNG_SEED"))
            runtime.rng_seed = *v;

        // Phase‑4 thresholds: allow overriding via environment (optional)
        if (auto v = u64_from_env("POMAI_HOT_SIZE_LIMIT"))
            runtime.hot_size_limit = *v;
        if (auto v = u64_from_env("POMAI_PROMOTE_LOOKAHEAD_MS"))
            runtime.promote_lookahead_ms = *v;
        if (auto v = u64_from_env("POMAI_DEMOTE_THRESHOLD_MS"))
            runtime.demote_threshold_ms = *v;

        // Async demote knobs
        if (auto v = u64_from_env("POMAI_DEMOTE_ASYNC_MAX_PENDING"))
            runtime.demote_async_max_pending = *v;

        const char *sync_fallback = std::getenv("POMAI_DEMOTE_SYNC_FALLBACK");
        if (sync_fallback)
        {
            if (std::strcmp(sync_fallback, "1") == 0 || strcasecmp(sync_fallback, "true") == 0)
                runtime.demote_sync_fallback = true;
            else
                runtime.demote_sync_fallback = false;
        }

        // NOTE: We intentionally DO NOT read POMAI_SHARD_COUNT from the environment here.
        // The default shard_count is already set above to hardware_concurrency().
        // If you want to override max_elements_total via env, you can add:
        // if (auto v = u64_from_env("POMAI_MAX_ELEMENTS_TOTAL")) runtime.max_elements_total = *v;
    }

    void init_from_args(int argc, char **argv)
    {
        // Minimal: you can parse basic flags here or call a proper CLI parser later.
        // For now prefer env or main can set values directly.
        (void)argc;
        (void)argv;
    }

} // namespace pomai::config