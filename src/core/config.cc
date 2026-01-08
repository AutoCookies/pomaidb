// core/config.cc
#include "src/core/config.h"
#include <cstdlib>
#include <string>
#include <iostream>
#include <thread> // for hardware_concurrency

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
        // Set to 128K by default to allow tests with >100K vectors.
        runtime.max_elements_total = 131072; // 128 * 1024

        std::cerr << "[config] default shard_count = " << runtime.shard_count
                  << ", default max_elements_total = " << runtime.max_elements_total << "\n";
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