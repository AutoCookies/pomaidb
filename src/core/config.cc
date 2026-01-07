// core/config.cc
#include "src/core/config.h"
#include <cstdlib>
#include <string>
#include <iostream>

namespace pomai::config
{

    Runtime runtime{}; // defaults defined in struct

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
    }

    void init_from_args(int argc, char **argv)
    {
        // Minimal: you can parse basic flags here or call a proper CLI parser later.
        // For now prefer env or main can set values directly.
        (void)argc;
        (void)argv;
    }

} // namespace pomai::config