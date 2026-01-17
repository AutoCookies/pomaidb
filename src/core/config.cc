/*
 * src/core/config.cc
 *
 * Implementation of Configuration Loading (Factory Pattern).
 * Refactored to match the new nested PomaiConfig structure.
 */

#include "src/core/config.h"
#include <cstdlib>
#include <string>
#include <iostream>
#include <thread>
#include <cstring>
#include <algorithm>

namespace pomai::config
{
    // Helper: Parse uint64 from env safely
    static std::optional<std::uint64_t> env_u64(const char *name)
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

    PomaiConfig load_from_args(int argc, char **argv)
    {
        PomaiConfig cfg;

        // 1. Hardware Defaults
        // (Optional: set defaults based on hardware if needed,
        // e.g. thread count for some internal pool not yet in config)
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 1;

        // 2. Env Overrides
        if (auto v = env_u64("POMAI_PORT"))
            cfg.net.port = static_cast<uint16_t>(*v);

        if (auto v = env_u64("POMAI_MEM_MB"))
            cfg.res.arena_mb_per_shard = *v;

        if (const char *root = std::getenv("POMAI_DB_DIR"))
            cfg.res.data_root = std::string(root);

        if (auto v = env_u64("POMAI_RNG_SEED"))
            cfg.rng_seed = *v;

        // 3. CLI Args Overrides
        for (int i = 1; i < argc; ++i)
        {
            const char *a = argv[i];

            // --port <p>
            if (std::strcmp(a, "--port") == 0 && i + 1 < argc)
            {
                try
                {
                    cfg.net.port = static_cast<uint16_t>(std::stoul(argv[++i]));
                }
                catch (...)
                {
                }
            }
            // --arena-mb <mb>
            else if (std::strcmp(a, "--arena-mb") == 0 && i + 1 < argc)
            {
                try
                {
                    cfg.res.arena_mb_per_shard = std::stoull(argv[++i]);
                }
                catch (...)
                {
                }
            }
            // --data-root <path>
            else if (std::strcmp(a, "--data-root") == 0 && i + 1 < argc)
            {
                cfg.res.data_root = argv[++i];
            }
            // --dim <n> (Legacy support: might hint default dim if not specified in CREATE)
            else if (std::strcmp(a, "--dim") == 0 && i + 1 < argc)
            {
                // Just consume arg to avoid error, logic handled per-membrance
                i++;
            }
            // --seed <n>
            else if (std::strcmp(a, "--seed") == 0 && i + 1 < argc)
            {
                try
                {
                    cfg.rng_seed = std::stoull(argv[++i]);
                }
                catch (...)
                {
                }
            }
        }

        // Sync aliases
        // Ensure network config port matches main port if they are intended to be the same
        cfg.network.udp_port = cfg.net.port;

        // Update the backward-compat alias
        cfg.cortex_cfg = cfg.network;

        std::cerr << "[Config] Loaded: Port=" << cfg.net.port
                  << ", ArenaMB=" << cfg.res.arena_mb_per_shard
                  << ", Root=" << cfg.res.data_root << "\n";

        return cfg;
    }

} // namespace pomai::config