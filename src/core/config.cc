#include "src/core/config.h"
#include <cstdlib>
#include <string>
#include <iostream>
#include <thread>
#include <cstring>

namespace pomai::config
{
    Runtime runtime{};

    // Helper: Parse uint64 from env safely
    static std::optional<std::uint64_t> env_u64(const char *name)
    {
        const char *v = std::getenv(name);
        if (!v) return std::nullopt;
        try { return static_cast<std::uint64_t>(std::stoull(v)); }
        catch (...) { return std::nullopt; }
    }

    // Initialize defaults based on hardware
    static void set_hardware_defaults()
    {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        
        runtime.shard_count = static_cast<std::uint32_t>(hc);
        runtime.max_elements_total = 1000000; 

        std::cerr << "[Config] Defaults: Shards=" << runtime.shard_count
                  << ", MaxElements=" << runtime.max_elements_total 
                  << ", Port=" << runtime.default_port << "\n";
    }

    void init_from_env()
    {
        // 1. Hardware defaults first
        set_hardware_defaults();

        // 2. Override with ENV variables (legacy support)
        if (auto v = env_u64("POMAI_PORT")) 
            runtime.default_port = static_cast<std::uint16_t>(*v);
        
        if (auto v = env_u64("POMAI_ARENA_MB")) 
            runtime.arena_mb_per_shard = *v;
        
        if (auto v = env_u64("POMAI_RNG_SEED")) 
            runtime.rng_seed = *v;

        // Async IO
        if (auto v = env_u64("POMAI_DEMOTE_ASYNC_MAX_PENDING"))
            runtime.demote_async_max_pending = *v;
        
        const char *sync_fb = std::getenv("POMAI_DEMOTE_SYNC_FALLBACK");
        if (sync_fb) {
            runtime.demote_sync_fallback = (std::strcmp(sync_fb, "1") == 0 || std::strcmp(sync_fb, "true") == 0 || std::strcmp(sync_fb, "TRUE") == 0);
        }

        // Algo Tuning
        if (auto v = env_u64("POMAI_FINGERPRINT_BITS"))
            runtime.fingerprint_bits = static_cast<std::uint32_t>(*v);

        // Optional vector dimension via env (legacy)
        if (auto v = env_u64("POMAI_DIM"))
            runtime.dim = static_cast<std::uint32_t>(*v);

        // Optional disable synapse via env (legacy)
        const char *ds = std::getenv("POMAI_DISABLE_SYNAPSE");
        if (ds) {
            runtime.disable_synapse = (std::strcmp(ds, "1") == 0 || std::strcmp(ds, "true") == 0 || std::strcmp(ds, "TRUE") == 0);
        }

        // [FIX] Legacy Map Env Vars (Optional overrides)
        if (auto v = env_u64("POMAI_INITIAL_ENTROPY"))
            runtime.initial_entropy = static_cast<std::uint32_t>(*v);
        if (auto v = env_u64("POMAI_MAX_ENTROPY"))
            runtime.max_entropy = static_cast<std::uint32_t>(*v);
    }

    void init_from_args(int argc, char **argv)
    {
        // parse simple CLI flags:
        // --arena-mb <n>, --shards <n>, --dim <n>, --port <n>, --disable-synapse
        for (int i = 1; i < argc; ++i) {
            const char* a = argv[i];
            if (std::strcmp(a, "--arena-mb") == 0 && i + 1 < argc) {
                try { runtime.arena_mb_per_shard = static_cast<uint64_t>(std::stoull(argv[++i])); }
                catch (...) {}
            } else if (std::strcmp(a, "--shards") == 0 && i + 1 < argc) {
                try { runtime.shard_count = static_cast<uint32_t>(std::stoul(argv[++i])); }
                catch (...) {}
            } else if (std::strcmp(a, "--dim") == 0 && i + 1 < argc) {
                try { runtime.dim = static_cast<uint32_t>(std::stoul(argv[++i])); }
                catch (...) {}
            } else if (std::strcmp(a, "--port") == 0 && i + 1 < argc) {
                try { runtime.default_port = static_cast<uint16_t>(std::stoul(argv[++i])); }
                catch (...) {}
            } else if (std::strcmp(a, "--disable-synapse") == 0) {
                runtime.disable_synapse = true;
            }
        }
    }

} // namespace pomai::config