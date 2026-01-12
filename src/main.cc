/*
 * examples/main.cc
 *
 * Entry point for Pomai Server.
 *
 * Responsibilities:
 * - Initialize Global Config.
 * - Initialize Global KV Arena (PomaiMap).
 * - Start PomaiServer (which manages the Vector Engine internally).
 * - Handle Signals for graceful shutdown.
 */

#include "src/core/config.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/facade/server.h"

#include <signal.h>
#include <sys/resource.h>
#include <iostream>
#include <algorithm>
#include <random>

static PomaiServer *g_server = nullptr;

// Signal handler: trigger server stop
static void on_signal(int)
{
    if (g_server)
        g_server->stop();
}

int main(int argc, char **argv)
{
    // 1. Config Init
    pomai::config::init_from_env();
    pomai::config::init_from_args(argc, argv);

    // 2. System tuning
    struct rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
    {
        rlim_t want = std::max<rlim_t>(rl.rlim_cur, 65536);
        if (want > rl.rlim_max)
            want = rl.rlim_max;
        rl.rlim_cur = want;
        setrlimit(RLIMIT_NOFILE, &rl);
    }

    // 3. Initialize Global KV Arena (For Metadata/Strings)
    // Note: Vector data lives in separate Shard arenas managed by PPSM.
    uint64_t arena_mb = pomai::config::runtime.arena_mb_per_shard; // Reuse config
    if (arena_mb == 0)
        arena_mb = 64; // Default for KV

    PomaiArena kv_arena = PomaiArena::FromMB(arena_mb);
    if (!kv_arena.is_valid())
    {
        std::cerr << "[Main] KV Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = kv_arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;

    PomaiMap kv_map(&kv_arena, slots);

    // 4. RNG Init
    if (pomai::config::runtime.rng_seed.has_value())
        kv_arena.seed_rng(*pomai::config::runtime.rng_seed);
    else
        kv_arena.seed_rng(std::random_device{}());

    // 5. Start Server
    int port = static_cast<int>(pomai::config::runtime.default_port);
    std::cout << "[Pomai] Orbit Edition starting on port=" << port << "\n";

    try
    {
        PomaiServer server(&kv_map, port);
        g_server = &server;

        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        // Keep main thread alive
        while (true) // In real server, server.run() might block
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            // Check if stopped?
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Main] Fatal error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}