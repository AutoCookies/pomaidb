// examples/main.cpp - improved startup and graceful shutdown
#include "core/config.h"
#include "memory/arena.h"
#include "core/map.h"
#include "core/seed.h"
#include "facade/server.h"

#include <signal.h>
#include <sys/resource.h>
#include <iostream>
#include <algorithm>
#include <random>

static PomaiServer *g_server = nullptr;

// Signal handler: trigger server stop (writes to server.eventfd)
static void on_signal(int)
{
    if (g_server)
        g_server->stop();
}

int main(int argc, char **argv)
{
    pomai::config::init_from_env();
    pomai::config::init_from_args(argc, argv);

    // Increase RLIMIT_NOFILE modestly (typical for servers)
    struct rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
    {
        rlim_t want = std::max<rlim_t>(rl.rlim_cur, 65536);
        if (want > rl.rlim_max)
            want = rl.rlim_max;
        rl.rlim_cur = want;
        setrlimit(RLIMIT_NOFILE, &rl);
    }

    uint64_t arena_mb = pomai::config::runtime.arena_mb_per_shard;
    if (arena_mb == 0)
        arena_mb = 16;
    PomaiArena arena = PomaiArena::FromMB(arena_mb);
    if (!arena.is_valid())
    {
        std::cerr << "Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    if (slots == 0)
        slots = 1;

    PomaiMap map(&arena, slots);

    // seed rng deterministically if provided, otherwise random_device
    if (pomai::config::runtime.rng_seed.has_value())
        arena.seed_rng(*pomai::config::runtime.rng_seed);
    else
        arena.seed_rng(std::random_device{}());

    int port = static_cast<int>(pomai::config::runtime.default_port);

    std::cout << "[Pomai] starting server on port=" << port
              << " arena_mb=" << arena_mb << " map_slots=" << slots << "\n";

    try
    {
        PomaiServer server(&map, port);
        g_server = &server;

        // install signals for graceful shutdown
        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        server.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Pomai] fatal: " << e.what() << "\n";
        return 2;
    }
    catch (...)
    {
        std::cerr << "[Pomai] fatal: unknown exception\n";
        return 3;
    }

    std::cout << "[Pomai] exited\n";
    return 0;
}