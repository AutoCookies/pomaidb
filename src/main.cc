/*
 * src/main.cc
 *
 * Entry point for Pomai Server (Hardcore Mode).
 *
 * Updates:
 * - Instantiates GlobalOrchestrator (The Engine).
 * - Wires the Orchestrator into PomaiServer (The Interface).
 * - Now VSET/VSEARCH/VGET/VDEL commands will reach the lock-free shards.
 */

#include "src/core/config.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/seed.h"
#include "src/facade/server.h"
#include "src/core/orchestrator.h"

#include <signal.h>
#include <sys/resource.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iomanip>

// Global synchronization for graceful shutdown
std::mutex g_exit_mutex;
std::condition_variable g_exit_cv;
bool g_stop_requested = false;

// Signal handler: Notifies the main thread immediately
static void on_signal(int /*signum*/)
{
    {
        std::lock_guard<std::mutex> lk(g_exit_mutex);
        g_stop_requested = true;
    }
    g_exit_cv.notify_one();
}

// ANSI Colors
#define ANSI_RESET  "\033[0m"
#define ANSI_BOLD   "\033[1m"
#define ANSI_RED    "\033[31m"
#define ANSI_GREEN  "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_CYAN   "\033[36m"

void print_banner(int port, uint64_t arena_mb, uint32_t shards)
{
    std::cout << ANSI_RED << R"(
  _____  ____  __  __          _____ 
 |  __ \|  _ \|  \/  |   /\   |_   _|   )" << ANSI_YELLOW << R"(  P O M A I   O R B I T)" << ANSI_RED << R"(
 | |__) | | | | \  / |  /  \    | |     )" << ANSI_RESET << "  The High-Performance Vector DB" << ANSI_RED << R"(
 |  ___/| |_| | |\/| | / /\ \   | |  
 | |    |____/| |  | |/ ____ \ _| |_ 
 |_|          |_|  |_/_/    \_\_____|   )" << ANSI_RESET << "  v2.0 (Hardcore Edition)" << "\n\n";

    std::cout << "  " << ANSI_GREEN << "PORT      : " << ANSI_RESET << port << "\n";
    std::cout << "  " << ANSI_GREEN << "PID       : " << ANSI_RESET << getpid() << "\n";
    std::cout << "  " << ANSI_GREEN << "SHARDS    : " << ANSI_RESET << shards << " (Lock-free/Atomic)" << "\n";
    std::cout << "  " << ANSI_GREEN << "ARENA MEM : " << ANSI_RESET << arena_mb << " MB/shard\n";
    std::cout << "  " << ANSI_GREEN << "MODE      : " << ANSI_RESET << ANSI_BOLD << "Shard-per-Core" << ANSI_RESET << "\n";
    std::cout << "  " << ANSI_CYAN << "-------------------------------------" << ANSI_RESET << "\n";
    std::cout << "  [Orchestrator] Online. Waiting for vectors...\n\n";
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
        if (want > rl.rlim_max) want = rl.rlim_max;
        rl.rlim_cur = want;
        setrlimit(RLIMIT_NOFILE, &rl);
    }

    // 3. Initialize Global KV Arena (Legacy Map Support)
    //    We still keep this small arena for the K-V Map metadata,
    //    while vectors go to the massive ShardArenas.
    uint64_t arena_mb = pomai::config::runtime.arena_mb_per_shard;
    if (arena_mb == 0) arena_mb = 64;

    pomai::memory::PomaiArena kv_arena = pomai::memory::PomaiArena::FromMB(64); // Small dedicated arena for KV keys
    if (!kv_arena.is_valid())
    {
        std::cerr << ANSI_RED << "[Fatal] KV Arena allocation failed (OOM?)" << ANSI_RESET << "\n";
        return 1;
    }

    uint64_t max_seeds = kv_arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds) slots <<= 1;
    if (slots == 0) slots = 1;

    // PomaiMap expects pomai::memory::PomaiArena*
    PomaiMap map(&kv_arena, slots);

    // RNG Init
    if (pomai::config::runtime.rng_seed.has_value())
        kv_arena.seed_rng(*pomai::config::runtime.rng_seed);
    else
        kv_arena.seed_rng(std::random_device{}());

    int port = static_cast<int>(pomai::config::runtime.default_port);

    try
    {
        // 4. Initialize GlobalOrchestrator (THE ENGINE)
        //    This spawns N shards, each with its own ShardArena (Lock-free) and PomaiOrbit.
        auto orchestrator = std::make_unique<pomai::core::GlobalOrchestrator>();

        // 5. Start Server - NOW WIRED UP!
        //    We pass the orchestrator pointer so the server can dispatch VSET/VSEARCH/VGET/VDEL.
        PomaiServer server(&map, orchestrator.get(), port);

        // Print nice banner AFTER successful bind
        print_banner(port, arena_mb, static_cast<uint32_t>(pomai::config::runtime.shard_count));

        // 6. Setup Signal Handlers
        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);  // Ctrl + C
        sigaction(SIGTERM, &sa, nullptr); // kill command

        // 7. Main thread waits efficiently
        std::unique_lock<std::mutex> lock(g_exit_mutex);
        g_exit_cv.wait(lock, [] { return g_stop_requested; });

        std::cout << "\n" << ANSI_YELLOW << "[Pomai] Shutdown signal received. Cleaning up..." << ANSI_RESET << "\n";

        server.stop();
        // orchestrator will be destroyed here when unique_ptr goes out of scope,
        // triggering a clean shutdown of all shards and arenas.
    }
    catch (const std::exception &e)
    {
        std::cerr << ANSI_RED << "[Fatal] Error starting server: " << e.what() << ANSI_RESET << "\n";
        return 2;
    }
    catch (...)
    {
        std::cerr << ANSI_RED << "[Fatal] Unknown exception occurred." << ANSI_RESET << "\n";
        return 3;
    }

    std::cout << ANSI_GREEN << "[Pomai] Bye bye! (Port released)" << ANSI_RESET << "\n";
    return 0;
}