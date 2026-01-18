/*
 * src/main.cc
 * Pomai Orbit Server - Fully Refactored with Centralized Config
 *
 * Updates:
 * [FIXED] Force "C" Locale to ensure float parsing works correctly (dot vs comma).
 * [UI] Enhanced startup banner.
 */

#include "src/core/config.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/seed.h"
#include "src/core/pomai_db.h"
#include "src/facade/server.h"
#include "src/core/cpu_kernels.h"

#include <signal.h>
#include <sys/resource.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <cstdlib>
#include <clocale> // [ADDED] For setlocale

std::mutex g_exit_mutex;
std::condition_variable g_exit_cv;
bool g_stop_requested = false;

static void on_signal(int /*signum*/)
{
    {
        std::lock_guard<std::mutex> lk(g_exit_mutex);
        g_stop_requested = true;
    }
    g_exit_cv.notify_one();
}

#define ANSI_RESET "\033[0m"
#define ANSI_GREEN "\033[32m"
#define ANSI_RED "\033[31m"
#define ANSI_CYAN "\033[36m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_BOLD "\033[1m"

void print_banner(int port)
{
    std::cout << ANSI_CYAN << ANSI_BOLD;
    std::cout << R"(
  ____  ___  __  __    _    ___ 
 |  _ \/ _ \|  \/  |  / \  |_ _|
 | |_) | | | | |\/| | / _ \  | | 
 |  __/| |_| | |  | |/ ___ \ | | 
 |_|    \___/|_|  |_/_/   \_\___|
                                 
)" << ANSI_RESET;
    std::cout << " ------------------------------------------------\n";
    std::cout << "  Port:   " << port << "\n";
    std::cout << "  PID:    " << getpid() << "\n";
    std::cout << "  Locale: " << ANSI_YELLOW << "C (Standard)" << ANSI_RESET << "\n";
    std::cout << " ------------------------------------------------\n";
}

int main(int argc, char **argv)
{
    // [CRITICAL Fix] Force standard C locale.
    // This ensures strtof/stof parses "3.14" as 3.14, not 3 (if system locale expects comma).
    std::setlocale(LC_ALL, "C");

    // 1. Load Config (Duy nhất 1 lần từ CLI/Env)
    auto config = pomai::config::load_from_args(argc, argv);

    // Print shard / arena info on startup as requested
    {
        size_t shard_count = (config.orchestrator.shard_count > 0)
                                 ? static_cast<size_t>(config.orchestrator.shard_count)
                                 : std::max<size_t>(1, std::thread::hardware_concurrency());
        uint64_t arena_mb = config.res.arena_mb_per_shard;
        uint64_t total_arena_mb = arena_mb * shard_count;
        std::clog << "[Init] Shard configuration: shard_count=" << shard_count
                  << ", arena_mb_per_shard=" << arena_mb << " MB"
                  << " (total ~" << total_arena_mb << " MB)"
                  << "\n";
    }

    // 2. Init CPU Kernels
    pomai_init_cpu_kernels();
    
    // 3. System tuning
    struct rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
    {
        rl.rlim_cur = std::min<rlim_t>(65536, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }

    // 4. Memory Arena & Key-Value Map
    // Dùng 64MB cho KV store metadata
    pomai::memory::PomaiArena kv_arena = pomai::memory::PomaiArena::FromMB(64);
    if (!kv_arena.is_valid())
    {
        std::cerr << ANSI_RED << "[Fatal] Arena allocation failed" << ANSI_RESET << "\n";
        return 1;
    }

    uint64_t slots = 1024 * 1024; // 1M slots
    // TRUYỀN config vào Map
    pomai::core::PomaiMap map(&kv_arena, slots, config);

    // Seed RNG
    if (config.rng_seed.has_value())
    {
        kv_arena.seed_rng(*config.rng_seed);
    }
    else
    {
        kv_arena.seed_rng(std::random_device{}());
    }

    if (config.metrics.enabled)
    {
        std::thread([&config]()
                    {
            std::this_thread::sleep_for(std::chrono::seconds(2)); 
            
            while (!g_stop_requested) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config.metrics.report_interval_ms));
                
                if (g_stop_requested) break;
            
                PomaiMetrics::print_summary();
            } })
            .detach();
    }

    print_banner(config.net.port);
    std::clog << "[Init] CPU kernels initialized\n";

    try
    {
        // 5. Khởi tạo PomaiDB (TRUYỀN config vào DB)
        auto pomai_db = std::make_unique<pomai::core::PomaiDB>(config);

        // Tạo màng mặc định nếu cần
        pomai::core::MembranceConfig default_cfg;
        default_cfg.dim = 128;
        pomai_db->create_membrance("default", default_cfg);

        // 6. Start Server (TRUYỀN config vào Server)
        PomaiServer server(&map, pomai_db.get(), config);

        // Signal Handlers
        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        // Wait until signal
        std::unique_lock<std::mutex> lock(g_exit_mutex);
        g_exit_cv.wait(lock, []
                       { return g_stop_requested; });

        std::cout << "\n"
                  << ANSI_GREEN << "[Pomai] Shutdown signal received..." << ANSI_RESET << "\n";
        server.stop();
        pomai_db->save_all_membrances();
        pomai_db->save_manifest();
    }
    catch (const std::exception &e)
    {
        std::cerr << ANSI_RED << "[Fatal] " << e.what() << ANSI_RESET << "\n";
        return 2;
    }

    std::cout << ANSI_GREEN << "[Pomai] Bye!" << ANSI_RESET << "\n";
    return 0;
}