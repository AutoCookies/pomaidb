#include "src/core/config.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/pomai_db.h"
#include "src/facade/server.h"
#include "src/core/cpu_kernels.h"

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <csignal>
#include <atomic>
#include <chrono>
#include <locale>
#include <stdexcept>
#include <unistd.h>
#include <sys/resource.h>

// Better global-state control for shutdown
namespace
{
    std::atomic<bool> g_running{true};
    std::condition_variable g_shutdown_cv;
    std::mutex g_shutdown_mu;
}

// Reliable and async-signal-safe signal handler toggling our running flag
extern "C" void signal_handler(int sig) noexcept
{
    // Only set atomic and notify_one (cannot lock mutex in handler safely)
    g_running.store(false, std::memory_order_release);
    g_shutdown_cv.notify_all();
}

// Set OS ulimits (with better return code checking)
void setup_os_limits()
{
    struct rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
    {
        rl.rlim_cur = std::min<rlim_t>(65536, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }
}

void print_banner(const pomai::config::PomaiConfig &cfg)
{
    std::cout << "\033[36m\033[1m" <<
        R"(
  ____  ___  __  __    _    ___
 |  _ \/ _ \|  \/  |  / \  |_ _|
 | |_) | | | | |\/| | / _ \  | |
 |  __/| |_| | |  | |/ ___ \ | |
 |_|    \___/|_|  |_/_/   \_\___|
)" << "\033[0m"
              << "\n [System Information]"
              << "\n  - Port:    " << cfg.net.port
              << "\n  - PID:     " << getpid()
              << "\n  - Storage: " << cfg.res.data_root
              << "\n ------------------------------------------------\n";
}

int main(int argc, char **argv)
try
{
    // Always set C locale (for float parsing and friends)
    std::setlocale(LC_ALL, "C");
    setup_os_limits();

    // Load config from CLI/env
    auto cfg = pomai::config::load_from_args(argc, argv);
    pomai_init_cpu_kernels();

    // Memory arena (fail hard if allocation fails)
    pomai::memory::PomaiArena kv_arena = pomai::memory::PomaiArena::FromMB(64);
    if (!kv_arena.is_valid())
    {
        std::cerr << "[Fatal] Failed to allocate Memory Arena\n";
        return 1;
    }

    // Map initialization (fail hard if config broken)
    uint64_t slots = cfg.map_tuning.default_slots;
    pomai::core::PomaiMap map(&kv_arena, slots, cfg);

    // Install signal handlers: block all during critical startup, then enable runtime-handle
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    std::unique_ptr<std::thread> metrics_worker;

    try
    {
        std::clog << "[Boot] Initializing Storage Engine at " << cfg.res.data_root << "...\n";
        auto pomai_db = std::make_unique<pomai::core::PomaiDB>(cfg);

        pomai::core::MembranceConfig default_cfg;
        default_cfg.dim = 128;
        pomai_db->create_membrance("default", default_cfg);

        std::clog << "[Boot] Starting Network Server on port " << cfg.net.port << "...\n";
        pomai::server::PomaiServer server(&map, pomai_db.get(), cfg);

        // If metrics enabled in config, run metrics printing thread
        if (cfg.metrics.enabled)
        {
            metrics_worker = std::make_unique<std::thread>([&cfg]()
                                                           {
                while (g_running.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.metrics.report_interval_ms));
                    if (!g_running.load(std::memory_order_acquire)) break;
                    PomaiMetrics::print_summary();
                } });
        }

        // Print banner only **AFTER** the server successfully binds the port
        print_banner(cfg);
        std::clog << "[System] Pomai is now ready for requests.\n";

        // Main service loop: wait for shutdown signal
        {
            std::unique_lock<std::mutex> lk(g_shutdown_mu);
            g_shutdown_cv.wait(lk, []()
                               { return !g_running.load(std::memory_order_acquire); });
        }

        std::clog << "[Shutdown] Initiating graceful stop...\n";

        server.stop();

        if (metrics_worker && metrics_worker->joinable())
        {
            metrics_worker->join();
        }

        pomai_db->save_all_membrances();
        pomai_db->save_manifest();
    }
    catch (const std::exception &e)
    {
        std::cerr << "\033[31m[Fatal Error] " << e.what() << "\033[0m\n";
        return 2;
    }

    std::cout << "[System] Shutdown complete. Resources released.\n";
    return 0;
}
catch (const std::exception &e)
{
    // Global fatal: uncaught exceptions crash the server and dump error
    std::cerr << "\033[31m[Fatal Crash] " << e.what() << "\033[0m\n";
    return 100;
}
catch (...)
{
    std::cerr << "\033[31m[Unknown Fatal Crash] Unhandled exception in main()\033[0m\n";
    return 101;
}