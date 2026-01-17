/*
 * src/main.cc
 * Pomai Orbit Server - Fully Refactored with Centralized Config
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

void print_banner(int port)
{
    std::cout << ANSI_CYAN << "POMAI MEMBRANCE SERVER" << ANSI_RESET << "\n";
    std::cout << "PORT: " << port << " | PID: " << getpid() << "\n";
    std::cout << "-------------------------------------\n";
}

int main(int argc, char **argv)
{
    // 1. Load Config (Duy nhất 1 lần từ CLI/Env)
    auto config = pomai::config::load_from_args(argc, argv);

    // 2. Init CPU Kernels
    pomai_init_cpu_kernels();
    std::clog << "[Init] CPU kernel selected: " << kernel_name_from_ptr(get_pomai_l2sq_kernel()) << "\n";

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