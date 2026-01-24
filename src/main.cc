#include "src/core/config.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/pomai_db.h"
#include "src/facade/server.h"
#include "src/core/cpu_kernels.h"

#include <signal.h>
#include <sys/resource.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <clocale>
#include <random>
#include <atomic>
#include <iomanip>
#include <unistd.h> // for getpid()

static std::mutex g_exit_mu;
static std::condition_variable g_exit_cv;
static std::atomic<bool> g_stop_requested{false};

static void sig_handler(int)
{
    g_stop_requested.store(true, std::memory_order_release);
    g_exit_cv.notify_one();
}

static void print_banner(uint16_t port)
{
    std::cout << "\033[36m\033[1m";
    std::cout <<
        R"(
  ____  ___  __  __    _    ___ 
 |  _ \/ _ \|  \/  |  / \  |_ _|
 | |_) | | | | |\/| | / _ \  | | 
 |  __/| |_| | |  | |/ ___ \ | | 
 |_|    \___/|_|  |_/_/   \_\___|
)";
    std::cout << "\033[0m\n";
    std::cout << " ------------------------------------------------\n";
    std::cout << "  Port:   " << port << "\n";
    std::cout << "  PID:    " << getpid() << "\n";
    std::cout << "  Locale: " << std::setw(10) << "C (forced)" << "\n";
    std::cout << " ------------------------------------------------\n";
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "C");
    auto cfg = pomai::config::load_from_args(argc, argv);

    pomai_init_cpu_kernels();

    struct rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
    {
        rl.rlim_cur = std::min<rlim_t>(65536, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }

    pomai::memory::PomaiArena kv_arena = pomai::memory::PomaiArena::FromMB(64);
    if (!kv_arena.is_valid())
    {
        std::cerr << "\033[31m[Fatal] Arena allocation failed\033[0m\n";
        return 1;
    }

    uint64_t slots = cfg.map_tuning.default_slots;
    pomai::core::PomaiMap map(&kv_arena, slots, cfg);

    if (cfg.rng_seed.has_value())
        kv_arena.seed_rng(*cfg.rng_seed);
    else
        kv_arena.seed_rng(std::random_device{}());

    if (cfg.metrics.enabled)
    {
        std::thread([](pomai::config::PomaiConfig cfg)
                    {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            while (!g_stop_requested.load(std::memory_order_acquire))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.metrics.report_interval_ms));
                if (g_stop_requested.load(std::memory_order_acquire)) break;
                PomaiMetrics::print_summary();
            } }, cfg)
            .detach();
    }

    print_banner(cfg.net.port);

    try
    {
        auto pomai_db = std::make_unique<pomai::core::PomaiDB>(cfg);

        pomai::core::MembranceConfig default_cfg;
        default_cfg.dim = 128;
        pomai_db->create_membrance("default", default_cfg);

        // Use fully-qualified name so we don't rely on 'using namespace' in header
        pomai::server::PomaiServer server(&map, pomai_db.get(), cfg);

        struct sigaction sa{};
        sa.sa_handler = sig_handler;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        {
            std::unique_lock<std::mutex> lk(g_exit_mu);
            g_exit_cv.wait(lk, []
                           { return g_stop_requested.load(std::memory_order_acquire); });
        }

        server.stop();
        pomai_db->save_all_membrances();
        pomai_db->save_manifest();
    }
    catch (const std::exception &e)
    {
        std::cerr << "\033[31m[Fatal] " << e.what() << "\033[0m\n";
        return 2;
    }
    catch (...)
    {
        std::cerr << "\033[31m[Fatal] Unknown error\033[0m\n";
        return 3;
    }

    std::cout << "\033[32m[Pomai] Shutdown complete. Bye!\033[0m\n";
    return 0;
}