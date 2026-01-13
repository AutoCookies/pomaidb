/*
 * src/main.cc
 *
 * Pomai Orbit Server v2 – Multi-membrance, multi-dim, fully concurrent
 * 2024 – True lựu server: tạo màng tự do, mỗi màng một phổi, INSERT/SEARCH/GET/DEL độc lập.
 */

#include "src/core/config.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/seed.h"
#include "src/core/pomai_db.h"
#include "src/facade/server.h"

#include <signal.h>
#include <sys/resource.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iomanip>

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

// Some ANSI coloring for banner
#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_CYAN "\033[36m"

void print_banner(int port)
{
    std::cout << ANSI_RED << R"(
  _____  ____  __  __          _____ 
 |  __ \|  _ \|  \/  |   /\   |_   _|   )"
              << ANSI_YELLOW << R"(  P O M A I   M E M B R A N C E)" << ANSI_RED << R"(
 | |__) | | | | \  / |  /  \    | |     )"
              << ANSI_RESET << "  Lựu server đa màng" << ANSI_RED << R"(
 |  ___/| |_| | |\/| | / /\ \   | |  
 | |    |____/| |  | |/ ____ \ _| |_ 
 |_|          |_|  |_/_/    \_\_____|   )"
              << ANSI_RESET << "  v2.0 (10/10 Multischema)" << "\n\n";
    std::cout << "  " << ANSI_GREEN << "PORT       : " << ANSI_RESET << port << "\n";
    std::cout << "  " << ANSI_GREEN << "PID        : " << ANSI_RESET << getpid() << "\n";
    std::cout << "  " << ANSI_CYAN << "-------------------------------------" << ANSI_RESET << "\n";
    std::cout << "  [PomaiDB] Ready - multi-membrance vector DB server.\n\n";
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

    // 3. Legacy KV Arena (để support SET/GET)
    uint64_t arena_mb = pomai::config::runtime.arena_mb_per_shard;
    if (arena_mb == 0)
        arena_mb = 64;
    pomai::memory::PomaiArena kv_arena = pomai::memory::PomaiArena::FromMB(64); // Small dedicated arena for KV keys
    if (!kv_arena.is_valid())
    {
        std::cerr << ANSI_RED << "[Fatal] KV Arena allocation failed (OOM?)" << ANSI_RESET << "\n";
        return 1;
    }

    uint64_t max_seeds = kv_arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    if (slots == 0)
        slots = 1;
    PomaiMap map(&kv_arena, slots);

    // RNG Init
    if (pomai::config::runtime.rng_seed.has_value())
        kv_arena.seed_rng(*pomai::config::runtime.rng_seed);
    else
        kv_arena.seed_rng(std::random_device{}());

    int port = static_cast<int>(pomai::config::runtime.default_port);

    try
    {
        // 4. Khởi tạo PomaiDB (đa màng lưu!)
        auto pomai_db = std::make_unique<pomai::core::PomaiDB>();

        // 4b. Tùy chọn: tạo màng mặc định để thuận tiện test
        pomai::core::MembranceConfig images_cfg;
        images_cfg.dim = 512;
        images_cfg.ram_mb = 256;
        pomai_db->create_membrance("images", images_cfg);

        pomai::core::MembranceConfig audio_cfg;
        audio_cfg.dim = 128;
        audio_cfg.ram_mb = 128;
        pomai_db->create_membrance("audio", audio_cfg);

        // 5. Start Server – wiring PomaiDB (multi-membrance) vào!
        PomaiServer server(&map, pomai_db.get(), port);

        print_banner(port);

        // 6. Signal Handlers
        struct sigaction sa{};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        // 7. Wait until signal
        std::unique_lock<std::mutex> lock(g_exit_mutex);
        g_exit_cv.wait(lock, []
                       { return g_stop_requested; });

        std::cout << "\n"
                  << ANSI_YELLOW << "[Pomai] Shutdown signal received. Cleaning up..." << ANSI_RESET << "\n";

        // 1) Stop network first (stop accepting new clients)
        server.stop();

        // 2) Request an explicit synchronous snapshot of DB (best-effort, catches exceptions)
        try
        {
            if (pomai_db)
            {
                bool ok_schemas = pomai_db->save_all_membrances();
                bool ok_manifest = pomai_db->save_manifest();
                if (ok_schemas && ok_manifest)
                {
                    std::cout << ANSI_GREEN << "[Pomai] Snapshot complete." << ANSI_RESET << "\n";
                }
                else
                {
                    std::cerr << ANSI_YELLOW << "[Pomai] Warning: snapshot completed with warnings. Schemas: "
                              << (ok_schemas ? "ok" : "fail") << ", manifest: " << (ok_manifest ? "ok" : "fail") << ANSI_RESET << "\n";
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << ANSI_RED << "[Pomai] Exception while snapshotting DB: " << e.what() << ANSI_RESET << "\n";
        }
        catch (...)
        {
            std::cerr << ANSI_RED << "[Pomai] Unknown exception while snapshotting DB" << ANSI_RESET << "\n";
        }
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