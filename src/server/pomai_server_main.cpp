#include <pomai/server/server.h>
#include <pomai/server/config.h>
#include <pomai/util/logger.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstring>

// This main provides a simple, safe signal handling pattern:
// - Start PomaiServer in a background thread (server.Start() blocks).
// - Install SIGINT / SIGTERM handler that sets an atomic flag.
// - Main thread waits for the flag, then calls server.Stop() to perform
//   a graceful shutdown (ensures WAL durability).
//
// Note: calling complex or non-async-signal-safe APIs from a signal handler
// is unsafe, so the handler only flips an atomic flag. The main thread reacts
// and calls server.Stop() on behalf of the handler.

using namespace std::chrono_literals;

static std::atomic_bool g_terminate{false};

static void SignalHandler(int /*signum*/)
{
    // Only set the atomic flag — safe from signal handler context.
    g_terminate.store(true);
}

static void InstallSignalHandlers()
{
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = SignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // Ignore SIGPIPE so socket send() doesn't kill the process.
    std::signal(SIGPIPE, SIG_IGN);
}

// Simple helper to load server config.
// If parsing a YAML/config file is already implemented in your codebase, replace
// this with your existing loader. For now we provide a fallback with sane defaults.
static pomai::server::ServerConfig LoadConfigOrDefault(const std::string &path)
{
    pomai::server::ServerConfig cfg;

    // Set defaults first (these should match your project's defaults).
    cfg.data_dir = "./data";
    cfg.listen_host = "127.0.0.1";
    cfg.listen_port = 7744;
    cfg.unix_socket = "/tmp/pomai.sock";
    cfg.shards = 4;
    cfg.shard_queue_capacity = 65536;
    cfg.default_dim = 128;

    // NEW: default server-level policy for allowing per-append synchronous fdatasync
    cfg.allow_sync_on_append = true;

    if (std::filesystem::exists(path))
    {
        // If you have a YAML config parser in-tree, use it instead.
        std::cout << "[init] Loading config from: " << path << "\n";
        // TODO: hook into your real parser here, e.g. cfg = ParseConfig(path);
    }
    else
    {
        std::cout << "[init] Config file not found: " << path << " — using defaults\n";
    }

    std::cout << "[init] allow_sync_on_append = " << (cfg.allow_sync_on_append ? "true" : "false") << "\n";

    return cfg;
}

int main(int argc, char **argv)
{
    std::string cfg_path = "config/pomai.yaml";
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc)
        {
            cfg_path = argv[++i];
        }
        else if (!arg.empty() && arg[0] != '-')
        {
            cfg_path = arg;
        }
    }

    // Install signal handlers early so we can react to Ctrl+C during init.
    InstallSignalHandlers();

    // Load configuration (replace with your real loader if available).
    auto cfg = LoadConfigOrDefault(cfg_path);

    // Create a logger. Replace with your project's logger construction if it's different.
    // We assume server::Logger has a constructor that accepts no args or simple options.
    pomai::Logger logger;
    if (!cfg.log_path.empty())
        logger.SetFile(cfg.log_path);
    if (cfg.log_level == "debug")
    {
        logger.SetLevel(pomai::LogLevel::debug);
        logger.EnableDebug(true);
    }
    else if (cfg.log_level == "info")
        logger.SetLevel(pomai::LogLevel::info);
    else if (cfg.log_level == "warn")
        logger.SetLevel(pomai::LogLevel::warn);
    else if (cfg.log_level == "error")
        logger.SetLevel(pomai::LogLevel::error);

    std::cout << R"(

  ____  ____  __  __    _    ___ 
 |  _ \|  _ \|  \/  |  / \  |_ _|
 | |_) | | | | |\/| | / _ \  | | 
 |  __/| |_| | |  | |/ ___ \ | | 
 |_|   |____/|_|  |_/_/   \_\___|

 :: Pomai Vector DB ::   (v0.1.0-alpha)

)" << std::endl;

    logger.Info("server.init", "Starting Pomai server");

    // Construct server
    pomai::server::PomaiServer server(cfg, &logger);

    // Start server in background thread because server.Start() blocks (it runs AcceptLoop()).
    std::thread server_thread([&server]()
                              {
        try
        {
            server.Start();
        }
        catch (const std::exception &e)
        {
            // In case Start throws, print to stderr and set terminate flag so main can exit.
            std::cerr << "[fatal] server.Start() threw: " << e.what() << "\n";
            g_terminate.store(true);
        } });

    // Wait until a termination signal arrives.
    while (!g_terminate.load())
    {
        std::this_thread::sleep_for(200ms);
    }

    // Termination requested: perform graceful shutdown.
    logger.Info("server.shutdown", "Shutdown signal received, stopping server...");
    server.Stop();

    // Join server thread; server.Start() should return after Stop().
    if (server_thread.joinable())
        server_thread.join();

    logger.Info("server.shutdown", "Server stopped, exiting.");
    return 0;
}
