#include "server/config.h"
#include "server/logger.h"
#include "server/server.h"

#include <iostream>
#include <string>

static void PrintBanner()
{
    // ASCII Art: ANSI Shadow font
    std::cout << R"(
  ____  ____  __  __    _    ___ 
 |  _ \|  _ \|  \/  |  / \  |_ _|
 | |_) | | | | |\/| | / _ \  | | 
 |  __/| |_| | |  | |/ ___ \ | | 
 |_|   |____/|_|  |_/_/   \_\___|

 :: Pomai Vector DB ::   (v0.1.0-alpha)
    )" << std::endl;
}

static pomai::server::LogLevel ParseLevel(const std::string &s)
{
    using pomai::server::LogLevel;
    if (s == "debug")
        return LogLevel::debug;
    if (s == "info")
        return LogLevel::info;
    if (s == "warn")
        return LogLevel::warn;
    if (s == "error")
        return LogLevel::error;
    return LogLevel::info;
}

int main(int argc, char **argv)
{
    PrintBanner();

    std::string config_path = "config/pomai.yaml";

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc)
            config_path = argv[++i];
    }

    std::cout << "[init] Loading config from: " << config_path << std::endl;

    pomai::server::ServerConfig cfg;
    try
    {
        cfg = pomai::server::LoadConfigFile(config_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[error] Failed to load config: " << e.what() << "\n";
        return 1;
    }

    pomai::server::Logger log;
    log.SetFile(cfg.log_path);
    log.SetLevel(ParseLevel(cfg.log_level));

    pomai::server::PomaiServer server(cfg, &log);

    // Start blocking
    if (!server.Start())
        return 1;

    server.Stop();
    return 0;
}