#include "pomai/server/config.h"
#include "pomai/server/logger.h"
#include "pomai/server/server.h"

#include <iostream>
#include <string>

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
    std::string config_path = "config/pomai.yaml";

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc)
            config_path = argv[++i];
    }

    pomai::server::ServerConfig cfg;
    try
    {
        cfg = pomai::server::LoadConfigFile(config_path);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    pomai::server::Logger log;
    log.SetFile(cfg.log_path);
    log.SetLevel(ParseLevel(cfg.log_level));

    pomai::server::PomaiServer server(cfg, &log);
    if (!server.Start())
        return 1;

    server.Stop();
    return 0;
}
