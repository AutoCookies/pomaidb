#include "src/facade/sql_executor.h"
#include "src/ai/whispergrain.h"
#include "src/core/pomai_db.h"
#include "src/core/config.h"
#include "src/core/cpu_kernels.h"

#include <iostream>
#include <filesystem>
#include <string>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <exception>

using namespace pomai::server;
using namespace pomai::core;
using namespace pomai::ai;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

int main()
{
    pomai_init_cpu_kernels();

    std::string data_root = (std::filesystem::temp_directory_path() / ("pomai_test_server_dbg_" + std::to_string(std::rand()))).string();
    std::filesystem::create_directories(data_root);

    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = data_root;
    cfg.db.manifest_file = "manifest.txt";
    cfg.wal.sync_on_append = false;
    cfg.server.backlog = 4;
    cfg.server.max_events = 8;
    cfg.server.epoll_timeout_ms = 10;
    cfg.server.cpu_sample_interval_ms = 100;

    PomaiDB db(cfg);
    pomai::config::WhisperConfig wcfg;
    WhisperGrain whisper(wcfg);
    SqlExecutor exec;
    ClientState state;

    std::vector<std::string> cmds = {
        "CREATE MEMBRANCE test DIM 8;",
        "SHOW MEMBRANCES;",
        "USE test;",
        // single insert with numeric label
        std::string("INSERT INTO test VALUES (12345, [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5]);"),
        "GET MEMBRANCE INFO test;",
        // search: use same vector as inserted (expected to produce a result)
        std::string("SEARCH test QUERY (0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5) TOP 1;"),
        "DROP MEMBRANCE test;",
        "SHOW MEMBRANCES;"};

    for (size_t i = 0; i < cmds.size(); ++i)
    {
        const auto &cmd = cmds[i];
        std::cout << "[DBG] Executing command[" << i << "]: " << cmd << std::endl;
        try
        {
            std::string resp = exec.execute(&db, whisper, state, cmd);
            std::cout << "[DBG] Response length: " << resp.size() << "\n";
            // print first few lines for context
            std::istringstream iss(resp);
            std::string line;
            size_t printed = 0;
            while (printed < 8 && std::getline(iss, line))
            {
                if (line.empty())
                    continue;
                std::cout << "  " << line << "\n";
                ++printed;
            }
        }
        catch (const std::exception &ex)
        {
            std::cerr << ANSI_RED << "[ERROR] std::exception on command index " << i << ": " << ex.what() << ANSI_RESET << "\n";
        }
        catch (...)
        {
            std::cerr << ANSI_RED << "[ERROR] unknown exception / crash on command index " << i << ANSI_RESET << "\n";
        }
        // small pause to reduce races in background threads
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cout << "[DBG] Done. Cleaning up data root: " << data_root << std::endl;
    std::error_code ec;
    std::filesystem::remove_all(data_root, ec);
    return 0;
}