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
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

using namespace pomai::server;
using namespace pomai::core;
using namespace pomai::ai;

static void print_backtrace_and_die(int sig)
{
    void *buf[64];
    int n = backtrace(buf, sizeof(buf) / sizeof(buf[0]));
    std::cerr << "Signal " << sig << " received. Backtrace (" << n << " frames):\n";
    backtrace_symbols_fd(buf, n, STDERR_FILENO);
    // flush and abort to get core
    std::cerr.flush();
    _exit(128 + sig);
}

static std::string make_temp_dir(const std::string &name)
{
    auto p = std::filesystem::temp_directory_path() / ("pomai_test_server_diag_" + name + "_" + std::to_string(std::rand()));
    std::filesystem::create_directories(p);
    return p.string();
}

int main()
{
    signal(SIGSEGV, print_backtrace_and_die);
    signal(SIGABRT, print_backtrace_and_die);

    pomai_init_cpu_kernels();

    std::string data_root = make_temp_dir("data");
    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = data_root;
    cfg.db.manifest_file = "manifest.txt";
    cfg.wal.sync_on_append = false;
    cfg.server.backlog = 4;
    cfg.server.max_events = 8;
    cfg.server.epoll_timeout_ms = 10;
    cfg.server.cpu_sample_interval_ms = 100;

    std::cerr << "[diag] data_root = " << data_root << "\n";

    PomaiDB db(cfg);

    pomai::config::WhisperConfig wcfg;
    WhisperGrain whisper(wcfg);

    SqlExecutor exec;
    ClientState state;

    try
    {
        std::string resp;

        resp = exec.execute(&db, whisper, state, "CREATE MEMBRANCE test DIM 8;");
        std::cerr << "[diag] CREATE -> " << resp << "\n";

        resp = exec.execute(&db, whisper, state, "SHOW MEMBRANCES;");
        std::cerr << "[diag] SHOW -> " << resp << "\n";

        resp = exec.execute(&db, whisper, state, "USE test;");
        std::cerr << "[diag] USE -> " << resp << "\n";

        std::string vec = "[";
        for (int i = 0; i < 8; ++i)
        {
            if (i)
                vec += ",";
            vec += std::to_string(static_cast<float>(i) * 0.5f);
        }
        vec += "]";

        std::string insert_cmd = "INSERT INTO test VALUES (12345, " + vec + ");";
        resp = exec.execute(&db, whisper, state, insert_cmd);
        std::cerr << "[diag] INSERT -> " << resp << "\n";

        resp = exec.execute(&db, whisper, state, "GET MEMBRANCE INFO test;");
        std::cerr << "[diag] GET INFO -> " << resp << "\n";

        // SEARCH: use parentheses style previously used in tests
        std::string qvec = "(";
        for (int i = 0; i < 8; ++i)
        {
            if (i)
                qvec += ",";
            qvec += std::to_string(static_cast<float>(i) * 0.5f);
        }
        qvec += ")";
        std::string search_cmd = "SEARCH test QUERY " + qvec + " TOP 1;";
        resp = exec.execute(&db, whisper, state, search_cmd);
        std::cerr << "[diag] SEARCH -> size=" << resp.size() << " bytes\n";
        std::cerr << resp << "\n";

        resp = exec.execute(&db, whisper, state, "DROP MEMBRANCE test;");
        std::cerr << "[diag] DROP -> " << resp << "\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "[diag] exception: " << e.what() << "\n";
    }
    catch (...)
    {
        std::cerr << "[diag] unknown exception\n";
    }

    // cleanup
    std::error_code ec;
    std::filesystem::remove_all(data_root, ec);

    std::cerr << "[diag] finished normally\n";
    return 0;
}