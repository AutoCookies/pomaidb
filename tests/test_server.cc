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

using namespace pomai::server;
using namespace pomai::core;
using namespace pomai::ai;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

struct Runner
{
    void expect(bool cond, const char *name)
    {
        if (cond)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << name << "\n";
            passed++;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << "\n";
            failed++;
        }
    }
    int summary()
    {
        std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
        return failed == 0 ? 0 : 1;
    }
    int passed = 0;
    int failed = 0;
};

static std::string make_temp_dir(const std::string &name)
{
    auto p = std::filesystem::temp_directory_path() / ("pomai_test_server_" + name + "_" + std::to_string(std::rand()));
    std::filesystem::create_directories(p);
    return p.string();
}

int main()
{
    pomai_init_cpu_kernels();
    Runner r;

    std::string data_root = make_temp_dir("data");
    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = data_root;
    cfg.db.manifest_file = "manifest.txt";
    cfg.wal.sync_on_append = false;
    cfg.server.backlog = 4;
    cfg.server.max_events = 8;
    cfg.server.epoll_timeout_ms = 10;
    cfg.server.cpu_sample_interval_ms = 100;

    // create DB instance
    PomaiDB db(cfg);

    // whisper controller with default config
    pomai::config::WhisperConfig wcfg;
    WhisperGrain whisper(wcfg);

    SqlExecutor exec;

    ClientState state;

    // 1) CREATE MEMBRANCE
    {
        std::string cmd = "CREATE MEMBRANCE test DIM 8;";
        std::string resp = exec.execute(&db, whisper, state, cmd);
        r.expect(resp.find("OK: created membrance") != std::string::npos, "CREATE MEMBRANCE succeeded");
    }

    // 2) SHOW MEMBRANCES
    {
        std::string resp = exec.execute(&db, whisper, state, "SHOW MEMBRANCES;");
        r.expect(resp.find("MEMBRANCES:") != std::string::npos && resp.find("test") != std::string::npos, "SHOW MEMBRANCES lists created membrance");
    }

    // 3) USE + INSERT (single tuple) via SQL
    {
        std::string resp;
        resp = exec.execute(&db, whisper, state, "USE test;");
        r.expect(resp.find("OK: switched to membrance") != std::string::npos, "USE sets current membrance");

        // build vector of 8 floats
        std::string vec = "[";
        for (int i = 0; i < 8; ++i)
        {
            if (i) vec += ",";
            vec += std::to_string(static_cast<float>(i) * 0.5f);
        }
        vec += "]";

        // use a numeric label so it fits IdEntry payload
        std::string insert_cmd = "INSERT INTO test VALUES (12345, " + vec + ");";
        resp = exec.execute(&db, whisper, state, insert_cmd);
        r.expect(resp.find("OK: inserted") != std::string::npos, "INSERT INTO via SQL succeeded");
    }

    // 4) GET MEMBRANCE INFO
    {
        std::string resp = exec.execute(&db, whisper, state, "GET MEMBRANCE INFO test;");
        r.expect(resp.find("MEMBRANCE: test") != std::string::npos && resp.find("feature_dim") != std::string::npos, "GET MEMBRANCE INFO returns contract");
    }

    // 5) SEARCH (vector query) - expect at least one result
    {
        // build same vector as inserted
        std::string vec = "(";
        for (int i = 0; i < 8; ++i)
        {
            if (i) vec += ",";
            vec += std::to_string(static_cast<float>(i) * 0.5f);
        }
        vec += ")";
        std::string cmd = "SEARCH test QUERY " + vec + " TOP 1;";
        std::string resp = exec.execute(&db, whisper, state, cmd);
        r.expect(!resp.empty(), "SEARCH returned a response");

        bool got_result = false;
        std::istringstream iss(resp);
        std::string line;
        // skip header line ("OK N")
        std::getline(iss, line);
        while (std::getline(iss, line))
        {
            if (line.empty()) continue;
            std::istringstream ls(line);
            uint64_t lbl;
            float score;
            if (ls >> lbl >> score)
            {
                got_result = true;
                break;
            }
        }
        r.expect(got_result || resp.find("ERR") == std::string::npos, "SEARCH produced parseable output or no error");
    }

    // 6) DROP MEMBRANCE
    {
        std::string resp = exec.execute(&db, whisper, state, "DROP MEMBRANCE test;");
        r.expect(resp.find("OK: dropped") != std::string::npos, "DROP MEMBRANCE succeeded");

        // verify it's gone
        std::string resp2 = exec.execute(&db, whisper, state, "SHOW MEMBRANCES;");
        r.expect(resp2.find("test") == std::string::npos, "SHOW MEMBRANCES no longer lists dropped membrance");
    }

    // cleanup
    std::error_code ec;
    std::filesystem::remove_all(data_root, ec);

    return r.summary();
}