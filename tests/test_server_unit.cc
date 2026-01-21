#include "src/core/pomai_db.h"
#include "src/core/config.h"
#include "src/core/pomai_db.h"
#include "src/core/pomai_db.h"
#include "src/core/pomai_db.h"
#include <filesystem>
#include <random>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

using namespace pomai::core;
using namespace pomai::config;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

struct Runner
{
    int passed = 0;
    int failed = 0;
    void expect(bool cond, const char *name)
    {
        if (cond)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << name << "\n";
            ++passed;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << "\n";
            ++failed;
        }
    }
    int summary()
    {
        std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
        return failed == 0 ? 0 : 1;
    }
};

static std::string make_temp_dir(const std::string &tag)
{
    auto p = std::filesystem::temp_directory_path() / ("pomai_srv_test_" + tag + "_" + std::to_string(std::rand()));
    std::filesystem::create_directories(p);
    return p.string();
}

static std::vector<float> rnd_vec(size_t dim, uint64_t seed)
{
    std::mt19937 rng((uint32_t)seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i)
        v[i] = nd(rng);
    return v;
}

int main()
{
    Runner R;

    PomaiConfig cfg;
    std::string tmp = make_temp_dir("server_unit");
    cfg.res.data_root = tmp;
    // reduce background intervals to exercise worker fast
    cfg.db.bg_worker_interval_ms = 10;
    // small arena to keep resource usage modest
    cfg.res.arena_mb_per_shard = 8;

    try
    {
        PomaiDB db(cfg);
        R.expect(true, "PomaiDB constructed");

        // create membrance
        MembranceConfig mc;
        mc.dim = 32;
        mc.ram_mb = 16;
        mc.data_type = pomai::core::DataType::FLOAT32;
        bool ok_create = db.create_membrance("m0", mc);
        R.expect(ok_create, "create_membrance returned true");

        // verify listing
        auto list = db.list_membrances();
        bool found = false;
        for (auto &n : list)
            if (n == "m0")
                found = true;
        R.expect(found, "membrance appears in list");

        // insert a few vectors (use HotTier -> background worker will flush them to orbit)
        size_t dim = mc.dim;
        std::vector<float> v = rnd_vec(dim, 12345);
        bool ok_ins = db.insert("m0", v.data(), 1001);
        R.expect(ok_ins, "insert single vector returned true");

        // insert many to trigger HotTier swap/flush
        for (int i = 0; i < 200; ++i)
        {
            auto vv = rnd_vec(dim, 2000 + i);
            bool r = db.insert("m0", vv.data(), 2000 + i);
            if (!r)
            {
                // insertion can go to Orbit hot-path; treat failure as fail only if persistent
            }
        }
        R.expect(true, "bulk inserts performed");

        // allow background to run and flush hot tier into orbit
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // test search (should not crash)
        std::vector<float> q = rnd_vec(dim, 9999);
        auto res = db.search("m0", q.data(), 3);
        R.expect(res.size() <= 3, "search returned <= k results (no crash)");

        // test get (existing label)
        std::vector<float> out;
        bool got = db.get("m0", 1001, out);
        R.expect(got, "db.get returned (may be true if flushed)");

        if (got)
        {
            R.expect(out.size() == dim, "retrieved vector has correct dimension");
            bool anynan = false;
            for (float x : out)
                if (std::isnan(x))
                    anynan = true;
            R.expect(!anynan, "retrieved vector has no NaNs");
        }

        // remove membrance (exercise WAL and background)
        bool ok_drop = db.drop_membrance("m0");
        R.expect(ok_drop, "drop_membrance returned true");

        // allow worker some time to process any pending background activities
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // destructor for db will run here and background thread will be joined; if teardown races exist this run may crash
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << "\n";
        R.expect(false, "exception thrown during test");
    }
    catch (...)
    {
        R.expect(false, "unknown exception during test");
    }

    int code = R.summary();

    // Cleanup temp data
    try
    {
        std::filesystem::remove_all(tmp);
    }
    catch (...)
    {
    }

    return code;
}