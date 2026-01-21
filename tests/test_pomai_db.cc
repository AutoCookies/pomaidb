#include "src/core/pomai_db.h"
#include "src/core/config.h"
#include "src/core/types.h"
#include "src/core/cpu_kernels.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace pomai::core;

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
    auto p = std::filesystem::temp_directory_path() / ("pomai_test_db_" + name + "_" + std::to_string(std::rand()));
    std::filesystem::create_directories(p);
    return p.string();
}

int main()
{
    pomai_init_cpu_kernels();
    Runner r;

    std::string data_root = make_temp_dir("pomai_db");
    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = data_root;
    cfg.db.manifest_file = "manifest.txt";
    cfg.wal.sync_on_append = false;

    // Scope to ensure destructor runs and files are flushed
    {
        PomaiDB db(cfg);

        // create membrance
        MembranceConfig mc;
        mc.dim = 16;
        mc.ram_mb = 16;
        mc.data_type = pomai::core::DataType::FLOAT32;

        bool created = db.create_membrance("test_mem", mc);
        r.expect(created, "create_membrance returns true");

        auto list = db.list_membrances();
        bool found = false;
        for (auto &n : list)
            if (n == "test_mem")
                found = true;
        r.expect(found, "list_membrances contains new membrance");

        size_t dim = db.get_membrance_dim("test_mem");
        r.expect(dim == mc.dim, "get_membrance_dim returns configured dim");

        // insert a batch (use insert_batch which writes directly to orbit)
        std::vector<float> v(dim);
        for (size_t i = 0; i < dim; ++i)
            v[i] = static_cast<float>(i) * 0.5f;
        std::vector<std::pair<uint64_t, std::vector<float>>> batch;
        batch.emplace_back(12345ull, v);
        bool ib = db.insert_batch("test_mem", batch);
        r.expect(ib, "insert_batch returns true");

        // retrieve inserted vector
        std::vector<float> out;
        bool got = db.get("test_mem", 12345ull, out);
        r.expect(got, "get returns true for inserted ID");
        if (got)
        {
            r.expect(out.size() == dim, "retrieved vector length matches dim");
            bool ok = true;
            for (size_t i = 0; i < dim; ++i)
            {
                if (std::isnan(out[i]))
                {
                    ok = false;
                    break;
                }
            }
            r.expect(ok, "retrieved vector contains no NaNs");
        }
    }

    // New DB instance should load manifest and expose the membrance entry
    {
        PomaiDB db2(cfg);
        auto list2 = db2.list_membrances();
        bool found2 = false;
        for (auto &n : list2)
            if (n == "test_mem")
                found2 = true;
        r.expect(found2, "manifest load: new PomaiDB instance contains membrance from manifest");
    }

    // cleanup
    std::error_code ec;
    std::filesystem::remove_all(data_root, ec);

    return r.summary();
}