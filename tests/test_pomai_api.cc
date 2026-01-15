// tests/test_pomai_api.cc
// Simple in-process test that exercises PomaiDB / PomaiOrbit APIs directly.
// - Creates a PomaiDB instance (data_root = ./data_test_api)
// - Creates a membrance
// - Inserts a small batch via PomaiDB::insert_batch
// - Runs search/get/delete via PomaiDB APIs and prints results
//
// Build (example):
//   g++ -std=c++17 -I./src -pthread tests/test_pomai_api.cc \
//       -o test_pomai_api <linker_flags_for_project_objects_and_libs>
//
// Run:
//   ./test_pomai_api
//
// Note: adjust include/library paths to match your build system.

#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "src/core/pomai_db.h"
#include "src/core/config.h"
#include "src/core/cpu_kernels.h"

int main(int argc, char **argv)
{
    using namespace std::chrono;
    try
    {
        // Initialize CPU kernels (some components rely on kernel selection).
        pomai_init_cpu_kernels();

        // Optional: initialize config from env/args if your app relies on it.
        // pomai::config::init_from_env();
        // pomai::config::init_from_args(argc, argv);

        std::cout << "[test] Starting in-process PomaiDB API test\n";

        // Create a PomaiDB instance using an isolated data root for tests.
        std::string data_root = "./data_test_api";
        pomai::core::PomaiDB db(data_root);

        // Membrance parameters
        const std::string membr_name = "api_test_membr";
        const size_t dim = 128;
        const size_t ram_mb = 64;

        pomai::core::MembranceConfig cfg;
        cfg.dim = dim;
        cfg.ram_mb = ram_mb;
        cfg.enable_metadata_index = false; // not needed for this test

        std::cout << "[test] Creating membrance '" << membr_name << "' dim=" << dim << " ram_mb=" << ram_mb << "\n";
        bool ok_create = db.create_membrance(membr_name, cfg);
        if (!ok_create)
        {
            std::cerr << "[test] create_membrance failed (exists or error). Continuing if it already exists.\n";
        }

        // Confirm membrance available and query dimension
        size_t got_dim = db.get_membrance_dim(membr_name);
        std::cout << "[test] get_membrance_dim -> " << got_dim << "\n";
        if (got_dim != dim)
        {
            std::cerr << "[test] Warning: expected dim=" << dim << " got=" << got_dim << "\n";
        }

        // Prepare a small batch of random vectors
        const size_t n = 8;
        std::vector<std::pair<uint64_t, std::vector<float>>> batch;
        batch.reserve(n);
        std::mt19937_64 rng(123456);
        std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
        for (size_t i = 0; i < n; ++i)
        {
            uint64_t label = 1000 + static_cast<uint64_t>(i); // simple numeric labels
            std::vector<float> v;
            v.reserve(dim);
            for (size_t d = 0; d < dim; ++d)
                v.push_back(ud(rng));
            batch.emplace_back(label, std::move(v));
        }

        std::cout << "[test] Inserting batch of " << batch.size() << " vectors via PomaiDB::insert_batch...\n";
        bool ok_batch = db.insert_batch(membr_name, batch);
        std::cout << "[test] insert_batch -> " << (ok_batch ? "ok" : "failed") << "\n";

        // Ask orbit (via db) for membrance info if available
        auto *m = db.get_membrance(membr_name);
        if (m && m->orbit)
        {
            try
            {
                auto info = m->orbit->get_info();
                std::cout << "[test] membrance get_info: dim=" << info.dim
                          << " num_vectors=" << info.num_vectors
                          << " disk_bytes=" << info.disk_bytes
                          << " disk_gb=" << std::fixed << std::setprecision(4) << info.disk_gb() << "\n";
            }
            catch (const std::exception &e)
            {
                std::cerr << "[test] orbit->get_info() threw: " << e.what() << "\n";
            }
            catch (...)
            {
                std::cerr << "[test] orbit->get_info() unknown exception\n";
            }
        }

        // Search: use the first inserted vector as a probe
        if (!batch.empty())
        {
            const std::vector<float> &probe = batch[0].second;
            std::cout << "[test] Running search for TOP 5 using the first inserted vector as probe...\n";
            auto results = db.search(membr_name, probe.data(), 5);
            std::cout << "[test] search returned " << results.size() << " results\n";
            for (size_t i = 0; i < results.size(); ++i)
            {
                std::cout << "  rank=" << i << " id=" << results[i].first << " dist=" << results[i].second << "\n";
            }
        }

        // GET: retrieve a stored vector by label (first label)
        if (!batch.empty())
        {
            uint64_t want_label = batch[0].first;
            std::vector<float> out;
            bool ok_get = db.get(membr_name, want_label, out);
            std::cout << "[test] get(label=" << want_label << ") -> " << (ok_get ? "found" : "not found") << "\n";
            if (ok_get)
            {
                std::cout << "[test] vector (first 8 dims):";
                for (size_t i = 0; i < std::min<size_t>(8, out.size()); ++i)
                    std::cout << " " << out[i];
                std::cout << "\n";
            }
        }

        // DELETE: remove all inserted labels
        std::cout << "[test] Removing inserted labels...\n";
        for (const auto &it : batch)
        {
            bool ok_rm = db.remove(membr_name, it.first);
            std::cout << "  remove " << it.first << " -> " << (ok_rm ? "ok" : "failed") << "\n";
        }

        // Search again to verify deleted items are excluded (best-effort)
        if (!batch.empty())
        {
            const std::vector<float> &probe = batch[0].second;
            auto results2 = db.search(membr_name, probe.data(), 5);
            std::cout << "[test] search after deletes returned " << results2.size() << " results\n";
            for (auto &p : results2)
                std::cout << "  id=" << p.first << " dist=" << p.second << "\n";
        }

        // Final membrance info
        if (m && m->orbit)
        {
            try
            {
                auto info = m->orbit->get_info();
                std::cout << "[test] final membrance get_info: dim=" << info.dim
                          << " num_vectors=" << info.num_vectors
                          << " disk_bytes=" << info.disk_bytes << "\n";
            }
            catch (...)
            { /* ignore */
            }
        }

        std::cout << "[test] Done.\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[test] uncaught exception: " << e.what() << "\n";
        return 2;
    }
    catch (...)
    {
        std::cerr << "[test] unknown uncaught exception\n";
        return 3;
    }
}