/*
 * tests/test_pomai_orbit.cc
 *
 * Production-Grade Unit Test for PomaiOrbit.
 * Corrected to match Config-based API.
 */

#include "src/ai/pomai_orbit.h"
#include "src/memory/arena.h"
#include "src/core/config.h"
#include "src/core/cpu_kernels.h"

#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <thread>
#include <atomic>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace fs = std::filesystem;
using namespace pomai::ai::orbit;
using namespace pomai::memory;

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

class TestRunner
{
public:
    void expect(bool condition, const std::string &test_name)
    {
        if (condition)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << test_name << "\n";
            passed_++;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << test_name << "\n";
            failed_++;
        }
    }
    int summary()
    {
        std::cout << "\nResults: " << passed_ << " passed, " << failed_ << " failed.\n";
        return failed_ == 0 ? 0 : 1;
    }

private:
    int passed_ = 0;
    int failed_ = 0;
};

// Helper: Create Arena using proper Config API
static PomaiArena create_test_arena(uint64_t bytes, const std::string &path)
{
    pomai::config::PomaiConfig cfg;
    cfg.arena.remote_dir = path;
    uint64_t mb = bytes / (1024 * 1024);
    if (mb == 0)
        mb = 1;
    cfg.res.arena_mb_per_shard = static_cast<uint32_t>(mb);
    return PomaiArena(cfg);
}

static std::string create_temp_dir(const std::string &suffix)
{
    auto path = fs::temp_directory_path() / ("pomai_test_" + suffix + "_" + std::to_string(std::rand()));
    if (fs::exists(path))
        fs::remove_all(path);
    fs::create_directories(path);
    return path.string();
}

static std::vector<float> make_random_vec(size_t dim, uint64_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i)
        v[i] = nd(rng);
    return v;
}

void test_basic_lifecycle(TestRunner &runner)
{
    std::string data_path = create_temp_dir("lifecycle");
    size_t dim = 128;
    PomaiArena arena = create_test_arena(64 * 1024 * 1024, data_path);

    PomaiOrbit::Config cfg;
    cfg.dim = dim;
    cfg.data_path = data_path;
    cfg.use_cortex = false;
    cfg.algo.num_centroids = 16;

    {
        PomaiOrbit orbit(cfg, &arena);
        runner.expect(orbit.num_centroids() == 0, "Initially empty centroids");
        std::vector<float> train_data;
        for (int i = 0; i < 200; ++i)
        {
            auto v = make_random_vec(dim, i);
            train_data.insert(train_data.end(), v.begin(), v.end());
        }
        bool trained = orbit.train(train_data.data(), 200);
        runner.expect(trained, "Training success");
        runner.expect(orbit.num_centroids() > 0, "Centroids created");
    }
    fs::remove_all(data_path);
}

void test_insert_and_get_integrity(TestRunner &runner)
{
    std::string data_path = create_temp_dir("integrity");
    size_t dim = 64;
    PomaiArena arena = create_test_arena(64 * 1024 * 1024, data_path);

    PomaiOrbit::Config cfg;
    cfg.dim = dim;
    cfg.data_path = data_path;
    cfg.use_cortex = false;
    cfg.eeq_cfg.quantize_scales = true;
    cfg.eeq_cfg.bits_per_layer = {8, 8, 8, 8};

    PomaiOrbit orbit(cfg, &arena);
    std::vector<float> dummy(dim * 100);
    orbit.train(dummy.data(), 100);

    std::vector<float> vec_in = make_random_vec(dim, 999);
    uint64_t label = 12345;
    bool ins = orbit.insert(vec_in.data(), label);
    runner.expect(ins, "Insert single vector");

    std::vector<float> vec_out;
    bool got = orbit.get(label, vec_out);
    runner.expect(got, "Get vector by label");

    if (got)
    {
        runner.expect(vec_out.size() == dim, "Retrieved dimension matches");
        bool has_nan = false;
        for (float x : vec_out)
            if (std::isnan(x))
                has_nan = true;
        runner.expect(!has_nan, "Retrieved vector has NO NaNs");
    }
    fs::remove_all(data_path);
}

void test_persistence_recovery(TestRunner &runner)
{
    std::string data_path = create_temp_dir("persistence");
    size_t dim = 32;
    size_t count = 500;

    {
        PomaiArena arena = create_test_arena(32 * 1024 * 1024, data_path);
        PomaiOrbit::Config cfg;
        cfg.dim = dim;
        cfg.data_path = data_path;
        cfg.use_cortex = false;

        PomaiOrbit orbit(cfg, &arena);
        std::vector<std::pair<uint64_t, std::vector<float>>> batch;
        for (size_t i = 0; i < count; ++i)
        {
            batch.push_back({i + 1000, make_random_vec(dim, i)});
        }
        orbit.insert_batch(batch);
        orbit.checkpoint();
        auto info = orbit.get_info();
        runner.expect(info.num_vectors == count, "Inserted vectors count correct before close");
    }

    {
        PomaiArena arena = create_test_arena(32 * 1024 * 1024, data_path);
        PomaiOrbit::Config cfg;
        cfg.dim = dim;
        cfg.data_path = data_path;
        cfg.use_cortex = false;

        PomaiOrbit orbit(cfg, &arena);
        auto info = orbit.get_info();
        runner.expect(info.num_vectors == count, "Recovered vector count matches");

        std::vector<float> out;
        bool found_first = orbit.get(1000, out);
        bool found_last = orbit.get(1000 + count - 1, out);
        runner.expect(found_first && found_last, "Recovered specific IDs");
    }
    fs::remove_all(data_path);
}

void test_search_accuracy(TestRunner &runner)
{
    std::string data_path = create_temp_dir("search");
    size_t dim = 32;
    PomaiArena arena = create_test_arena(32 * 1024 * 1024, data_path);
    PomaiOrbit::Config cfg;
    cfg.dim = dim;
    cfg.data_path = data_path;
    cfg.use_cortex = false;
    cfg.eeq_cfg.bits_per_layer = {8, 8, 8, 8};

    PomaiOrbit orbit(cfg, &arena);
    std::vector<float> target = make_random_vec(dim, 777);
    orbit.insert(target.data(), 8888);

    for (int i = 0; i < 100; ++i)
    {
        auto noise = make_random_vec(dim, i);
        orbit.insert(noise.data(), i);
    }

    auto results = orbit.search(target.data(), 5, 2);
    bool found_target = false;
    for (const auto &p : results)
    {
        if (p.first == 8888)
            found_target = true;
    }
    runner.expect(found_target, "Search found exact match (ID 8888)");
    runner.expect(!results.empty(), "Search returned results");
    fs::remove_all(data_path);
}

int main()
{
    pomai_init_cpu_kernels();
    TestRunner runner;
    try
    {
        test_basic_lifecycle(runner);
        test_insert_and_get_integrity(runner);
        test_persistence_recovery(runner);
        test_search_accuracy(runner);
    }
    catch (const std::exception &e)
    {
        std::cerr << ANSI_RED << "FATAL EXCEPTION: " << e.what() << ANSI_RESET << "\n";
        return 1;
    }
    return runner.summary();
}