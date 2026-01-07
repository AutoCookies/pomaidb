// tests/test_stress_churn.cc
//
// Heavier stress test that performs many randomized insert/delete operations
// to exercise:
//  - blob allocation/free paths
//  - seed allocation exhaustion & harvest path
//  - arena freelist reuse under churn
//
// Usage (env vars optional):
//  ARENA_MB (default 64) - arena size in MB
//  INITIAL_KEYS (default 10000) - initial keys to insert
//  OPS (default 50000) - number of random operations (insert/delete) to perform
//
// This test prints periodic progress and final metrics. It's intentionally "heavy" but tunable.
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <unordered_set>
#include <chrono>
#include <cstdlib>

#include "memory/arena.h"
#include "core/map.h"
#include "core/metrics.h"

static uint32_t rand_size(std::mt19937 &rng, uint32_t min_b = 16, uint32_t max_b = 16384)
{
    std::uniform_int_distribution<uint32_t> dist(min_b, max_b);
    return dist(rng);
}

int main()
{
    const uint64_t arena_mb = std::getenv("ARENA_MB") ? std::strtoull(std::getenv("ARENA_MB"), nullptr, 10) : 64;
    const uint32_t initial_keys = std::getenv("INITIAL_KEYS") ? std::atoi(std::getenv("INITIAL_KEYS")) : 10000;
    const uint32_t ops = std::getenv("OPS") ? std::atoi(std::getenv("OPS")) : 50000;

    std::cout << "[STRESS] arena_mb=" << arena_mb << " INITIAL_KEYS=" << initial_keys << " OPS=" << ops << "\n";

    PomaiArena arena = PomaiArena::FromMB(arena_mb);
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;

    PomaiMap map(&arena, slots);

    std::mt19937 rng(1234567);
    std::uniform_int_distribution<int> op_dist(0, 1); // 0=delete, 1=insert

    std::vector<std::string> live_keys;
    live_keys.reserve(initial_keys * 2);

    uint64_t key_counter = 0;

    auto make_key = [&](uint64_t id)
    {
        return "stress_k_" + std::to_string(id);
    };

    // Initial population
    std::cout << "[STRESS] Initial inserting " << initial_keys << " keys...\n";
    for (uint32_t i = 0; i < initial_keys; ++i)
    {
        std::string key = make_key(key_counter++);
        uint32_t vlen = rand_size(rng, 32, 4096);
        std::string val;
        val.resize(vlen);
        for (uint32_t j = 0; j < vlen; ++j)
            val[j] = 'a' + (j % 26);
        bool ok = map.put(key.c_str(), key.size(), val.data(), val.size());
        if (ok)
            live_keys.push_back(std::move(key));
        // else ignore (harvest path might be invoked)
        if ((i & 0xFFF) == 0)
            std::cout << "." << std::flush;
    }
    std::cout << "\n[STRESS] initial done, live_keys=" << live_keys.size() << "\n";

    // Perform random churn
    std::cout << "[STRESS] Performing churn ops...\n";
    auto t0 = std::chrono::steady_clock::now();
    uint32_t progress_step = std::max<uint32_t>(1, ops / 20);
    for (uint32_t i = 0; i < ops; ++i)
    {
        int op = op_dist(rng);
        if (op == 0 && !live_keys.empty())
        {
            // Delete random existing key
            std::uniform_int_distribution<size_t> idx_dist(0, live_keys.size() - 1);
            size_t idx = idx_dist(rng);
            const std::string key = live_keys[idx];
            map.erase(key.c_str());
            // swap-pop to remove
            live_keys[idx] = live_keys.back();
            live_keys.pop_back();
        }
        else
        {
            // Insert new key
            std::string key = make_key(key_counter++);
            uint32_t vlen = rand_size(rng, 16, 16384); // varied sizes
            std::string val;
            val.resize(vlen);
            for (uint32_t j = 0; j < vlen; ++j)
                val[j] = 'A' + (j % 26);
            bool ok = map.put(key.c_str(), key.size(), val.data(), val.size());
            if (ok)
                live_keys.push_back(std::move(key));
        }

        if ((i & (progress_step - 1)) == 0)
        {
            std::cout << "[" << i << "/" << ops << "] live=" << live_keys.size()
                      << " puts=" << PomaiMetrics::puts.load()
                      << " arena_alloc_fails=" << PomaiMetrics::arena_alloc_fails.load()
                      << "\n";
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[STRESS] churn completed in " << elapsed_s << "s\n";
    std::cout << "[METRICS] hits=" << PomaiMetrics::hits.load()
              << " misses=" << PomaiMetrics::misses.load()
              << " puts=" << PomaiMetrics::puts.load()
              << " evictions=" << PomaiMetrics::evictions.load()
              << " harvests=" << PomaiMetrics::harvests.load()
              << " arena_alloc_fails=" << PomaiMetrics::arena_alloc_fails.load()
              << "\n";

    // Basic sanity checks
    if (PomaiMetrics::arena_alloc_fails.load() > 0)
    {
        std::cout << "[WARN] arena_alloc_fails > 0 (some blob allocs failed under stress)\n";
    }
    else
    {
        std::cout << "[PASS] No arena allocation failures observed\n";
    }

    std::cout << "[STRESS] remaining live keys: " << live_keys.size() << "\n";

    return 0;
}