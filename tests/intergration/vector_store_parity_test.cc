// tests/intergration/vector_store_parity_test.cc
//
// End-to-end integration test to verify VectorStore parity between single-map and sharded modes.
//
// - Upsert a small set of vectors
// - Search for nearest neighbors in both single and sharded modes and compare results
// - Remove one key and ensure it is removed in both modes
//
// Notes:
// - The test reduces per-shard arena requirement via pomai::config::runtime.arena_mb_per_shard
//   so ShardManager doesn't try to allocate very large arenas in CI.
// - For sharded mode, VectorStore routes upserts to PPSM (async). The test waits (poll)
//   on VectorStore::size() so the async workers finish processing before search assertions.
#include "src/ai/vector_store.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/shard_manager.h"
#include "src/core/config.h"

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>
#include <cassert>

using namespace pomai::ai;
using namespace pomai::memory;

// wait until vs.size() >= want or timeout_ms expires
static bool wait_for_size(VectorStore &vs, size_t want, uint64_t timeout_ms = 5000)
{
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() < (long long)timeout_ms)
    {
        if (vs.size() >= want)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

int main()
{
    std::cout << "[TEST] VectorStore parity (single-map vs sharded)\n";

    // Make shard manager allocate small arenas for tests
    pomai::config::runtime.arena_mb_per_shard = 4; // small for test

    // --- Single-mode setup ---
    PomaiArena arena = PomaiArena::FromMB(4);
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] arena allocation failed\n";
        return 1;
    }

    // Seed is declared at global scope (src/core/seed.h), not inside pomai::core, so use sizeof(Seed)
    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    if (slots == 0)
        slots = 1;

    PomaiMap map(&arena, slots);

    VectorStore vs_single;
    const size_t dim = 3;
    if (!vs_single.init(dim, 128, 4, 50, &arena))
    {
        std::cerr << "[FAIL] VectorStore::init single failed\n";
        return 2;
    }
    vs_single.attach_map(&map);

    // Points
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {10.0f, 0.0f, 0.0f};
    std::vector<float> c = {0.0f, 10.0f, 0.0f};

    if (!vs_single.upsert("a", 1, a.data()) ||
        !vs_single.upsert("b", 1, b.data()) ||
        !vs_single.upsert("c", 1, c.data()))
    {
        std::cerr << "[FAIL] upsert in single mode failed\n";
        return 3;
    }

    // simple checks for single mode
    {
        std::vector<float> q = {0.05f, -0.02f, 0.01f};
        auto res = vs_single.search(q.data(), dim, 1);
        if (res.empty() || res[0].first != "a")
        {
            std::cerr << "[FAIL] single: expected nearest 'a' got ";
            if (res.empty()) std::cerr << "empty\n"; else std::cerr << "'" << res[0].first << "'\n";
            return 4;
        }
    }

    // --- Sharded mode setup ---
    const uint32_t shard_count = 2;
    ShardManager shard_mgr(shard_count);

    VectorStore vs_sharded;
    // init with no arena (single-mode arena unused) — we'll attach shard manager after init
    if (!vs_sharded.init(dim, 256, 4, 50, /*arena=*/nullptr))
    {
        std::cerr << "[FAIL] VectorStore::init sharded failed\n";
        return 5;
    }
    vs_sharded.attach_shard_manager(&shard_mgr, shard_count);

    // Upserts in sharded mode (async). Keys should be routed to shards via shard router.
    if (!vs_sharded.upsert("a", 1, a.data()) ||
        !vs_sharded.upsert("b", 1, b.data()) ||
        !vs_sharded.upsert("c", 1, c.data()))
    {
        std::cerr << "[FAIL] upsert in sharded mode returned false\n";
        return 6;
    }

    // Wait for async inserts to be processed (poll size)
    if (!wait_for_size(vs_sharded, 3, 5000))
    {
        std::cerr << "[FAIL] sharded mode: inserts did not finish in time (size=" << vs_sharded.size() << ")\n";
        return 7;
    }

    // search in sharded mode
    {
        std::vector<float> q = {0.05f, -0.02f, 0.01f};
        auto res = vs_sharded.search(q.data(), dim, 1);
        if (res.empty())
        {
            std::cerr << "[FAIL] sharded: search returned empty\n";
            return 8;
        }
        // Compare name to single-mode result
        auto res_single = vs_single.search(q.data(), dim, 1);
        if (res_single.empty() || res_single[0].first != res[0].first)
        {
            std::cerr << "[FAIL] mismatch single vs sharded nearest: single='"
                      << (res_single.empty() ? "empty" : res_single[0].first)
                      << "' sharded='" << res[0].first << "'\n";
            return 9;
        }
    }

    // Test removal parity: single-mode remove
    if (!vs_single.remove("b", 1))
    {
        std::cerr << "[FAIL] single: remove b failed\n";
        return 10;
    }
    auto res_after = vs_single.search(b.data(), dim, 1);
    if (!res_after.empty() && res_after[0].first == "b")
    {
        std::cerr << "[FAIL] single: 'b' still present after remove\n";
        return 11;
    }

    // Sharded remove: ensure API routes to shard manager/PPSM remove
    if (!vs_sharded.remove("b", 1))
    {
        std::cerr << "[FAIL] sharded: remove(b) returned false\n";
        return 12;
    }

    // wait a little for async removal to take effect (implementation may be best-effort)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto res_sh_after = vs_sharded.search(b.data(), dim, 1);
    if (!res_sh_after.empty() && res_sh_after[0].first == "b")
    {
        std::cerr << "[FAIL] sharded: 'b' still present after remove\n";
        return 13;
    }

    std::cout << "[PASS] VectorStore parity tests OK\n";
    return 0;
}