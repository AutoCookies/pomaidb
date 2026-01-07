#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/core/seed.h"
#include "src/ai/vector_store.h"

// Minimal test harness for VectorStore
// Exit codes: 0 = pass, non-zero = fail

static bool approx_equal(float a, float b, float eps = 1e-4f)
{
    return std::fabs(a - b) <= eps;
}

int main()
{
    std::cout << "[TEST] VectorStore basic upsert/search\n";

    // Small arena for tests
    pomai::memory::PomaiArena arena = pomai::memory::PomaiArena::FromMB(4);
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    if (slots == 0)
        slots = 1;

    PomaiMap map(&arena, slots);

    // Create VectorStore
    pomai::ai::VectorStore vs;
    const size_t dim = 3;
    if (!vs.init(dim, 128, 8, 10, &arena))
    {
        std::cerr << "[FAIL] VectorStore::init failed\n";
        return 2;
    }
    vs.attach_map(&map);

    // Points: a=(0,0,0), b=(10,0,0), c=(0,10,0)
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {10.0f, 0.0f, 0.0f};
    std::vector<float> c = {0.0f, 10.0f, 0.0f};

    if (!vs.upsert("a", 1, a.data()))
    {
        std::cerr << "[FAIL] upsert a failed\n";
        return 3;
    }
    if (!vs.upsert("b", 1, b.data()))
    {
        std::cerr << "[FAIL] upsert b failed\n";
        return 4;
    }
    if (!vs.upsert("c", 1, c.data()))
    {
        std::cerr << "[FAIL] upsert c failed\n";
        return 5;
    }

    // Search near a
    std::vector<float> q1 = {0.1f, -0.05f, 0.02f};
    auto res1 = vs.search(q1.data(), dim, 1);
    if (res1.empty())
    {
        std::cerr << "[FAIL] search returned empty\n";
        return 6;
    }
    if (res1[0].first != "a")
    {
        std::cerr << "[FAIL] nearest to q1 expected 'a' got '" << res1[0].first << "'\n";
        return 7;
    }

    // Search near b
    std::vector<float> q2 = {9.2f, 0.1f, 0.0f};
    auto res2 = vs.search(q2.data(), dim, 1);
    if (res2.empty() || res2[0].first != "b")
    {
        std::cerr << "[FAIL] nearest to q2 expected 'b' got ";
        if (res2.empty())
            std::cerr << "empty\n";
        else
            std::cerr << "'" << res2[0].first << "'\n";
        return 8;
    }

    // Search near c for top2 and verify 'c' and near neighbor present
    std::vector<float> q3 = {-0.2f, 9.9f, 0.0f};
    auto res3 = vs.search(q3.data(), dim, 2);
    if (res3.size() < 1)
    {
        std::cerr << "[FAIL] search top2 returned insufficient\n";
        return 9;
    }
    if (res3[0].first != "c")
    {
        std::cerr << "[FAIL] nearest to q3 expected 'c' got '" << res3[0].first << "'\n";
        return 10;
    }

    std::cout << "[PASS] VectorStore basic tests OK\n";
    return 0;
}