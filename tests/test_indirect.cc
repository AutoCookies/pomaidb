#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include "src/memory/arena.h"
#include "src/core/map.h"

// Simple test that verifies indirect/blob storage path (no GTest).
// Inserts a key with a value larger than Seed::payload capacity and verifies retrieval.

int main()
{
    std::cout << "[TEST] Indirect/blob path (no-gtest)\n";

    // Small arena (1 MB) for fast test
    PomaiArena arena = PomaiArena::FromMB(1);
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena allocation failed\n";
        return 1;
    }

    // Make map with enough slots to avoid table-full early
    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    if (max_seeds == 0)
    {
        std::cerr << "[FAIL] arena too small for any seeds\n";
        return 2;
    }

    // Round up to power of two
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;

    PomaiMap map(&arena, slots);

    // Prepare a large value (> Seed payload minus key length)
    std::string bigv;
    bigv.reserve(200);
    for (int i = 0; i < 200; ++i)
        bigv.push_back(char('A' + (i % 26)));

    const char *key = "indirect_key";
    bool ok = map.put(key, bigv.c_str());
    if (!ok)
    {
        std::cerr << "[FAIL] put for indirect value failed\n";
        return 3;
    }

    const char *got = map.get(key);
    if (!got)
    {
        std::cerr << "[FAIL] get returned null for indirect key\n";
        return 4;
    }

    if (std::string(got) != bigv)
    {
        std::cerr << "[FAIL] indirect value mismatch. got len=" << strlen(got) << "\n";
        return 5;
    }

    std::cout << "[PASS] Indirect/blob path OK (key='" << key << "', value_len=" << bigv.size() << ")\n";
    return 0;
}