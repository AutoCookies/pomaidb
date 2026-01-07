#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include "memory/arena.h"
#include "core/map.h"

// Helper to compute next power of two >= v
static uint64_t next_power_of_two(uint64_t v)
{
    if (v == 0)
        return 1;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return ++v;
}

// Test harvest path without GTest. Returns non-zero on failure.
int main()
{
    std::cout << "[TEST] Map Harvest: fill arena then insert extra key\n";

    PomaiArena arena = PomaiArena::FromMB(1); // 1 MB
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena allocation failed\n";
        return 1;
    }

    // Determine how many unique seeds fit into the arena
    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    if (max_seeds == 0)
    {
        std::cerr << "[FAIL] max_seeds == 0\n";
        return 2;
    }

    // Ensure the hash table has at least max_seeds slots (rounded up to pow2)
    const uint64_t map_slots = next_power_of_two(max_seeds);
    std::cout << "[INFO] map_slots = " << map_slots << ", max_seeds = " << max_seeds << "\n";

    PomaiMap map(&arena, map_slots);

    // Insert distinct keys to consume the arena
    for (uint64_t i = 0; i < max_seeds; ++i)
    {
        std::ostringstream k, v;
        k << "k_" << i;
        v << "v_" << i;
        bool ok = map.put(k.str().c_str(), v.str().c_str());
        if (!ok)
        {
            std::cerr << "[FAIL] put failed at i=" << i << "\n";
            return 3;
        }
    }

    // Ensure arena reports exhaustion
    if (arena.alloc_seed() != nullptr)
    {
        std::cerr << "[FAIL] expected arena exhausted but alloc_seed() returned non-null\n";
        return 4;
    }

    // Insert a new key; harvest should allow insertion
    const char *newk = "new_key_XYZ";
    const char *newv = "new_value";
    bool ok = map.put(newk, newv);
    if (!ok)
    {
        std::cerr << "[FAIL] put(new_key) failed (harvest did not succeed)\n";
        return 5;
    }

    const char *got = map.get(newk);
    if (!got)
    {
        std::cerr << "[FAIL] get(new_key) returned nullptr after put\n";
        return 6;
    }
    if (std::string(got) != std::string(newv))
    {
        std::cerr << "[FAIL] get(new_key) returned unexpected value: '" << got << "'\n";
        return 7;
    }

    std::cout << "[PASS] Map harvest test OK; new key inserted and retrievable\n";
    return 0;
}