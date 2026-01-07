#include <iostream>
#include <cstdint>
#include <string>
#include "src/memory/arena.h"
#include "src/core/seed.h"

// Minimal test harness (no GTest). Exit codes:
// 0 = all OK
// non-zero = failure

int main()
{
    std::cout << "[TEST] Arena: FromMB and exhaustion\n";

    PomaiArena arena = PomaiArena::FromMB(1); // 1 MB arena
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena::FromMB(1) returned invalid arena\n";
        return 1;
    }

    uint64_t capacity = arena.get_capacity_bytes();
    if (capacity == 0)
    {
        std::cerr << "[FAIL] arena.get_capacity_bytes() == 0\n";
        return 2;
    }

    uint64_t max_seeds = capacity / sizeof(Seed);
    if (max_seeds == 0)
    {
        std::cerr << "[FAIL] computed max_seeds == 0\n";
        return 3;
    }

    // allocate until exhaustion
    uint64_t count = 0;
    while (true)
    {
        Seed *s = arena.alloc_seed();
        if (!s)
            break;
        ++count;
    }

    if (arena.num_active_seeds() != count)
    {
        std::cerr << "[FAIL] num_active_seeds mismatch: " << arena.num_active_seeds()
                  << " != " << count << "\n";
        return 4;
    }

    if (count != max_seeds)
    {
        std::cerr << "[WARN] number allocated (" << count << ") != theoretical max (" << max_seeds << ")\n";
        // Not fatal: alignment/rounding could affect count
    }

    if (arena.alloc_seed() != nullptr)
    {
        std::cerr << "[FAIL] alloc_seed succeeded after expected exhaustion\n";
        return 5;
    }

    std::cout << "[PASS] Arena basic exhaustion test OK (allocated " << count << " seeds)\n";
    return 0;
}