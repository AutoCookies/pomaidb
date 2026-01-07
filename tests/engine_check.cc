#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <cstddef>
#include <sstream>

#include "src/core/map.h"
#include "src/memory/arena.h"
#include "src/core/metrics.h"

/*
  Demo program:
  - creates a PomaiArena (1 MB) and PomaiMap
  - demonstrates storing a large (>48B) value (indirect/blob)
  - seeds arena RNG for deterministic sampling
  - fills arena until alloc_seed() is exhausted
  - performs a put to trigger harvest and shows deterministic victim selection
*/

int main()
{
    std::cout << "[Pomai] Engine Initialization (demo)...\n";

    PomaiArena arena = PomaiArena::FromMB(1); // 1 MB for quick demo
    if (!arena.is_valid())
    {
        std::cerr << "Arena allocate failed\n";
        return 2;
    }

    // seed RNG for deterministic behavior in get_random_seed()
    arena.seed_rng(12345);

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    std::cout << "Arena capacity bytes: " << arena.get_capacity_bytes() << ", max_seeds: " << max_seeds << "\n";

    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    PomaiMap map(&arena, slots);

    // 1) Demonstrate indirect/blob storage
    std::string bigv;
    for (int i = 0; i < 200; ++i)
        bigv.push_back(char('a' + (i % 26)));
    const char *bigkey = "big:key";
    std::cout << "[Demo] Storing big value (len=" << bigv.size() << ") under key '" << bigkey << "'\n";
    bool ok = map.put(bigkey, bigv.c_str());
    if (!ok)
    {
        std::cerr << "put(bigkey) failed\n";
    }
    else
    {
        const char *got = map.get(bigkey);
        if (got && std::string(got) == bigv)
        {
            std::cout << "  - big value retrieval OK (len=" << strlen(got) << ")\n";
        }
        else
        {
            std::cerr << "  - big value retrieval FAILED\n";
        }
    }

    // 2) Show deterministic RNG sequence (addresses of random seeds)
    std::cout << "[Demo] Deterministic RNG sample (first 5 random seed addresses):\n";
    for (int i = 0; i < 5; ++i)
    {
        Seed *s = arena.get_random_seed();
        std::cout << "  - sample[" << i << "] = " << s << "\n";
    }

    // 3) Fill arena with many small keys to exhaust alloc_seed() (use small inline values)
    std::cout << "[Demo] Filling arena with small keys to exhaust alloc_seed()...\n";
    uint64_t inserted = 0;
    for (uint64_t i = 0; i < max_seeds; ++i)
    {
        std::ostringstream k, v;
        k << "k_fill_" << i;
        v << "val" << i;
        if (!map.put(k.str().c_str(), v.str().c_str()))
        {
            // If put fails because harvest triggered, continue until alloc_seed exhausted
        }
        else
        {
            ++inserted;
        }
        // probe by checking num_active_seeds instead of calling alloc_seed
        if (arena.num_active_seeds() >= max_seeds)
        {
            break;
        }
    }

    std::cout << "Inserted (approx) small keys: " << inserted << ", num_active_seeds()=" << arena.num_active_seeds() << "\n";
    if (arena.alloc_seed() == nullptr)
    {
        std::cout << "Arena appears exhausted (alloc_seed() -> nullptr)\n";
    }
    else
    {
        std::cout << "Arena still has space for seeds\n";
    }

    // 4) Trigger harvest deterministically: because we seeded RNG earlier, the victim choice should be reproducible.
    std::cout << "[Demo] Triggering harvest: inserting key 'harvest_me' with small value\n";
    const char *hkey = "harvest_me";
    const char *hval = "hv";
    bool put_ok = map.put(hkey, hval);
    if (!put_ok)
    {
        std::cerr << "put(harvest_me) failed\n";
    }
    else
    {
        const char *hgot = map.get(hkey);
        if (hgot)
        {
            std::cout << "  - harvest inserted key OK (value='" << hgot << "')\n";
            // find the seed via the map (safe for inline and indirect)
            Seed *s = map.find_seed(hkey, strlen(hkey));
            std::cout << "  - harvest stored at seed addr: " << s << "\n";
        }
        else
        {
            std::cerr << "  - harvest insertion yielded no retrievable value\n";
        }
    }

    // Print metrics
    std::cout << "[Metrics] hits=" << PomaiMetrics::hits.load()
              << " misses=" << PomaiMetrics::misses.load()
              << " puts=" << PomaiMetrics::puts.load()
              << " evictions=" << PomaiMetrics::evictions.load()
              << " harvests=" << PomaiMetrics::harvests.load()
              << " arena_alloc_fails=" << PomaiMetrics::arena_alloc_fails.load()
              << "\n";

    std::cout << "[Pomai] Demo complete.\n";
    return 0;
}