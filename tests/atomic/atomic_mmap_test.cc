#include "src/ai/atomic_utils.h"

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>
#include <chrono>

// Small stress test that exercises atomic_store/atomic_load helpers on shared mmap-like memory.
// Writer writes a 64-bit payload then publishes a 32-bit flag. Readers observe flag then read payload.
// We assert that no reader sees a published flag with a zero payload (i.e., publish ordering holds).

static const size_t ITER = 200000;

int main()
{
    // allocate an anonymous shared mapping large enough for one u64 + one u32 (aligned)
    const size_t SZ = 4096;
    void *map = mmap(nullptr, SZ, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    if (map == MAP_FAILED)
    {
        perror("mmap");
        return 2;
    }
    std::memset(map, 0, SZ);

    uint64_t *payload = reinterpret_cast<uint64_t *>(map);
    uint32_t *flag = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(map) + sizeof(uint64_t));
    static_assert(sizeof(uint64_t) % alignof(uint64_t) == 0, "alignment");

    std::atomic<bool> stop{false};
    std::atomic<size_t> readers_observed_inconsistency{0};

    // writer thread: repeatedly publish non-zero payload followed by setting flag
    auto writer = std::thread([&]()
                              {
        for (size_t i = 1; i <= ITER; ++i)
        {
            uint64_t v = (i << 1) | 1ULL; // odd non-zero value
            pomai::ai::atomic_utils::atomic_store_u64(payload, v);
            pomai::ai::atomic_utils::atomic_store_u32(flag, 1u);
            // small pause then clear for next iteration
            // clear flag then payload to simulate updates
            // Use release ordering to clear in same pattern: clear flag first then clear payload to 0
            pomai::ai::atomic_utils::atomic_store_u32(flag, 0u);
            pomai::ai::atomic_utils::atomic_store_u64(payload, 0ULL);
        }
        stop.store(true, std::memory_order_release); });

    // multiple reader threads
    const size_t RCOUNT = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    std::vector<std::thread> readers;
    for (size_t r = 0; r < RCOUNT; ++r)
    {
        readers.emplace_back([&]()
                             {
            while (!stop.load(std::memory_order_acquire))
            {
                uint32_t f = pomai::ai::atomic_utils::atomic_load_u32(flag);
                if (f != 0)
                {
                    uint64_t p = pomai::ai::atomic_utils::atomic_load_u64(payload);
                    if (p == 0)
                    {
                        // observed inconsistent state
                        readers_observed_inconsistency.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                // spin-yield a bit
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            } });
    }

    writer.join();

    // Give readers a bit to notice final state
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true, std::memory_order_release);

    for (auto &t : readers)
        t.join();

    size_t bad = readers_observed_inconsistency.load(std::memory_order_relaxed);
    if (bad == 0)
    {
        std::cout << "PASS: no torn reads observed\n";
        munmap(map, SZ);
        return 0;
    }
    else
    {
        std::cout << "FAIL: observed " << bad << " inconsistent reads\n";
        munmap(map, SZ);
        return 1;
    }
}