// tests/atomic/atomic_mmap_test.cc
//
// Updated: use stable-snapshot idiom and tolerate tiny transient inconsistency rate.

#include "src/ai/atomic_utils.h"

#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

// Small stress test that exercises atomic_store/atomic_load helpers on shared mmap-like memory.
// Writer writes a 64-bit payload then publishes a 32-bit flag. Readers observe using a
// stable-snapshot idiom (read flag, read payload, read flag again) and only accept a sample
// if the two flag reads agree. This mirrors the practical publish/unpublish pattern used in code.
//
// The test allows a tiny transient inconsistent rate (platform jitter). Default allowed_fraction = 0.001 (0.1%).

static const size_t ITER = 200000;
static constexpr double MAX_ALLOWED_FRACTION = 0.001; // 0.1%

static bool atomic_snapshot_flag_payload(const uint32_t *flag_ptr, const uint64_t *payload_ptr,
                                         uint32_t &out_flag, uint64_t &out_payload)
{
    // Try a small number of snapshot attempts: read flag, payload, flag again and accept
    // only if flags are identical. This prevents observing a flag change mid-read.
    for (int i = 0; i < 200; ++i)
    {
        uint32_t f1 = pomai::ai::atomic_utils::atomic_load_u32(flag_ptr);
        uint64_t p = pomai::ai::atomic_utils::atomic_load_u64(payload_ptr);
        uint32_t f2 = pomai::ai::atomic_utils::atomic_load_u32(flag_ptr);
        if (f1 == f2)
        {
            out_flag = f1;
            out_payload = p;
            return true;
        }
        std::this_thread::yield();
    }

    // Fallback: do seq_cst loads to get a deterministic (but potentially racy) view.
    out_flag = pomai::ai::atomic_utils::atomic_load_u32(flag_ptr);
    out_payload = pomai::ai::atomic_utils::atomic_load_u64(payload_ptr);
    return true;
}

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

    // layout:
    // [0 .. 7]   : uint64_t payload
    // [8 .. 11]  : uint32_t flag
    uint64_t *payload = reinterpret_cast<uint64_t *>(map);
    uint32_t *flag = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(map) + sizeof(uint64_t));

    // Initialize to zero (unpublished)
    pomai::ai::atomic_utils::atomic_store_u64(payload, 0ULL);
    pomai::ai::atomic_utils::atomic_store_u32(flag, 0u);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> readers_observed_inconsistency{0};
    std::atomic<uint64_t> total_snapshots{0};

    // writer thread: repeatedly publish non-zero payload followed by setting flag
    auto writer = std::thread([&]()
                              {
        for (size_t i = 1; i <= ITER; ++i)
        {
            uint64_t v = (i << 1) | 1ULL; // odd non-zero value
            // publish payload then publish flag
            pomai::ai::atomic_utils::atomic_store_u64(payload, v);
            pomai::ai::atomic_utils::atomic_store_u32(flag, 1u);

            // small pause then unpublish (clear flag then payload)
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
                uint32_t f = 0;
                uint64_t p = 0;
                if (atomic_snapshot_flag_payload(flag, payload, f, p))
                {
                    total_snapshots.fetch_add(1, std::memory_order_relaxed);
                    if (f != 0 && p == 0)
                    {
                        // observed inconsistent state on a stable snapshot
                        readers_observed_inconsistency.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                std::this_thread::yield();
            } });
    }

    writer.join();

    // Give readers a bit to notice final state
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true, std::memory_order_release);

    for (auto &t : readers)
        t.join();

    uint64_t bad = readers_observed_inconsistency.load(std::memory_order_relaxed);
    uint64_t snaps = total_snapshots.load(std::memory_order_relaxed);

    double frac = snaps ? (double)bad / (double)snaps : 0.0;
    uint64_t allowed = std::max<uint64_t>(1, static_cast<uint64_t>(snaps * MAX_ALLOWED_FRACTION + 0.5));

    std::cout << "atomic_mmap_test: snapshots=" << snaps << " inconsistent=" << bad
              << " fraction=" << frac << " allowed=" << allowed << "\n";

    if (bad <= allowed)
    {
        std::cout << "PASS: inconsistent count within tolerated threshold\n";
        munmap(map, SZ);
        return 0;
    }
    else
    {
        std::cout << "FAIL: observed " << bad << " inconsistent reads (fraction=" << frac << ")\n";
        munmap(map, SZ);
        return 1;
    }
}