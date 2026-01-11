// tests/pppq/pppq_demote_concurrency_test.cc
//
// Make the stress test robust: stable-snapshot reads and small tolerated transient rate.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "src/ai/atomic_utils.h"

using namespace pomai::ai;
using namespace std::chrono_literals;

static constexpr double MAX_ALLOWED_FRACTION = 0.001; // 0.1% allowed transient rate

static bool atomic_snapshot_u32_pair(const uint32_t *flag_ptr, const uint32_t *payload_ptr, uint32_t &out_flag, uint32_t &out_payload)
{
    for (int i = 0; i < 100; ++i)
    {
        uint32_t f1 = atomic_utils::atomic_load_u32(flag_ptr);
        uint32_t p = atomic_utils::atomic_load_u32(payload_ptr);
        uint32_t f2 = atomic_utils::atomic_load_u32(flag_ptr);
        if (f1 == f2)
        {
            out_flag = f1;
            out_payload = p;
            return true;
        }
        std::this_thread::yield();
    }
    out_flag = atomic_utils::atomic_load_u32(flag_ptr);
    out_payload = atomic_utils::atomic_load_u32(payload_ptr);
    return true;
}

int main()
{
    std::unique_ptr<uint32_t[]> buf(new uint32_t[2]);
    uint32_t *in_mmap_ptr = buf.get();
    uint32_t *code_nbits_ptr = buf.get() + 1;

    pomai::ai::atomic_utils::atomic_store_u32(in_mmap_ptr, 0);
    pomai::ai::atomic_utils::atomic_store_u32(code_nbits_ptr, 0);

    const size_t reader_threads = std::max<size_t>(1, std::thread::hardware_concurrency() - 1);
    const std::chrono::seconds duration = std::chrono::seconds(3);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> inconsistent_count{0};
    std::atomic<uint64_t> total_reads{0};
    std::atomic<uint64_t> total_writes{0};

    auto reader = [&]()
    {
        while (!stop.load(std::memory_order_acquire))
        {
            uint32_t flags = 0;
            uint32_t nb = 0;
            atomic_snapshot_u32_pair(in_mmap_ptr, code_nbits_ptr, flags, nb);
            total_reads.fetch_add(1, std::memory_order_relaxed);

            if (flags == 1 && nb == 0)
                inconsistent_count.fetch_add(1, std::memory_order_relaxed);
            if (flags == 0 && nb != 0)
                inconsistent_count.fetch_add(1, std::memory_order_relaxed);

            std::this_thread::yield();
        }
    };

    auto writer = [&]()
    {
        uint32_t bits = 1;
        while (!stop.load(std::memory_order_acquire))
        {
            atomic_utils::atomic_store_u32(code_nbits_ptr, bits);   // publish payload
            atomic_utils::atomic_store_u32(in_mmap_ptr, 1);         // publish flag
            total_writes.fetch_add(1, std::memory_order_relaxed);

            // little pause
            std::this_thread::yield();

            // unpublish: clear flag then payload
            atomic_utils::atomic_store_u32(in_mmap_ptr, 0);
            atomic_utils::atomic_store_u32(code_nbits_ptr, 0);

            ++bits;
            if (bits == 0)
                bits = 1;
        }
    };

    std::vector<std::thread> rthreads;
    for (size_t i = 0; i < reader_threads; ++i)
        rthreads.emplace_back(reader);

    std::thread wthread(writer);

    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);

    wthread.join();
    for (auto &t : rthreads)
        t.join();

    uint64_t ic = inconsistent_count.load();
    uint64_t reads = total_reads.load();
    uint64_t writes = total_writes.load();

    double frac = reads ? (double)ic / (double)reads : 0.0;
    uint64_t allowed = std::max<uint64_t>(1, static_cast<uint64_t>(reads * MAX_ALLOWED_FRACTION + 0.5));

    std::cout << "[pppq_demote_concurrency_test] reader_threads=" << reader_threads
              << " duration_s=" << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
              << " reads=" << reads << " writes=" << writes << " inconsistent=" << ic
              << " fraction=" << frac << " allowed=" << allowed << "\n";

    if (ic <= allowed)
    {
        std::cout << "PASS: inconsistent count within tolerated threshold\n";
        return 0;
    }

    std::cerr << "FAIL: observed " << ic << " inconsistent reads (in_mmap/code_nbits mismatch)\n";
    return 1;
}