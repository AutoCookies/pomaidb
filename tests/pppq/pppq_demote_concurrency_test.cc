/*
 * tests/pppq/pppq_demote_concurrency_test.cc
 *
 * Stress test for concurrent demote/promote state (in_mmap / code_nbits).
 *
 * Purpose:
 *  - Simulate the concurrent pattern used when publishing/demoting PPPQ codes:
 *      * writers publish a non-zero code_nbits then set in_mmap
 *      * writers unpublish by clearing in_mmap then zeroing code_nbits
 *  - Readers must observe a consistent pair (in_mmap, code_nbits):
 *      - If in_mmap == 1 then code_nbits != 0
 *      - If in_mmap == 0 then code_nbits == 0
 *
 * This test verifies a robust read pattern: snapshot flags+payload using
 * repeated reads (read flag, read payload, read flag again) and accept only
 * when the flag value is stable. Any observation where snapshot returns
 * in_mmap==1 but code_nbits==0 is counted as an inconsistency.
 *
 * The test is standalone and does not depend on PPPQ internals; it validates
 * the publish/unpublish + snapshot idiom that callers (and PPPQ) should use.
 *
 * Exit code 0 == pass (no inconsistent observations).
 */

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

static bool atomic_snapshot_u32_pair(const uint32_t *flag_ptr, const uint32_t *payload_ptr, uint32_t &out_flag, uint32_t &out_payload)
{
    // Try a small number of snapshot attempts: read flag, payload, flag again and accept
    // only if flags are identical. This prevents observing a flag change mid-read.
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
        // yield to increase interleaving (and avoid busy spinning)
        std::this_thread::yield();
    }

    // Fallback: do seq_cst loads to get a deterministic (but potentially racy) view.
    out_flag = atomic_utils::atomic_load_u32(flag_ptr);
    out_payload = atomic_utils::atomic_load_u32(payload_ptr);
    return true;
}

int main()
{
    // One small "PPPQ entry" that holds:
    //   uint32_t in_mmap;     // 0 == not mapped, 1 == mapped
    //   uint32_t code_nbits;  // 0 == no code, >0 == number of bits present
    //
    // We place them in a small heap buffer to mimic memory-mapped placement.
    std::unique_ptr<uint32_t[]> buf(new uint32_t[2]);
    uint32_t *in_mmap_ptr = buf.get();
    uint32_t *code_nbits_ptr = buf.get() + 1;

    // initialize to zeros (unpublished)
    pomai::ai::atomic_utils::atomic_store_u32(in_mmap_ptr, 0);
    pomai::ai::atomic_utils::atomic_store_u32(code_nbits_ptr, 0);

    // test parameters
    const size_t reader_threads = std::max<size_t>(1, std::thread::hardware_concurrency() - 1);
    const std::chrono::seconds duration = std::chrono::seconds(3);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> inconsistent_count{0};
    std::atomic<uint64_t> total_reads{0};
    std::atomic<uint64_t> total_writes{0};

    // reader worker
    auto reader = [&]()
    {
        while (!stop.load(std::memory_order_acquire))
        {
            uint32_t flags = 0;
            uint32_t nb = 0;
            atomic_snapshot_u32_pair(in_mmap_ptr, code_nbits_ptr, flags, nb);
            // Interpret: in_mmap == 1 means mapped -> expect nb != 0
            if (flags == 1 && nb == 0)
            {
                inconsistent_count.fetch_add(1, std::memory_order_relaxed);
            }
            // Conversely, if in_mmap == 0 but nb != 0, also inconsistent
            if (flags == 0 && nb != 0)
            {
                inconsistent_count.fetch_add(1, std::memory_order_relaxed);
            }

            total_reads.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    };

    // writer worker: repeatedly publish non-zero code_nbits then set in_mmap, then later unpublish
    auto writer = [&]()
    {
        uint32_t bits = 1;
        while (!stop.load(std::memory_order_acquire))
        {
            // publish: store code_nbits (release) then set in_mmap (seq_cst)
            pomai::ai::atomic_utils::atomic_store_u32(code_nbits_ptr, bits);
            pomai::ai::atomic_utils::atomic_store_u32(in_mmap_ptr, 1);
            total_writes.fetch_add(1, std::memory_order_relaxed);

            // small pause to let readers observe
            std::this_thread::sleep_for(0us);

            // IMPORTANT: unpublish must clear the flag first, then clear the payload.
            // This avoids a window where in_mmap==1 while code_nbits==0.
            pomai::ai::atomic_utils::atomic_store_u32(in_mmap_ptr, 0);
            pomai::ai::atomic_utils::atomic_store_u32(code_nbits_ptr, 0);

            // advance bits
            ++bits;
            if (bits == 0)
                bits = 1;
        }
    };

    // start readers
    std::vector<std::thread> rthreads;
    for (size_t i = 0; i < reader_threads; ++i)
        rthreads.emplace_back(reader);

    // start writer
    std::thread wthread(writer);

    // run for duration
    std::this_thread::sleep_for(duration);
    stop.store(true, std::memory_order_release);

    // join
    wthread.join();
    for (auto &t : rthreads)
        t.join();

    uint64_t ic = inconsistent_count.load();
    uint64_t reads = total_reads.load();
    uint64_t writes = total_writes.load();

    std::cout << "[pppq_demote_concurrency_test] reader_threads=" << reader_threads
              << " duration_s=" << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
              << " reads=" << reads << " writes=" << writes << " inconsistent=" << ic << "\n";

    if (ic != 0)
    {
        std::cerr << "FAIL: observed " << ic << " inconsistent reads (in_mmap/code_nbits mismatch)\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}