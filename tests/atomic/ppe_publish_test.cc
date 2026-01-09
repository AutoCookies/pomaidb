// tests/atomic/ppe_publish_test.cc
//
// Stress test for publish ordering:
//  - writer repeatedly stores a non-zero 64-bit payload to memory (atomic_store_u64)
//    then sets PPE_FLAG_INDIRECT (atomic_set_flags).
//  - multiple readers concurrently read flags (atomic_load_u32) and when they see
//    PPE_FLAG_INDIRECT they read the payload (atomic_load_u64) and verify payload != 0.
// Any observed case where flags has INDIRECT but payload==0 is considered a failure.
//
// This is a standalone test program (no test framework). Exit code 0 == pass,
// non-zero == fail.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <cstring> // <-- added

#include "src/ai/ppe.h"
#include "src/ai/atomic_utils.h"

using namespace pomai::ai;
using namespace std::chrono_literals;

int main()
{
    // allocate a small buffer that holds PPEHeader + uint64_t payload
    const size_t BUF_SZ = sizeof(PPEHeader) + sizeof(uint64_t);
    std::unique_ptr<char[]> buf(new char[BUF_SZ]);

    // placement-new PPEHeader at start
    void *hdr_ptr = buf.get();
    std::memset(hdr_ptr, 0, BUF_SZ);
    PPEHeader *h = new (hdr_ptr) PPEHeader();

    // payload pointer immediately after header
    uint64_t *payload_ptr = reinterpret_cast<uint64_t *>(buf.get() + sizeof(PPEHeader));

    // initialize to zero
    pomai::ai::atomic_utils::atomic_store_u64(payload_ptr, 0);
    h->atomic_clear_flags(0xFFFFFFFFu);

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
            uint64_t val = 0;
            // Use snapshot API to read flags+payload consistently
            h->atomic_snapshot_payload_and_flags(reinterpret_cast<const uint64_t *>(payload_ptr), flags, val);
            if (flags & PPE_FLAG_INDIRECT)
            {
                // if we ever see INDIRECT while payload==0 -> inconsistent
                if (val == 0)
                {
                    inconsistent_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
            total_reads.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    };

    // writer worker: repeatedly publish non-zero payload then set flag, then clear for next iteration
    auto writer = [&]()
    {
        uint64_t v = 1;
        while (!stop.load(std::memory_order_acquire))
        {
            // publish payload and set flag atomically using PPE helper
            h->atomic_publish_payload(reinterpret_cast<uint64_t *>(payload_ptr), v, PPE_FLAG_INDIRECT);
            total_writes.fetch_add(1, std::memory_order_relaxed);

            // small pause to let readers observe
            std::this_thread::sleep_for(0us);

            // unpublish (clear flag then reset payload) atomically
            h->atomic_unpublish_payload(reinterpret_cast<uint64_t *>(payload_ptr), PPE_FLAG_INDIRECT);

            ++v;
            if (v == 0)
                v = 1;
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

    std::cout << "[ppe_publish_test] reader_threads=" << reader_threads
              << " duration_s=" << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
              << " reads=" << reads << " writes=" << writes << " inconsistent=" << ic << "\n";

    if (ic != 0)
    {
        std::cerr << "FAIL: observed " << ic << " inconsistent reads (flag INDIRECT with payload==0)\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}