/*
 * tests/pppq/pppq_demote_concurrency_test.cc
 *
 * Updated test: readers now use PPPQ::stable_snapshot_read_pub to obtain a
 * consistent snapshot of (in_mmap, code_nbits, packed payload) instead of
 * reading the individual atomics separately. This prevents observing transient
 * inconsistent states during concurrent demotion.
 *
 * This test is a modest reproduction of the concurrency stress harness used
 * previously: multiple writer threads call addVec() which may trigger async
 * demotion; multiple reader threads repeatedly sample a random id using the
 * stable snapshot API and validate the (in_mmap -> nbits) invariant.
 *
 * Note: This test is intentionally defensive (retries + time-limited) and
 * asserts that inconsistent observations are near-zero.
 */

#include "src/ai/pppq.h"

#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>
#include <iostream>
#include <cassert>

using namespace pomai::ai;

int main(int argc, char **argv)
{
    const size_t DIM = 64;
    const size_t M = 8;
    const size_t K = 16;
    const size_t MAX_ELEMS = 1024;
    const size_t NUM_WRITERS = 4;
    const size_t NUM_READERS = 2;
    const int duration_s = 3;

    PPPQ ppq(DIM, M, K, MAX_ELEMS, "./pppq_test.mmap");

    // train with random samples
    {
        std::vector<float> samples(1000 * DIM);
        std::mt19937_64 rng(123);
        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        for (auto &v : samples)
            v = ud(rng);
        ppq.train(samples.data(), 1000, 5);
    }

    std::atomic<bool> stop{false};
    std::atomic<size_t> writes{0};
    std::atomic<size_t> reads{0};
    std::atomic<size_t> inconsistent{0};

    // Writer threads: continuously update random ids
    std::vector<std::thread> writers;
    for (size_t t = 0; t < NUM_WRITERS; ++t)
    {
        writers.emplace_back([&]()
        {
            std::mt19937_64 rng(std::random_device{}());
            std::uniform_int_distribution<size_t> idd(0, MAX_ELEMS - 1);
            std::vector<float> vec(DIM);
            std::uniform_real_distribution<float> ud(0.0f, 1.0f);

            while (!stop.load(std::memory_order_acquire))
            {
                size_t id = idd(rng);
                for (size_t i = 0; i < DIM; ++i)
                    vec[i] = ud(rng);
                ppq.addVec(vec.data(), id);
                writes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Reader threads: use stable_snapshot_read_pub to sample consistent snapshots
    std::vector<std::thread> readers;
    for (size_t t = 0; t < NUM_READERS; ++t)
    {
        readers.emplace_back([&]()
        {
            std::mt19937_64 rng(std::random_device{}());
            std::uniform_int_distribution<size_t> idd(0, MAX_ELEMS - 1);

            while (!stop.load(std::memory_order_acquire))
            {
                size_t id = idd(rng);

                uint8_t inmap = 0, nbits = 8;
                std::vector<uint8_t> packed;
                bool ok = ppq.stable_snapshot_read_pub(id, inmap, nbits, packed, 200);
                reads.fetch_add(1, std::memory_order_relaxed);

                if (!ok)
                {
                    // if we couldn't obtain a snapshot, count as transient inconsistency
                    inconsistent.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // Validate invariant: if inmap==1 then nbits must be 4 (packed)
                if (inmap != 0 && nbits != 4)
                {
                    inconsistent.fetch_add(1, std::memory_order_relaxed);
                }

                // conversely, if nbits==4, inmap should be 1 (packed read present)
                if (nbits == 4 && inmap == 0)
                {
                    inconsistent.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Run for duration_s seconds
    std::this_thread::sleep_for(std::chrono::seconds(duration_s));
    stop.store(true, std::memory_order_release);

    for (auto &th : writers)
        th.join();
    for (auto &th : readers)
        th.join();

    size_t w = writes.load();
    size_t r = reads.load();
    size_t ic = inconsistent.load();

    std::cout << "[pppq_demote_concurrency_test] writer_threads=" << NUM_WRITERS
              << " reader_threads=" << NUM_READERS
              << " duration_s=" << duration_s
              << " reads=" << r << " writes=" << w
              << " inconsistent=" << ic << "\n";

    // Allow a tiny number of transient failures, but test should be robust after using stable snapshot.
    const size_t allowed = std::max<size_t>(1, r / 1000); // 0.1%
    if (ic > allowed)
    {
        std::cerr << "FAIL: observed " << ic << " inconsistent reads (allowed=" << allowed << ")\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}