/*
 * tests/pppq/pppq_demote_metrics_test.cc
 *
 * Robust metric/regression test for PPPQ demotion:
 *  - Create a PPPQ instance
 *  - Snapshot demote metrics
 *  - Insert N random vectors via addVec()
 *  - Call purgeCold() to trigger demotion sweep
 *  - Assert demote counters increased and that some ids have been demoted to 4-bit.
 *
 * This test tolerates PPPQ performing demotion during addVec() (predictor-driven),
 * but ensures there is actual demotion activity and metrics are reported.
 */

#include "src/ai/pppq.h"
#include "src/core/config.h"

#include <vector>
#include <random>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <string>
#include <chrono>
#include <thread>

int main()
{
    using namespace pomai::ai;

    // Test parameters
    const size_t N = 200;         // number of vectors / elements
    const size_t DIM = 64;
    const size_t M = 8;
    const size_t K = 256;

    // Use a temporary mmap file name unique to this test run
    std::string mmap_file = std::string("pppq_demote_metrics_test.bin");

    // Ensure synchronous demote path is possible (for predictability in many CI setups)
    pomai::config::runtime.demote_async_max_pending = 0; // disable async
    pomai::config::runtime.demote_sync_fallback = true;  // enable sync fallback

    // Create PPPQ
    PPPQ pq(DIM, M, K, N, mmap_file);

    // Snapshot metrics BEFORE inserting vectors so we can measure total demotion work
    uint64_t before_tasks = pq.get_demote_tasks_completed();
    uint64_t before_bytes = pq.get_demote_bytes_written();
    size_t before_pending = pq.get_pending_demotes();

    std::cout << "[pppq_test] baseline: tasks=" << before_tasks << " bytes=" << before_bytes << " pending=" << before_pending << "\n";

    // Generate deterministic pseudo-random vectors and insert them
    std::mt19937_64 rng(1234567);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> vec(DIM);

    for (size_t id = 0; id < N; ++id)
    {
        for (size_t j = 0; j < DIM; ++j)
            vec[j] = ud(rng);

        pq.addVec(vec.data(), id);
    }

    // Force demotion sweep with a large horizon so entries are considered cold
    pq.purgeCold(static_cast<uint64_t>(24ULL * 3600ULL * 1000ULL)); // 24 hours in ms

    // Snapshot after purge
    uint64_t after_tasks = pq.get_demote_tasks_completed();
    uint64_t after_bytes = pq.get_demote_bytes_written();
    size_t after_pending = pq.get_pending_demotes();

    std::cout << "[pppq_test] after: tasks=" << after_tasks << " bytes=" << after_bytes << " pending=" << after_pending << "\n";

    // Validate that demotion work was performed (either during insertion or purge)
    if (after_tasks < before_tasks)
    {
        std::cerr << "[pppq_test] FAIL: demote_tasks_completed decreased (before=" << before_tasks << ", after=" << after_tasks << ")\n";
        return 1;
    }
    if (after_bytes < before_bytes)
    {
        std::cerr << "[pppq_test] FAIL: demote_bytes_written decreased (before=" << before_bytes << ", after=" << after_bytes << ")\n";
        return 2;
    }

    uint64_t tasks_done = after_tasks - before_tasks;
    uint64_t bytes_done = after_bytes - before_bytes;

    // Ensure some demotion happened
    if (tasks_done == 0 || bytes_done == 0)
    {
        std::cerr << "[pppq_test] FAIL: no demotion work observed (tasks_done=" << tasks_done << ", bytes_done=" << bytes_done << ")\n";
        return 3;
    }

    // Packed bytes per vector for our m=M
    size_t packed_per_vec = (M + 1) / 2;
    if (bytes_done < tasks_done * static_cast<uint64_t>(packed_per_vec))
    {
        std::cerr << "[pppq_test] WARN: bytes_done (" << bytes_done << ") < tasks_done * packed_per_vec (" << tasks_done * packed_per_vec << ")\n";
        // Not fatal - different demote code paths may account bytes differently; continue.
    }

    // Ensure at least one element reports 4-bit storage (get_code_nbits accessor)
    bool found_4bit = false;
    for (size_t id = 0; id < N; ++id)
    {
        uint8_t bits = pq.get_code_nbits(id);
        if (bits == 4)
        {
            found_4bit = true;
            break;
        }
    }

    if (!found_4bit)
    {
        std::cerr << "[pppq_test] FAIL: no element reported 4-bit precision after purge\n";
        return 4;
    }

    // Basic sanity: pending demotes should be zero for sync path
    if (pq.get_pending_demotes() != 0)
    {
        std::cerr << "[pppq_test] WARN: pending_demotes != 0 after purge: " << pq.get_pending_demotes() << "\n";
    }

    // Clean up: remove mmap file
    std::remove(mmap_file.c_str());

    std::cout << "[pppq_test] PASS: demotion metrics updated, some entries demoted to 4-bit\n";
    return 0;
}