#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cmath>
#include <thread>

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/pomai_hnsw.h"
#include "src/ai/ppe.h"

// Demotion integration test:
// - build small index with arena attached
// - mark a few nodes as "cold" by setting last_access_ns far in the past
// - start background demoter, wait until those nodes become REMOTE (demoted)
// - verify PPE_FLAG_REMOTE gets set and blob_ptr_from_offset_for_map can resolve remote ids

int main()
{
    using namespace pomai::ai;
    using namespace std::chrono;

    std::cout << "=== PPHNSW demotion integration test ===\n";

    const int dim = 8;
    const size_t max_elements = 256;
    const size_t num_items = 64;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    // Create simple data
    std::vector<std::vector<float>> pool(num_items, std::vector<float>(dim));
    for (size_t i = 0; i < num_items; ++i)
        for (int d = 0; d < dim; ++d)
            pool[i][d] = ud(rng);

    // Underlying L2 space
    hnswlib::L2Space l2(dim);

    // Small arena: 2 MB
    PomaiArena arena = PomaiArena::FromMB(2);

    if (!arena.is_valid())
    {
        std::cerr << "Arena invalid\n";
        return 2;
    }

    // Create PPHNSW and attach arena
    PPHNSW<float> alg(&l2, max_elements, /*M*/ 16, /*ef_construction*/ 100);
    alg.setPomaiArena(&arena);

    // Insert points
    for (size_t i = 0; i < num_items; ++i)
    {
        alg.addPoint(pool[i].data(), static_cast<hnswlib::labeltype>(i));
    }
    std::cout << "[INFO] Inserted " << num_items << " items\n";

    // Ensure payloads were moved to arena (INDIRECT) by earlier addPoint logic
    size_t seed_size = alg.getSeedSize();
    if (seed_size == 0)
    {
        std::cerr << "Unexpected seed size 0\n";
        return 3;
    }

    // Choose a small set of indices to mark cold and expect them to be demoted
    std::vector<size_t> to_demote = {10, 11, 12, 13, 14};

    // Set last_access_ns of these nodes far in the past to force demotion
    int64_t now = PPEHeader::now_ns();
    int64_t old_time = now - static_cast<int64_t>(10ULL * 1000ULL * 1000ULL * 1000ULL); // 10 seconds ago
    for (size_t idx : to_demote)
    {
        char *ptr = alg.getDataByInternalId(static_cast<hnswlib::tableint>(idx));
        // apply stores directly to the header to avoid unused-variable warnings
        reinterpret_cast<PPEHeader *>(ptr)->last_access_ns.store(old_time, std::memory_order_relaxed);
        reinterpret_cast<PPEHeader *>(ptr)->ema_interval_ns.store(100.0, std::memory_order_relaxed);
    }
    std::cout << "[INFO] Marked " << to_demote.size() << " nodes as cold (old timestamps)\n";

    // Start background demoter: run frequently with small lookahead so promotion is unlikely in the short test.
    alg.startBackgroundDemoter(100 /*ms*/, 500000000 /*ns = 0.5s lookahead*/);
    std::cout << "[INFO] Background demoter started\n";

    // Wait for demoter to demote the selected nodes (timeout)
    const auto deadline = steady_clock::now() + seconds(3);
    std::vector<size_t> still_not_demoted = to_demote;
    while (steady_clock::now() < deadline && !still_not_demoted.empty())
    {
        std::this_thread::sleep_for(milliseconds(120));
        std::vector<size_t> remaining;
        for (size_t idx : still_not_demoted)
        {
            char *ptr = alg.getDataByInternalId(static_cast<hnswlib::tableint>(idx));
            PPEHeader *h = reinterpret_cast<PPEHeader *>(ptr);
            if ((h->flags & PPE_FLAG_REMOTE) == 0)
                remaining.push_back(idx);
        }
        still_not_demoted.swap(remaining);
    }

    if (!still_not_demoted.empty())
    {
        std::cerr << "[FAIL] Some nodes were not demoted within timeout: ";
        for (size_t v : still_not_demoted)
            std::cerr << v << " ";
        std::cerr << "\n";
        alg.stopBackgroundDemoter();
        return 4;
    }

    std::cout << "[PASS] All target nodes were demoted to remote storage.\n";

    // Verify that arena.blob_ptr_from_offset_for_map can resolve remote ids for one demoted node
    {
        size_t idx = to_demote.front();
        char *ptr = alg.getDataByInternalId(static_cast<hnswlib::tableint>(idx));
        PPEHeader *h = reinterpret_cast<PPEHeader *>(ptr);
        uint64_t stored = 0;
        std::memcpy(&stored, ptr + sizeof(PPEHeader), sizeof(stored));
        // stored should be remote id (>= arena blob_region_bytes). We can't access blob_region_bytes directly,
        // but arena.blob_ptr_from_offset_for_map must return non-null and length header should match dim*sizeof(float)
        const char *blob_hdr = arena.blob_ptr_from_offset_for_map(stored);
        if (!blob_hdr)
        {
            std::cerr << "[FAIL] blob_ptr_from_offset_for_map returned null for remote id\n";
            alg.stopBackgroundDemoter();
            return 5;
        }
        uint32_t blen = *reinterpret_cast<const uint32_t *>(blob_hdr);
        size_t expected = dim * sizeof(float);
        if (blen != expected)
        {
            std::cerr << "[FAIL] remote blob length mismatch: " << blen << " != " << expected << "\n";
            alg.stopBackgroundDemoter();
            return 6;
        }
        std::cout << "[PASS] remote blob resolved and length matches expected (" << blen << ")\n";
    }

    // Stop background demoter and finish
    alg.stopBackgroundDemoter();
    std::cout << "[INFO] Background demoter stopped\n";

    std::cout << "Demotion integration test completed successfully.\n";
    return 0;
}