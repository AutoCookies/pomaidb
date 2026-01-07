#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <cassert>
#include <cstring>

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/pomai_hnsw.h"
#include "src/ai/ppe.h"

// NOTE: The test assumes your project provides memory/arena.h which defines PomaiArena
// with these methods used by the production code:
//  - static PomaiArena PomaiArena::FromMB(uint64_t mb)  (factory used here)
//  - char *PomaiArena::alloc_blob(uint32_t len)
//  - uint64_t PomaiArena::offset_from_blob_ptr(char *blob_ptr)
//  - const char *PomaiArena::blob_ptr_from_offset_for_map(uint64_t offset)
//
// If your PomaiArena factory or API differs, adjust the arena construction accordingly.

int main()
{
    using namespace pomai::ai;
    using namespace std::chrono;

    std::cout << "=== PPHNSW + Arena Integration Test ===\n";

    const int dim = 8; // small dim for test
    const size_t max_elements = 128;
    const size_t num_items = 16;

    // Create underlying L2 space
    hnswlib::L2Space l2space(dim);

    // Create PomaiArena via explicit factory to avoid constructor overload ambiguity.
    PomaiArena arena_obj = PomaiArena::FromMB(1); // 1 MB
    PomaiArena *arena = &arena_obj;

    if (!arena || !arena->is_valid())
    {
        std::cerr << "Failed to initialise PomaiArena via FromMB(1). Adjust test to match your PomaiArena API.\n";
        return 2;
    }

    // Create PPHNSW and attach arena
    PPHNSW<float> alg(&l2space, max_elements, /*M*/ 16, /*ef_construction*/ 100);
    alg.setPomaiArena(arena);

    // Also ensure the PomaiSpace has the arena (setPomaiArena already does it)
    if (alg.getSeedSize() == 0)
    {
        std::cerr << "Unexpected seed size 0\n";
        return 2;
    }

    // Generate deterministic data
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    std::vector<float> pool(dim * num_items);
    for (size_t i = 0; i < pool.size(); ++i)
        pool[i] = ud(rng);

    // Insert points with label == index
    for (size_t i = 0; i < num_items; ++i)
    {
        const float *vec = pool.data() + i * dim;
        alg.addPoint(vec, static_cast<hnswlib::labeltype>(i));
    }

    std::cout << "[PASS] Insertion: added " << num_items << " items.\n";

    // Now verify that elements are stored as indirect blobs in the arena
    // For each internal id 0..num_items-1, read payload and verify PPE_FLAG_INDIRECT and offset resolves.
    for (size_t i = 0; i < num_items; ++i)
    {
        // get pointer to element payload (header+vec/offset)
        char *elem = alg.getDataByInternalId(static_cast<hnswlib::tableint>(i));
        // header is at elem
        PPEHeader *h = reinterpret_cast<PPEHeader *>(elem);
        // header flags must include indirect
        if ((h->flags & PPE_FLAG_INDIRECT) == 0)
        {
            std::cerr << "[FAIL] element " << i << " not marked indirect in PPEHeader.flags\n";
            return 3;
        }
        // payload holds uint64_t offset
        uint64_t offset = 0;
        std::memcpy(&offset, elem + sizeof(PPEHeader), sizeof(offset));
        const char *blob_hdr = arena->blob_ptr_from_offset_for_map(offset);
        if (!blob_hdr)
        {
            std::cerr << "[FAIL] element " << i << " blob_ptr_from_offset_for_map returned null\n";
            return 4;
        }
        // first 4 bytes at blob_hdr are length
        uint32_t blen = *reinterpret_cast<const uint32_t *>(blob_hdr);
        if (blen != dim * sizeof(float))
        {
            std::cerr << "[FAIL] element " << i << " blob length mismatch: " << blen << " expected " << (dim * sizeof(float)) << "\n";
            return 5;
        }
        const char *blob_payload = blob_hdr + sizeof(uint32_t);
        // Compare bytes
        if (std::memcmp(blob_payload, pool.data() + i * dim, blen) != 0)
        {
            std::cerr << "[FAIL] element " << i << " blob payload mismatch\n";
            return 6;
        }
    }

    std::cout << "[PASS] Arena store: all elements moved to arena blobs and offsets stored.\n";

    // Run a search for label 3's vector and ensure found
    const float *query = pool.data() + 3 * dim;
    auto result = alg.searchKnnAdaptive(query, 3, 0.0f);

    bool found = false;
    while (!result.empty())
    {
        if (result.top().second == 3)
        {
            found = true;
            break;
        }
        result.pop();
    }

    if (found)
    {
        std::cout << "[PASS] Search: found target label 3.\n";
    }
    else
    {
        std::cerr << "[FAIL] Search: did not find target label 3.\n";
        return 7;
    }

    std::cout << "All tests passed.\n";
    return 0;
}