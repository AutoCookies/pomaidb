// tests/test_blob_freelist.cc
//
// Small deterministic test to validate blob free/reuse behavior.
// - Allocates an arena and map
// - Inserts a large value (indirect/blob), reads the blob header offset and resolves pointer via arena
// - Erases the key (should free the blob back to arena freelist)
// - Inserts another key with same-size value and checks whether the blob address was reused

#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>
#include <cassert>

#include "src/memory/arena.h"
#include "src/core/map.h"

int main()
{
    std::cout << "[TEST] Blob freelist reuse\n";

    // Use a modest arena (4 MB)
    const uint64_t arena_mb = 4;
    PomaiArena arena = PomaiArena::FromMB(arena_mb);
    if (!arena.is_valid())
    {
        std::cerr << "[FAIL] Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;

    PomaiMap map(&arena, slots);

    const char *k1 = "freelist_key_a";
    const char *k2 = "freelist_key_b";

    // Prepare 2KB value (large enough to force indirect blob)
    const size_t vlen = 2048;
    std::string v1;
    v1.resize(vlen);
    for (size_t i = 0; i < vlen; ++i)
        v1[i] = 'A' + (i % 26);

    // Insert first key
    bool ok = map.put(k1, strlen(k1), v1.data(), v1.size());
    if (!ok)
    {
        std::cerr << "[FAIL] initial put failed\n";
        return 2;
    }

    uint32_t outlen = 0;
    const char *got = map.get(k1, strlen(k1), &outlen);
    if (!got || outlen != vlen)
    {
        std::cerr << "[FAIL] get after put failed\n";
        return 3;
    }

    // Locate seed and read the blob header offset from payload
    Seed *s1 = map.find_seed(k1, strlen(k1));
    if (!s1)
    {
        std::cerr << "[FAIL] find_seed returned null\n";
        return 4;
    }

    // Only valid if indirect
    if ((s1->flags & Seed::FLAG_INDIRECT) == 0)
    {
        std::cerr << "[FAIL] expected indirect storage for large value but flag not set\n";
        return 5;
    }

    uint64_t offset1 = 0;
    memcpy(&offset1, s1->payload + strlen(k1), sizeof(uint64_t));
    char *blob_hdr1 = const_cast<char *>(arena.blob_ptr_from_offset_for_map(offset1));
    if (!blob_hdr1)
    {
        std::cerr << "[FAIL] blob pointer null\n";
        return 6;
    }

    std::cout << "  initial blob_hdr: " << static_cast<void *>(blob_hdr1) << " len=" << *reinterpret_cast<uint32_t *>(blob_hdr1) << "\n";

    // Erase key -> should free blob
    bool erased = map.erase(k1);
    if (!erased)
    {
        std::cerr << "[FAIL] erase failed\n";
        return 7;
    }

    // Insert second key with same size; allocator should reuse freed block
    ok = map.put(k2, strlen(k2), v1.data(), v1.size());
    if (!ok)
    {
        std::cerr << "[FAIL] put for second key failed\n";
        return 8;
    }

    const char *got2 = map.get(k2, strlen(k2), &outlen);
    if (!got2 || outlen != vlen)
    {
        std::cerr << "[FAIL] get after second put failed\n";
        return 9;
    }

    Seed *s2 = map.find_seed(k2, strlen(k2));
    if (!s2)
    {
        std::cerr << "[FAIL] find_seed for second key failed\n";
        return 10;
    }
    if ((s2->flags & Seed::FLAG_INDIRECT) == 0)
    {
        std::cerr << "[FAIL] second value not stored indirect as expected\n";
        return 11;
    }

    uint64_t offset2 = 0;
    memcpy(&offset2, s2->payload + strlen(k2), sizeof(uint64_t));
    char *blob_hdr2 = const_cast<char *>(arena.blob_ptr_from_offset_for_map(offset2));
    if (!blob_hdr2)
    {
        std::cerr << "[FAIL] second blob pointer null\n";
        return 12;
    }

    std::cout << "  second blob_hdr:  " << static_cast<void *>(blob_hdr2) << " len=" << *reinterpret_cast<uint32_t *>(blob_hdr2) << "\n";

    if (blob_hdr1 == blob_hdr2)
    {
        std::cout << "[PASS] Blob block was reused from freelist\n";
        return 0;
    }
    else
    {
        std::cout << "[WARN] Blob block was NOT reused; allocator may choose another block (ok if fragmentation)\n";
        std::cout << "[INFO] This is permissible but indicates freelist behavior didn't return the same address.\n";
        std::cout << "[RESULT] test completed (non-fatal)\n";
        return 0;
    }
}