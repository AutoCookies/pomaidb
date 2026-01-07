#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstddef>
#include "memory/arena.h"
#include "core/map.h"

// Updated sanity_check: uses PomaiMap::find_seed(...) to obtain Seed*
// and uses arena.blob_ptr_from_offset_for_map(...) to resolve blob offsets.

int main()
{
    std::cout << "[SANITY] Start\n";

    PomaiArena arena = PomaiArena::FromMB(1); // 1MB
    if (!arena.is_valid())
    {
        std::cerr << "Arena allocation failed\n";
        return 1;
    }

    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;

    PomaiMap map(&arena, slots);

    // Small inline string
    const char *k1 = "small";
    const char *v1 = "hello"; // small
    if (!map.put(k1, strlen(k1), v1, strlen(v1)))
    {
        std::cerr << "put small failed\n";
        return 2;
    }

    uint32_t outlen1 = 0;
    const char *got1 = map.get(k1, strlen(k1), &outlen1);
    if (!got1)
    {
        std::cerr << "get small returned null\n";
        return 3;
    }
    std::cout << "[SMALL] value='" << std::string(got1, outlen1) << "' len=" << outlen1 << "\n";

    // Use find_seed to obtain the Seed* (safe for both inline and indirect)
    Seed *seed1 = map.find_seed(k1, strlen(k1));
    if (!seed1)
    {
        std::cerr << "find_seed small failed\n";
        return 11;
    }

    std::cout << "  seed addr: " << seed1 << "\n";
    std::cout << "  payload addr: " << static_cast<const void *>(seed1->payload) << "\n";
    std::cout << "  value addr: " << static_cast<const void *>(got1) << "\n";
    std::uintptr_t seed_addr = reinterpret_cast<std::uintptr_t>(seed1);
    std::uintptr_t val_addr = reinterpret_cast<std::uintptr_t>(got1);
    std::cout << "  seed delta: " << std::hex << (val_addr - seed_addr) << std::dec << " bytes\n";

    // Large string -> force indirect blob (1KB)
    const char *k2 = "big";
    std::string big;
    big.resize(1024);
    for (size_t i = 0; i < big.size(); ++i)
        big[i] = 'A' + (i % 26);

    if (!map.put(k2, strlen(k2), big.data(), big.size()))
    {
        std::cerr << "put big failed\n";
        return 4;
    }

    uint32_t outlen2 = 0;
    const char *got2 = map.get(k2, strlen(k2), &outlen2);
    if (!got2)
    {
        std::cerr << "get big returned null\n";
        return 5;
    }
    std::cout << "[BIG] value len=" << outlen2 << " first bytes: " << std::string(got2, std::min<size_t>(8, outlen2)) << "\n";

    // Use find_seed to get the correct Seed* for the key (safe)
    Seed *seed2 = map.find_seed(k2, strlen(k2));
    if (!seed2)
    {
        std::cerr << "find_seed big failed\n";
        return 12;
    }

    std::cout << "  seed addr: " << seed2 << "\n";

    // read blob offset (uint64_t) from seed payload and resolve to pointer via arena
    uint64_t offset = 0;
    // ensure we read the correct number of bytes for stored offset
    memcpy(&offset, seed2->payload + strlen(k2), sizeof(uint64_t));
    char *blob_hdr = const_cast<char *>(arena.blob_ptr_from_offset_for_map(offset));
    uint32_t blob_len = 0;
    if (blob_hdr != nullptr)
    {
        blob_len = *reinterpret_cast<uint32_t *>(blob_hdr);
    }
    std::cout << "  blob hdr addr: " << static_cast<void *>(blob_hdr) << ", blob_len=" << blob_len << "\n";
    std::cout << "  blob payload addr: " << static_cast<void *>(blob_hdr + sizeof(uint32_t)) << "\n";
    std::cout << "  value addr returned: " << static_cast<const void *>(got2) << "\n";

    // Validate that inline & blob data match
    if (std::string(got1, outlen1) != std::string(v1))
    {
        std::cerr << "small mismatch\n";
        return 6;
    }
    if (static_cast<uint32_t>(big.size()) != outlen2)
    {
        std::cerr << "big length mismatch: expected=" << big.size() << " got=" << outlen2 << "\n";
        return 7;
    }
    if (std::string(got2, outlen2) != big)
    {
        std::cerr << "big mismatch\n";
        return 8;
    }

    std::cout << "[SANITY] OK\n";
    return 0;
}