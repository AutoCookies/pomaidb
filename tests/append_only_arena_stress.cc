// tests/append_only_arena_stress.cc
//
// Stress / concurrency tests for AppendOnlyArena
//
// Goals:
//  - Exercise concurrent alloc_blob() + grow_to() (remap) with many threads.
//  - Spawn concurrent readers that call blob_ptr_from_offset_for_map(),
//    offset_from_blob_ptr() and persist_range() repeatedly.
//  - Perform long-running fuzz-style allocation/grow cycles.
//  - Try two growth patterns to increase chance of hitting mremap in-place
//    and fallback mmap paths (not guaranteed, but both patterns exercised).
//
// Usage (manual):
//   Build as part of the project and link with pomai core objects that contain
//   AppendOnlyArena (src/memory). Example:
//     mkdir build && cd build
//     cmake ..
//     make append_only_arena_stress   # or add test target in your CMake
//   Then run: ./tests/append_only_arena_stress /tmp/pomai_arena_test.bin
//
// The test is intentionally defensive: crashes/faults indicate real concurrency bugs.
// It performs many allocations and validates offsets / pointer round-trips.
//
// Note: This test doesn't rely on any testing framework to keep integration simple.
//
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <unistd.h>

#include "src/memory/append_only_arena.h"
#include "src/core/config.h"

using pomai::memory::AppendOnlyArena;

static void hexdump_prefix(const void *p, size_t n, std::ostream &os = std::cerr) {
    const uint8_t *b = reinterpret_cast<const uint8_t *>(p);
    os << std::hex;
    for (size_t i = 0; i < n; ++i) {
        if (i) os << ' ';
        os << static_cast<int>(b[i]);
    }
    os << std::dec << "\n";
}

struct AllocRecord {
    uint64_t offset;
    uint32_t payload_size;
    uint8_t pattern; // byte pattern filled by writer
};

int main(int argc, char **argv) {
    std::string path = "/tmp/pomai_arena_stress.bin";
    if (argc >= 2) path = argv[1];

    // Clean old file
    std::error_code ec;
    std::filesystem::remove(path, ec);

    pomai::config::StorageConfig cfg;
    cfg.initial_arena_size_mb = 4; // not used by OpenOrCreate (we pass size directly)
    cfg.growth_factor = 2.0f;
    cfg.alignment = 8;
    cfg.prefer_fallocate = false; // avoid posix_fallocate interfering in some environments

    size_t initial_bytes = 1ULL << 20; // 1MiB

    AppendOnlyArena *arena = nullptr;
    try {
        arena = AppendOnlyArena::OpenOrCreate(path, initial_bytes, cfg);
    } catch (const std::exception &e) {
        std::cerr << "OpenOrCreate failed: " << e.what() << "\n";
        return 1;
    }

    assert(arena);

    std::atomic<bool> stop{false};
    std::mutex rec_mu;
    std::vector<AllocRecord> records;
    records.reserve(1 << 20);

    // writer threads: allocate and write pattern
    const int N_WRITERS = 8;
    const int N_READERS = 4;
    const int N_GROWERS = 2;

    std::atomic<uint64_t> alloc_count{0};
    std::atomic<uint64_t> fail_allocs{0};

    // Random generator for sizes
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<int> szdist(8, 2048);
    std::uniform_int_distribution<int> patdist(1, 255);

    auto writer_main = [&](int id) {
        std::mt19937_64 local_rng(rd() ^ (id + 0x9e3779b97f4a7c15ULL));
        std::uniform_int_distribution<int> local_sz(8, 2048);
        std::uniform_int_distribution<int> local_pat(1, 255);
        while (!stop.load(std::memory_order_acquire)) {
            int payload = local_sz(local_rng);
            void *hdr = nullptr;
            try {
                hdr = arena->alloc_blob(static_cast<uint32_t>(payload));
            } catch (...) {
                hdr = nullptr;
            }
            if (!hdr) {
                fail_allocs.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            uint64_t off = arena->offset_from_blob_ptr(hdr);
            if (off == UINT64_MAX) {
                std::cerr << "[writer] offset_from_blob_ptr returned invalid for hdr\n";
            }
            uint8_t pat = static_cast<uint8_t>(local_pat(local_rng));
            // write payload bytes
            char *payload_ptr = reinterpret_cast<char *>(hdr) + sizeof(uint32_t);
            for (int i = 0; i < payload; ++i) payload_ptr[i] = static_cast<char>(pat);

            {
                std::lock_guard<std::mutex> lk(rec_mu);
                records.push_back(AllocRecord{off, static_cast<uint32_t>(payload), pat});
            }
            alloc_count.fetch_add(1, std::memory_order_relaxed);

            // short sleep to increase interleaving
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    };

    // grower threads: call grow_to with different policies to exercise remap paths
    auto grower_main = [&](int id) {
        std::mt19937_64 local_rng(rd() ^ (id * 1337));
        std::uniform_int_distribution<int> scale_small(1, 4);
        std::uniform_int_distribution<int> scale_large(4, 32);

        for (;;) {
            if (stop.load(std::memory_order_acquire)) break;
            // choose growth pattern
            size_t cur_cap = arena->get_capacity_bytes();
            size_t grow_by_pages = (id % 2 == 0) ? scale_small(local_rng) : scale_large(local_rng);
            long ps = sysconf(_SC_PAGESIZE);
            size_t page = (ps > 0) ? static_cast<size_t>(ps) : 4096;
            size_t new_size = cur_cap + grow_by_pages * page;
            bool ok = arena->grow_to(new_size);
            if (!ok) {
                // best-effort retry later
                // Log occasionally
                static std::atomic<uint64_t> bad{0};
                if ((bad.fetch_add(1) & 0xFFF) == 0) {
                    std::cerr << "[grower] grow_to(" << new_size << ") failed (cur=" << cur_cap << ")\n";
                }
            } else {
                // occasionally call persist_range over random region
                size_t off = (cur_cap > page) ? (page) : 0;
                size_t len = std::min<size_t>(page, new_size - off);
                arena->persist_range(off, len, false);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5 + (id * 3)));
        }
    };

    // reader threads: pick random recorded offsets and validate round-trip and payload
    auto reader_main = [&](int id) {
        std::mt19937_64 local_rng(rd() ^ (id + 0xabcdef));
        while (!stop.load(std::memory_order_acquire)) {
            AllocRecord rec;
            {
                std::lock_guard<std::mutex> lk(rec_mu);
                if (records.empty()) {
                    // no records yet
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                size_t idx = static_cast<size_t>(local_rng() % records.size());
                rec = records[idx];
            }

            const char *ptr = arena->blob_ptr_from_offset_for_map(rec.offset);
            if (!ptr) {
                // Could be remote or not yet published; skip
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            uint32_t stored_len = *reinterpret_cast<const uint32_t *>(ptr);
            if (stored_len < rec.payload_size) {
                std::cerr << "[reader] stored_len < expected: " << stored_len << " < " << rec.payload_size << "\n";
            }
            const char *payload = ptr + sizeof(uint32_t);
            // Validate prefix of payload matches pattern
            bool ok = true;
            for (uint32_t i = 0; i < std::min<uint32_t>(stored_len, rec.payload_size); ++i) {
                if (static_cast<uint8_t>(payload[i]) != rec.pattern) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                std::cerr << "[reader] payload mismatch at offset " << rec.offset << " exp=" << (int)rec.pattern
                          << " len=" << rec.payload_size << " stored_len=" << stored_len << "\n";
                // dump first bytes
                hexdump_prefix(payload, std::min<size_t>(16, stored_len));
            }

            // validate offset_from_blob_ptr round-trip
            uint64_t back_off = arena->offset_from_blob_ptr(ptr);
            if (back_off != rec.offset) {
                std::cerr << "[reader] round-trip offset mismatch: rec=" << rec.offset << " back=" << back_off << "\n";
            }

            // occasionally persist the region
            if ((local_rng() & 0xFFF) == 0) {
                // align to page for msync heuristic
                long ps = sysconf(_SC_PAGESIZE);
                size_t page = (ps > 0) ? static_cast<size_t>(ps) : 4096;
                size_t start = (rec.offset / page) * page;
                size_t len = ((sizeof(uint32_t) + rec.payload_size + page - 1) / page) * page;
                arena->persist_range(start, len, false);
            }

            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    };

    // Spawn threads
    std::vector<std::thread> writers;
    for (int i = 0; i < N_WRITERS; ++i) writers.emplace_back(writer_main, i);

    std::vector<std::thread> readers;
    for (int i = 0; i < N_READERS; ++i) readers.emplace_back(reader_main, i);

    std::vector<std::thread> growers;
    for (int i = 0; i < N_GROWERS; ++i) growers.emplace_back(grower_main, i);

    // Run test for a duration
    const auto RUN_SEC = 20;
    std::cerr << "[test] running for " << RUN_SEC << "s with " << N_WRITERS << " writers, " << N_READERS << " readers, " << N_GROWERS << " growers\n";
    std::this_thread::sleep_for(std::chrono::seconds(RUN_SEC));

    // stop threads
    stop.store(true, std::memory_order_release);

    for (auto &t : writers) if (t.joinable()) t.join();
    for (auto &t : readers) if (t.joinable()) t.join();
    for (auto &t : growers) if (t.joinable()) t.join();

    std::cerr << "[test] joined threads. allocations=" << alloc_count.load() << " failed_allocs=" << fail_allocs.load()
              << " records=" << records.size() << "\n";

    // Final validation: ensure every recorded offset still maps and round-trip works
    size_t bad = 0;
    {
        std::lock_guard<std::mutex> lk(rec_mu);
        for (size_t i = 0; i < records.size(); ++i) {
            auto &r = records[i];
            const char *ptr = arena->blob_ptr_from_offset_for_map(r.offset);
            if (!ptr) {
                // Could be remote demoted; for AppendOnlyArena test we assume local
                bad++;
                if ((bad & 0x3FF) == 0) std::cerr << "[validate] missing mapping for offset " << r.offset << "\n";
                continue;
            }
            uint64_t roff = arena->offset_from_blob_ptr(ptr);
            if (roff != r.offset) {
                bad++;
                std::cerr << "[validate] round-trip mismatch idx=" << i << " off=" << r.offset << " -> " << roff << "\n";
            }
            uint32_t slen = *reinterpret_cast<const uint32_t *>(ptr);
            if (slen < r.payload_size) {
                bad++;
                std::cerr << "[validate] stored len smaller than expected off=" << r.offset << " stored=" << slen << " exp=" << r.payload_size << "\n";
            } else {
                const char *payload = ptr + sizeof(uint32_t);
                bool ok = true;
                for (uint32_t k = 0; k < r.payload_size; ++k) {
                    if (static_cast<uint8_t>(payload[k]) != r.pattern) { ok = false; break; }
                }
                if (!ok) {
                    bad++;
                    std::cerr << "[validate] payload content mismatch off=" << r.offset << "\n";
                }
            }
        }
    }

    if (bad == 0) {
        std::cerr << "[test] SUCCESS: All records validated\n";
    } else {
        std::cerr << "[test] FAIL: " << bad << " records failed validation out of " << records.size() << "\n";
    }

    // Clean up
    delete arena;
    // remove file
    std::filesystem::remove(path, ec);

    return (bad == 0) ? 0 : 2;
}