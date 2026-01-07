#pragma once
// memory/arena.h
//
// PomaiArena: mmap-backed arena with segregated seed & blob regions, thread-safe access,
// free-list for seeds and slab-style freelists for blobs, active-seed index for safe sampling.
// - Seeds live in the first region (seed_region_bytes) and are fixed-size slots.
// - Blobs live in the second region and are allocated from a bump pointer with power-of-two slab buckets.
// - Seed allocation reuses indices from free_seeds (LIFO). Double-free detection and active-index
//   bookkeeping prevents sampling freed slots and detects misuse in debug builds.
//
// Key safety / op changes (all in English):
// - Thread-safety: protect mutable arena state (free_seeds, free_lists, seed_next_slot, blob_next_offset, rng, active tables)
//   with a std::mutex (single mutex). This keeps the implementation simple and correct for multi-writer use.
// - Segregation: split the mmap region into SEED_REGION_RATIO for seeds and remainder for blobs to avoid interleaved fragmentation.
// - Offsets: store blob offsets (uint64_t) into seed payload rather than raw pointers. Offsets are relative to base_addr.
//   This avoids issues if the arena is relocated or if pointers are serialized erroneously.
// - Active index table: maintain active_seeds vector + active_pos index map so get_random_seed samples only active seeds.
// - Double-free detection: active_pos[idx] == UINT64_MAX indicates free; freeing an already-free slot is detected and ignored.
// - Freelist caps: cap per-bucket freelist growth to avoid pathological freelist explosion.
// - Alignment assertion: runtime check that mmap'd base is properly aligned for Seed.

#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <new>
#include <mutex>
#include <limits>
#include "core/seed.h"
#include "core/config.h"

class PomaiArena
{
private:
    char *base_addr = nullptr;
    uint64_t capacity = 0;

    // Seed region (fixed-size slots)
    char *seed_base = nullptr;
    uint64_t seed_region_bytes = 0;
    uint64_t seed_max_slots = 0; // capacity / sizeof(Seed) for seed region
    uint64_t seed_next_slot = 0; // next slot index for bump allocation

    // Blob region (variable-size slabs)
    char *blob_base = nullptr;
    uint64_t blob_region_bytes = 0;
    uint64_t blob_next_offset = 0; // offset within blob region (bytes)

    // Seed freelist & active tracking
    std::vector<uint64_t> free_seeds;   // indices of freed seed slots (LIFO)
    std::vector<uint64_t> active_seeds; // list of active seed indices for O(1) random sampling
    std::vector<uint64_t> active_pos;   // mapping slot idx -> position in active_seeds (UINT64_MAX == free)
    std::mt19937_64 rng;

    // Free lists for blobs keyed by block_size (power-of-two)
    // value: vector of offsets (uint64_t) relative to blob_base
    std::unordered_map<uint64_t, std::vector<uint64_t>> free_lists;

    // Synchronization for all mutable state inside the arena.
    // This is a single coarse-grained mutex. If contention becomes an issue, we can shard.
    // Marked mutable so const methods can lock it for logical const operations (e.g. read-only queries
    // that still need to inspect internal containers). This preserves logical constness while allowing
    // thread-safe access.
    mutable std::mutex mu;

    // Minimum blob block size (including header). Choose 64B to avoid too many tiny buckets.
    static constexpr uint64_t MIN_BLOB_BLOCK = 64;

    // Cap freelist per bucket to avoid freelist explosion under pathological workloads.
    static constexpr size_t MAX_FREELIST_PER_BUCKET = 4096;

    // Seed region ratio (fraction of arena bytes reserved for seeds).
    // This is conservative: seeds are small but numerous; adjust if needed.
    static constexpr double SEED_REGION_RATIO = 0.25;

    // Round up to nearest power-of-two >= v, with minimum MIN_BLOB_BLOCK
    static uint64_t block_size_for(uint64_t v)
    {
        uint64_t needed = std::max<uint64_t>(v, MIN_BLOB_BLOCK);
        uint64_t b = 1;
        while (b < needed)
            b <<= 1;
        return b;
    }

public:
    PomaiArena() : base_addr(nullptr), capacity(0), seed_base(nullptr), seed_region_bytes(0), seed_max_slots(0), seed_next_slot(0),
                   blob_base(nullptr), blob_region_bytes(0), blob_next_offset(0), rng(std::random_device{}()) {}

    explicit PomaiArena(uint64_t bytes) : PomaiArena()
    {
        if (bytes)
            allocate_region(bytes);
    }

    explicit PomaiArena(double gb) : PomaiArena()
    {
        if (gb > 0.0)
        {
            uint64_t bytes = static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
            if (bytes == 0)
                bytes = 1;
            allocate_region(bytes);
        }
    }

    // Factories (use config if gb <= 0)
    static PomaiArena FromGB(double gb)
    {
        PomaiArena a;
        double use_gb = gb;
        if (use_gb <= 0.0)
        {
            uint64_t mb = pomai::config::runtime.arena_mb_per_shard;
            if (mb == 0)
                mb = 512;
            use_gb = static_cast<double>(mb) / 1024.0;
        }
        uint64_t bytes = static_cast<uint64_t>(use_gb * 1024.0 * 1024.0 * 1024.0);
        if (bytes == 0)
            bytes = 1;
        a.allocate_region(bytes);
        return a;
    }

    static PomaiArena FromMB(uint64_t mb)
    {
        PomaiArena a;
        if (mb == 0)
            return a;
        uint64_t bytes = mb * 1024ULL * 1024ULL;
        a.allocate_region(bytes);
        return a;
    }

    bool is_valid() const noexcept { return base_addr != nullptr && capacity > 0; }

    void seed_rng(uint64_t s)
    {
        std::lock_guard<std::mutex> lk(mu);
        rng.seed(s);
    }

    // Convert an offset (relative to blob_base) to an absolute pointer.
    // Public because Map needs to translate stored offsets into pointers.
    inline char *ptr_from_blob_offset(uint64_t offset) const
    {
        if (!blob_base)
            return nullptr;
        if (offset >= blob_region_bytes)
            return nullptr;
        return blob_base + offset;
    }

    // Convert an absolute pointer inside blob region back to offset relative to blob_base.
    inline uint64_t offset_from_blob_ptr(const char *p) const
    {
        if (!blob_base || p < blob_base || p >= blob_base + blob_region_bytes)
            return UINT64_MAX;
        return static_cast<uint64_t>(p - blob_base);
    }

    // alloc_seed: prefer free_seeds, else bump in seed region.
    // Behavior:
    // - returns pointer to a value-initialized Seed (placement-new used).
    // - maintains active_seeds and active_pos for safe sampling.
    Seed *alloc_seed()
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!is_valid() || seed_max_slots == 0)
            return nullptr;

        uint64_t idx = UINT64_MAX;
        if (!free_seeds.empty())
        {
            idx = free_seeds.back();
            free_seeds.pop_back();
            // sanity
            if (idx >= seed_max_slots)
                return nullptr;
        }
        else
        {
            if (seed_next_slot >= seed_max_slots)
                return nullptr;
            idx = seed_next_slot++;
        }

        char *addr = seed_base + idx * sizeof(Seed);
        Seed *s = reinterpret_cast<Seed *>(addr);
        // Value-initialize in-place
        new (s) Seed();
        // Ensure deterministic starting state
        s->header.store(0ULL, std::memory_order_relaxed);
        s->entropy = 0;
        s->checksum = 0;
        s->type = 0;
        s->flags = 0;

        // Add to active list
        uint64_t pos = active_seeds.size();
        active_seeds.push_back(idx);
        active_pos[idx] = pos;

        return s;
    }

    // free_seed: remove from active list and push index to free_seeds.
    // Double-free is detected by checking active_pos map.
    void free_seed(Seed *s)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!is_valid() || s == nullptr)
            return;
        uintptr_t off = reinterpret_cast<char *>(s) - seed_base;
        if (reinterpret_cast<char *>(s) < seed_base || reinterpret_cast<char *>(s) >= seed_base + seed_region_bytes)
            return;
        if (off % sizeof(Seed) != 0)
            return;
        uint64_t idx = off / sizeof(Seed);
        if (idx >= seed_max_slots)
            return;

        // Double-free detection: ensure it's active
        uint64_t pos = active_pos[idx];
        if (pos == UINT64_MAX)
        {
            // already free; ignore (or optionally log in debug)
            return;
        }

        // Free any runtime-visible resources by zeroing semantic fields
        s->header.store(0ULL, std::memory_order_release);
        s->entropy = 0;
        s->checksum = 0;
        s->type = 0;
        s->flags = 0;

        // Call destructor to match placement-new semantics
        s->~Seed();

        // Remove from active_seeds using swap-pop for O(1)
        uint64_t last_pos = active_seeds.size() - 1;
        if (pos != last_pos)
        {
            uint64_t moved_idx = active_seeds[last_pos];
            active_seeds[pos] = moved_idx;
            active_pos[moved_idx] = pos;
        }
        active_seeds.pop_back();
        active_pos[idx] = UINT64_MAX;

        // Push into free list (LIFO)
        free_seeds.push_back(idx);
    }

    // Allocates blob with 4-byte length prefix: [uint32_t len][data...], returns pointer to start (the length header)
    // slab-like power-of-two buckets; freelist reused on free_blob.
    // Returns nullptr on OOM.
    char *alloc_blob(uint32_t len)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!is_valid())
            return nullptr;

        uint64_t hdr = sizeof(uint32_t);
        uint64_t total = hdr + static_cast<uint64_t>(len) + 1; // +1 null terminator
        uint64_t block = block_size_for(total);

        // Try freelist for this block size
        auto it = free_lists.find(block);
        if (it != free_lists.end() && !it->second.empty())
        {
            uint64_t offset = it->second.back();
            it->second.pop_back();
            char *p = blob_base + offset;
            uint32_t *lenptr = reinterpret_cast<uint32_t *>(p);
            *lenptr = len;
            char *payload = p + hdr;
            payload[len] = '\0';
            return p;
        }

        // Bump allocate inside blob region (aligned to block)
        uint64_t aligned_next = (blob_next_offset + (block - 1)) & ~(block - 1);
        if (aligned_next + block > blob_region_bytes)
            return nullptr;
        uint64_t offset = aligned_next;
        char *p = blob_base + offset;
        uint32_t *lenptr = reinterpret_cast<uint32_t *>(p);
        *lenptr = len;
        char *payload = p + hdr;
        payload[len] = '\0';
        blob_next_offset = offset + block;
        return p;
    }

    // free_blob: returns blob to freelist keyed by block_size computed from stored length header.
    // Accepts pointer returned by alloc_blob.
    void free_blob(char *ptr)
    {
        std::lock_guard<std::mutex> lk(mu);
        if (!is_valid() || ptr == nullptr)
            return;
        if (ptr < blob_base || ptr >= blob_base + blob_region_bytes)
            return;

        // header is at ptr (alloc_blob returned header pointer)
        uint32_t stored_len = *reinterpret_cast<uint32_t *>(ptr);
        uint64_t hdr = sizeof(uint32_t);
        uint64_t total = hdr + static_cast<uint64_t>(stored_len) + 1;
        uint64_t block = block_size_for(total);

        uint64_t offset = static_cast<uint64_t>(ptr - blob_base);
        auto &vec = free_lists[block];
        if (vec.size() < MAX_FREELIST_PER_BUCKET)
            vec.push_back(offset);
        // else drop it (we cap freelist per bucket)
    }

    // Number of active (allocated and not freed) seeds
    uint64_t num_active_seeds() const
    {
        std::lock_guard<std::mutex> lk(mu);
        return active_seeds.size();
    }

    // Randomly sample only among active seeds (safe because we maintain active_seeds)
    Seed *get_random_seed()
    {
        std::lock_guard<std::mutex> lk(mu);
        if (active_seeds.empty())
            return nullptr;
        std::uniform_int_distribution<uint64_t> dist(0, active_seeds.size() - 1);
        uint64_t pos = dist(rng);
        uint64_t idx = active_seeds[pos];
        return reinterpret_cast<Seed *>(seed_base + idx * sizeof(Seed));
    }

    uint64_t get_capacity_bytes() const noexcept { return capacity; }

    // Translate a blob-offset stored in seed payload to char* pointer (safe public accessor)
    const char *blob_ptr_from_offset_for_map(uint64_t offset) const
    {
        if (!is_valid())
            return nullptr;
        return ptr_from_blob_offset(offset);
    }

    ~PomaiArena()
    {
        std::lock_guard<std::mutex> lk(mu);
        // Note: we do not call destructors for all seeds here; shards are expected to be torn down when single-writer is quiescent.
        if (base_addr && capacity > 0)
        {
            munmap(base_addr, capacity);
            base_addr = nullptr;
            capacity = 0;

            seed_base = nullptr;
            seed_region_bytes = 0;
            seed_max_slots = 0;
            seed_next_slot = 0;

            blob_base = nullptr;
            blob_region_bytes = 0;
            blob_next_offset = 0;

            free_seeds.clear();
            free_lists.clear();
            active_seeds.clear();
            active_pos.clear();
        }
    }

    // Move semantics (locks not moved; moving an arena is uncommon)
    PomaiArena(PomaiArena &&o) noexcept
        : base_addr(o.base_addr), capacity(o.capacity),
          seed_base(o.seed_base), seed_region_bytes(o.seed_region_bytes),
          seed_max_slots(o.seed_max_slots), seed_next_slot(o.seed_next_slot),
          blob_base(o.blob_base), blob_region_bytes(o.blob_region_bytes), blob_next_offset(o.blob_next_offset),
          free_seeds(std::move(o.free_seeds)), active_seeds(std::move(o.active_seeds)),
          active_pos(std::move(o.active_pos)), rng(std::move(o.rng)), free_lists(std::move(o.free_lists))
    {
        o.base_addr = nullptr;
        o.capacity = 0;
        o.seed_base = nullptr;
        o.blob_base = nullptr;
        o.seed_region_bytes = 0;
        o.blob_region_bytes = 0;
        o.seed_max_slots = 0;
        o.seed_next_slot = 0;
        o.blob_next_offset = 0;
    }

    PomaiArena &operator=(PomaiArena &&o) noexcept
    {
        if (this != &o)
        {
            std::lock_guard<std::mutex> lk(mu);
            if (base_addr && capacity > 0)
                munmap(base_addr, capacity);

            base_addr = o.base_addr;
            capacity = o.capacity;

            seed_base = o.seed_base;
            seed_region_bytes = o.seed_region_bytes;
            seed_max_slots = o.seed_max_slots;
            seed_next_slot = o.seed_next_slot;

            blob_base = o.blob_base;
            blob_region_bytes = o.blob_region_bytes;
            blob_next_offset = o.blob_next_offset;

            free_seeds = std::move(o.free_seeds);
            active_seeds = std::move(o.active_seeds);
            active_pos = std::move(o.active_pos);
            rng = std::move(o.rng);
            free_lists = std::move(o.free_lists);

            o.base_addr = nullptr;
            o.capacity = 0;
            o.seed_base = nullptr;
            o.blob_base = nullptr;
            o.seed_region_bytes = 0;
            o.blob_region_bytes = 0;
            o.seed_max_slots = 0;
            o.seed_next_slot = 0;
            o.blob_next_offset = 0;
        }
        return *this;
    }

private:
    void allocate_region(uint64_t bytes)
    {
        if (bytes == 0)
        {
            base_addr = nullptr;
            capacity = 0;
            seed_base = nullptr;
            seed_region_bytes = 0;
            seed_max_slots = 0;
            blob_base = nullptr;
            blob_region_bytes = 0;
            blob_next_offset = 0;
            return;
        }
        void *p = MAP_FAILED;
#ifdef MAP_HUGETLB
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED)
        {
            base_addr = reinterpret_cast<char *>(p);
            capacity = bytes;
            // continue to partition below
        }
        else
        {
            p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED)
            {
                base_addr = nullptr;
                capacity = 0;
                int err = errno;
                std::cerr << "PomaiArena mmap failed: errno=" << err << " (" << strerror(err) << ")\n";
                return;
            }
            base_addr = reinterpret_cast<char *>(p);
            capacity = bytes;
        }
#else
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
        {
            base_addr = nullptr;
            capacity = 0;
            int err = errno;
            std::cerr << "PomaiArena mmap failed: errno=" << err << " (" << strerror(err) << ")\n";
            return;
        }
        base_addr = reinterpret_cast<char *>(p);
        capacity = bytes;
#endif

        // Partition the region into seed and blob areas.
        seed_region_bytes = static_cast<uint64_t>(static_cast<double>(capacity) * SEED_REGION_RATIO);
        // ensure seed region can hold at least 1 Seed
        if (seed_region_bytes < sizeof(Seed))
            seed_region_bytes = sizeof(Seed);
        // round down seed_region_bytes to multiple of sizeof(Seed)
        seed_region_bytes = (seed_region_bytes / sizeof(Seed)) * sizeof(Seed);
        seed_base = base_addr;
        seed_max_slots = seed_region_bytes / sizeof(Seed);
        seed_next_slot = 0;
        active_seeds.clear();
        active_pos.assign(seed_max_slots, UINT64_MAX);
        free_seeds.clear();

        blob_base = base_addr + seed_region_bytes;
        blob_region_bytes = capacity - seed_region_bytes;
        blob_next_offset = 0;

        // runtime assert: ensure base_addr alignment for Seed
        uintptr_t base_ptr = reinterpret_cast<uintptr_t>(seed_base);
        if ((base_ptr % alignof(Seed)) != 0)
        {
            std::cerr << "Warning: PomaiArena base not aligned to Seed alignment (" << alignof(Seed) << ")\n";
            // Not fatal: unaligned base may still work but performance may suffer.
        }
    }

    // Convert pointer returned by alloc_blob (payload pointer) to blob offset relative to blob_base.
    // Return 0 on error.
    inline uint64_t offset_from_blob_ptr(const char *ptr) const
    {
        if (!blob_base || ptr < blob_base)
            return 0;
        return static_cast<uint64_t>(ptr - blob_base);
    }

    // Convert offset to payload pointer
    inline const char *blob_ptr_from_offset_for_map(uint64_t offset) const
    {
        if (!blob_base || offset >= blob_region_bytes)
            return nullptr;
        return blob_base + offset;
    }

    PomaiArena(const PomaiArena &) = delete;
    PomaiArena &operator=(const PomaiArena &) = delete;
};