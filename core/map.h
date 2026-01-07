#pragma once
// core/map.h
//
// PomaiMap: open-addressing (linear probing) hash table storing pointers to Seeds
// located inside a PomaiArena.
//
// NOTE (English):
// - The arena stores blobs in a separate region and seeds contain a 64-bit offset
//   to the blob header (relative to blob_base). We never store raw pointers into seeds anymore.
// - PomaiMap translates offsets -> pointers via PomaiArena::blob_ptr_from_offset_for_map().
// - All blob frees/allocs must use arena APIs which are thread-safe via the arena mutex.

#include <vector>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <functional>
#include <queue>

#include "core/config.h"
#include "core/seed.h"
#include "memory/arena.h"
#include "core/metrics.h"

// Simple FNV-1a hash for strings - zero-dependency, good-enough for prototype
inline uint64_t fnv1a_hash(const char *data, size_t len)
{
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i)
    {
        hash ^= static_cast<uint8_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

class PomaiMap
{
private:
    std::vector<Seed *> table; // array of pointers into the Arena
    uint64_t size;
    uint64_t mask;
    PomaiArena *arena; // Map does NOT own arena

    // Tunables (snapshotted from pomai::config::runtime at construction)
    uint32_t initial_entropy_;
    uint32_t max_entropy_;
    int harvest_sample_;
    int harvest_max_attempts_;

    // Payload/layout constants taken from central config
    static constexpr size_t PAYLOAD_BYTES = pomai::config::SEED_PAYLOAD_BYTES;
    static constexpr size_t MAX_KEY_INLINE = pomai::config::MAP_MAX_INLINE_KEY;
    static constexpr size_t PTR_SZ = pomai::config::MAP_PTR_BYTES; // blob offset size (uint64_t)

    // Helper - free any indirect blob referenced by a Seed.
    void free_existing_blob_if_any(Seed *s)
    {
        if (!s)
            return;
        if ((s->flags & Seed::FLAG_INDIRECT) != 0)
        {
            uint16_t old_klen = s->get_klen();
            uint64_t offset = 0;
            memcpy(&offset, s->payload + old_klen, PTR_SZ);
            char *blob_hdr = const_cast<char *>(arena->blob_ptr_from_offset_for_map(offset));
            if (blob_hdr)
            {
                arena->free_blob(blob_hdr);
            }
            s->flags &= static_cast<uint8_t>(~Seed::FLAG_INDIRECT);
        }
    }

    // Write key+value into seed. If seed already initialized with an indirect blob, free it first.
    void write_string_into_seed(Seed *s, const char *key, uint16_t klen, const char *value, uint32_t vlen)
    {
        assert(s != nullptr);

        if (s->is_initialized())
        {
            free_existing_blob_if_any(s);
        }

        s->flags = 0;
        s->type = Seed::OBJ_STRING;

        if (static_cast<size_t>(klen) + static_cast<size_t>(vlen) + 1 <= PAYLOAD_BYTES)
        {
            memcpy(s->payload, key, klen);
            memcpy(s->payload + klen, value, vlen);
            s->payload[klen + vlen] = '\0';
            s->entropy = initial_entropy_;
            s->checksum = 0;
            uint16_t vlen16 = static_cast<uint16_t>(vlen <= 0xFFFF ? vlen : 0xFFFF);
            s->set_meta(klen, vlen16); // publish header
            return;
        }

        if (static_cast<size_t>(klen) + PTR_SZ <= PAYLOAD_BYTES)
        {
            char *blob_hdr = arena->alloc_blob(vlen);
            if (!blob_hdr)
            {
                PomaiMetrics::arena_alloc_fails.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            memcpy(blob_hdr + sizeof(uint32_t), value, vlen);
            memcpy(s->payload, key, klen);

            uint64_t offset = arena->offset_from_blob_ptr(blob_hdr);
            memcpy(s->payload + klen, &offset, PTR_SZ);

            s->flags |= Seed::FLAG_INDIRECT;
            s->entropy = initial_entropy_;
            s->checksum = 0;
            s->set_meta(klen, 0);
            return;
        }

        return;
    }

    // Extract pointer to value data and optionally its length.
    const char *extract_string_from_seed(const Seed *s, uint16_t klen, uint32_t *out_vlen) const
    {
        if ((s->flags & Seed::FLAG_INDIRECT) == 0)
        {
            if (out_vlen)
                *out_vlen = s->get_vlen();
            return s->payload + klen;
        }
        else
        {
            uint64_t offset = 0;
            memcpy(&offset, s->payload + klen, PTR_SZ);
            const char *blob_hdr = arena->blob_ptr_from_offset_for_map(offset);
            if (!blob_hdr)
            {
                if (out_vlen)
                    *out_vlen = 0;
                return nullptr;
            }
            uint32_t blen = *reinterpret_cast<const uint32_t *>(blob_hdr);
            if (out_vlen)
                *out_vlen = blen;
            return blob_hdr + sizeof(uint32_t);
        }
    }

    // Remove a specific Seed* pointer from the table and shift the cluster backward.
    bool remove_seed_from_table(Seed *s)
    {
        if (!s)
            return false;
        for (uint64_t i = 0; i < size; ++i)
        {
            if (table[i] == s)
            {
                table[i] = nullptr;

                // Shift subsequent cluster entries backward to fill the hole
                uint64_t next = (i + 1) & mask;
                while (table[next] != nullptr)
                {
                    Seed *move_seed = table[next];
                    table[next] = nullptr;

                    uint64_t move_hash = fnv1a_hash(move_seed->payload, move_seed->get_klen());
                    uint64_t dest = move_hash & mask;
                    while (table[dest] != nullptr)
                        dest = (dest + 1) & mask;

                    table[dest] = move_seed;
                    next = (next + 1) & mask;
                }
                return true;
            }
        }
        return false;
    }

    // Harvest & eviction (unchanged)...
    bool harvest_and_put_binary(const char *key, size_t klen, const char *value, size_t vlen)
    {
        if (!arena)
            return false;

        Seed *best = nullptr;
        uint32_t best_entropy = UINT32_MAX;
        std::time_t now = std::time(nullptr);

        int attempts = 0;
        int sampled = 0;

        while (sampled < harvest_sample_ && attempts < harvest_max_attempts_)
        {
            ++attempts;
            Seed *cand = arena->get_random_seed();
            if (!cand)
                continue;

            if (!cand->is_initialized())
            {
                best = cand;
                best_entropy = 0;
                break;
            }

            uint32_t expiry = cand->get_expiry();
            if (expiry != 0 && static_cast<std::time_t>(expiry) <= now)
            {
                best = cand;
                best_entropy = 0;
                break;
            }

            uint32_t e = cand->entropy;
            if (e < best_entropy)
            {
                best_entropy = e;
                best = cand;
            }
            ++sampled;
        }

        if (!best)
            return false;

        if (remove_seed_from_table(best))
            PomaiMetrics::evictions.fetch_add(1, std::memory_order_relaxed);
        else
            PomaiMetrics::harvests.fetch_add(1, std::memory_order_relaxed);

        write_string_into_seed(best, key, static_cast<uint16_t>(klen), value, static_cast<uint32_t>(vlen));

        uint64_t h = fnv1a_hash(key, klen);
        uint64_t idx = h & mask;
        uint64_t start_idx = idx;
        while (table[idx] != nullptr)
        {
            idx = (idx + 1) & mask;
            if (idx == start_idx)
                break;
        }
        table[idx] = best;
        return true;
    }

public:
    PomaiMap(PomaiArena *a, uint64_t size_power_of_2) : arena(a),
                                                        initial_entropy_(static_cast<uint32_t>(pomai::config::runtime.initial_entropy)),
                                                        max_entropy_(static_cast<uint32_t>(pomai::config::runtime.max_entropy)),
                                                        harvest_sample_(static_cast<int>(pomai::config::runtime.harvest_sample)),
                                                        harvest_max_attempts_(static_cast<int>(pomai::config::runtime.harvest_max_attempts))
    {
        assert((size_power_of_2 & (size_power_of_2 - 1)) == 0 && "size must be power of two");
        size = size_power_of_2;
        mask = size - 1;
        table.resize(size, nullptr);
    }

    // Binary-safe put
    bool put(const char *key, size_t klen, const char *value, size_t vlen)
    {
        PomaiMetrics::puts.fetch_add(1, std::memory_order_relaxed);

        assert(klen <= MAX_KEY_INLINE && "Key too long for inline storage in Seed.payload");
        if (klen == 0)
            return false;

        uint64_t hash = fnv1a_hash(key, klen);
        uint64_t idx = hash & mask;
        uint64_t start_idx = idx;

        while (table[idx] != nullptr)
        {
            Seed *s = table[idx];
            if (s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
            {
                write_string_into_seed(s, key, static_cast<uint16_t>(klen), value, static_cast<uint32_t>(vlen));
                return true;
            }

            idx = (idx + 1) & mask;
            if (idx == start_idx)
            {
                return harvest_and_put_binary(key, klen, value, vlen);
            }
        }

        Seed *s = arena->alloc_seed();
        if (!s)
        {
            PomaiMetrics::arena_alloc_fails.fetch_add(1, std::memory_order_relaxed);
            return harvest_and_put_binary(key, klen, value, vlen);
        }

        write_string_into_seed(s, key, static_cast<uint16_t>(klen), value, static_cast<uint32_t>(vlen));
        table[idx] = s;
        return true;
    }

    bool put(const char *key, const char *value)
    {
        return put(key, strlen(key), value, strlen(value));
    }

    const char *get(const char *key, size_t klen, uint32_t *out_len)
    {
        uint64_t hash = fnv1a_hash(key, klen);
        uint64_t idx = hash & mask;
        uint64_t start_idx = idx;

        while (table[idx] != nullptr)
        {
            Seed *s = table[idx];
            if (!s)
            {
                idx = (idx + 1) & mask;
                if (idx == start_idx)
                    break;
                continue;
            }

            (void)s->header.load(std::memory_order_acquire);

            if (s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
            {
                if (s->entropy < max_entropy_)
                    ++s->entropy;
                PomaiMetrics::hits.fetch_add(1, std::memory_order_relaxed);
                return extract_string_from_seed(s, static_cast<uint16_t>(klen), out_len);
            }

            if (s->entropy > 0)
                --s->entropy;

            idx = (idx + 1) & mask;
            if (idx == start_idx)
                break;
        }

        PomaiMetrics::misses.fetch_add(1, std::memory_order_relaxed);
        if (out_len)
            *out_len = 0;
        return nullptr;
    }

    const char *get(const char *key)
    {
        uint32_t l = 0;
        return get(key, strlen(key), &l);
    }

    bool erase(const char *key)
    {
        size_t klen = strlen(key);
        uint64_t hash = fnv1a_hash(key, klen);
        uint64_t idx = hash & mask;
        uint64_t start_idx = idx;

        while (table[idx] != nullptr)
        {
            Seed *s = table[idx];
            if (s && s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
            {
                if ((s->flags & Seed::FLAG_INDIRECT) != 0)
                    free_existing_blob_if_any(s);

                s->header.store(0ULL, std::memory_order_release);
                s->entropy = 0;
                s->checksum = 0;
                s->type = 0;
                s->flags = 0;

                table[idx] = nullptr;
                arena->free_seed(s);

                uint64_t next = (idx + 1) & mask;
                while (table[next] != nullptr)
                {
                    Seed *move_seed = table[next];
                    table[next] = nullptr;
                    uint64_t move_hash = fnv1a_hash(move_seed->payload, move_seed->get_klen());
                    uint64_t dest = move_hash & mask;
                    while (table[dest] != nullptr)
                        dest = (dest + 1) & mask;
                    table[dest] = move_seed;
                    next = (next + 1) & mask;
                }

                return true;
            }

            idx = (idx + 1) & mask;
            if (idx == start_idx)
                break;
        }
        return false;
    }

    size_t size_count() const
    {
        size_t cnt = 0;
        for (auto p : table)
            if (p != nullptr && p->is_initialized())
                ++cnt;
        return cnt;
    }

    void clear()
    {
        for (auto &p : table)
        {
            if (p)
            {
                if ((p->flags & Seed::FLAG_INDIRECT) != 0)
                    free_existing_blob_if_any(p);

                p->header.store(0ULL, std::memory_order_release);
                p->entropy = 0;
                p->checksum = 0;
                p->type = 0;
                p->flags = 0;
                arena->free_seed(p);
            }
            p = nullptr;
        }
    }

    // find_seed helper
    Seed *find_seed(const char *key, size_t klen)
    {
        uint64_t hash = fnv1a_hash(key, klen);
        uint64_t idx = hash & mask;
        uint64_t start_idx = idx;
        while (table[idx] != nullptr)
        {
            Seed *s = table[idx];
            if (s && s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
                return s;
            idx = (idx + 1) & mask;
            if (idx == start_idx)
                break;
        }
        return nullptr;
    }

    Seed *find_seed(const char *key) { return find_seed(key, strlen(key)); }

    // Expose arena pointer for external modules (read-only)
    PomaiArena *get_arena() const { return arena; }

    // Visitor to iterate initialized seeds (safe snapshot of pointers)
    // Callback is invoked for each Seed* that is non-null and is_initialized().
    // Note: the callback executes in caller thread; ensure single-writer invariant or external synchronization.
    void for_each_seed(const std::function<void(Seed *)> &cb) const
    {
        for (auto p : table)
        {
            if (p != nullptr && p->is_initialized())
                cb(p);
        }
    }

    // New: scan_all - lightweight scanner with expiry check and timestamp snapshot.
    // Template so caller can pass any callable; this avoids allocations and enables inlining.
    // WARNING: This is a snapshot-style scan. It iterates the table and calls f(s) for each
    // initialized seed. Callers must obey single-writer or external synchronization rules.
    template <typename Func>
    void scan_all(Func &&f) const
    {
        uint32_t now = static_cast<uint32_t>(std::time(nullptr));
        // iterate by index to avoid copies and allow future locality-aware scanning
        for (uint64_t i = 0; i < size; ++i)
        {
            Seed *s = table[i];
            if (s == nullptr)
                continue;
            if (!s->is_initialized())
                continue;
            uint32_t expiry = s->get_expiry();
            if (expiry != 0 && expiry < now)
                continue; // skip expired seeds
            f(s);
        }
    }
};