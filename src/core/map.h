#pragma once
// core/map.h

#include <vector>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <functional>
#include <queue>

#include "src/core/config.h"
#include "src/core/seed.h"
#include "src/memory/arena.h"
#include "src/core/metrics.h"

namespace pomai::core
{

    // Simple FNV-1a hash
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
        std::vector<Seed *> table_;
        uint64_t size_;
        uint64_t mask_;
        pomai::memory::PomaiArena *arena_;

        uint32_t initial_entropy_;
        uint32_t max_entropy_;
        int harvest_sample_;
        int harvest_max_attempts_;
        size_t max_key_inline_;

        static constexpr size_t PAYLOAD_BYTES = pomai::config::SEED_PAYLOAD_BYTES;
        static constexpr size_t PTR_SZ = pomai::config::MAP_PTR_BYTES;

        void free_existing_blob_if_any(Seed *s)
        {
            if (!s)
                return;
            if ((s->flags & Seed::FLAG_INDIRECT) != 0)
            {
                uint16_t old_klen = s->get_klen();
                uint64_t offset = 0;
                memcpy(&offset, s->payload + old_klen, PTR_SZ);

                char *blob_hdr = const_cast<char *>(arena_->blob_ptr_from_offset_for_map(offset));
                if (blob_hdr)
                {
                    arena_->free_blob(blob_hdr);
                }
                s->flags &= static_cast<uint8_t>(~Seed::FLAG_INDIRECT);
            }
        }

        void write_string_into_seed(Seed *s, const char *key, uint16_t klen, const char *value, uint32_t vlen)
        {
            assert(s != nullptr);
            if (s->is_initialized())
                free_existing_blob_if_any(s);

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
                s->set_meta(klen, vlen16);
                return;
            }

            if (static_cast<size_t>(klen) + PTR_SZ <= PAYLOAD_BYTES)
            {
                char *blob_hdr = arena_->alloc_blob(vlen);
                if (!blob_hdr)
                {
                    PomaiMetrics::arena_alloc_fails.fetch_add(1, std::memory_order_relaxed);
                    return;
                }

                memcpy(blob_hdr + sizeof(uint32_t), value, vlen);
                memcpy(s->payload, key, klen);

                uint64_t offset = arena_->offset_from_blob_ptr(blob_hdr);
                memcpy(s->payload + klen, &offset, PTR_SZ);

                s->flags |= Seed::FLAG_INDIRECT;
                s->entropy = initial_entropy_;
                s->checksum = 0;
                s->set_meta(klen, 0);
            }
        }

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

                const char *blob_hdr = arena_->blob_ptr_from_offset_for_map(offset);
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

        bool remove_seed_from_table(Seed *s)
        {
            if (!s)
                return false;
            for (uint64_t i = 0; i < size_; ++i)
            {
                if (table_[i] == s)
                {
                    table_[i] = nullptr;
                    uint64_t next = (i + 1) & mask_;
                    while (table_[next] != nullptr)
                    {
                        Seed *move_seed = table_[next];
                        table_[next] = nullptr;
                        uint64_t move_hash = fnv1a_hash(move_seed->payload, move_seed->get_klen());
                        uint64_t dest = move_hash & mask_;
                        while (table_[dest] != nullptr)
                            dest = (dest + 1) & mask_;
                        table_[dest] = move_seed;
                        next = (next + 1) & mask_;
                    }
                    return true;
                }
            }
            return false;
        }

        bool harvest_and_put_binary(const char *key, size_t klen, const char *value, size_t vlen)
        {
            if (!arena_)
                return false;

            Seed *best = nullptr;
            uint32_t best_entropy = UINT32_MAX;
            std::time_t now = std::time(nullptr);

            int attempts = 0;
            int sampled = 0;

            while (sampled < harvest_sample_ && attempts < harvest_max_attempts_)
            {
                ++attempts;
                Seed *cand = arena_->get_random_seed();
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

                if (cand->entropy < best_entropy)
                {
                    best_entropy = cand->entropy;
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
            uint64_t idx = h & mask_;
            while (table_[idx] != nullptr)
            {
                idx = (idx + 1) & mask_;
            }
            table_[idx] = best;
            return true;
        }

    public:
        PomaiMap(pomai::memory::PomaiArena *arena, uint64_t slots, const pomai::config::PomaiConfig &cfg)
            : table_(slots, nullptr),
              size_(slots),
              mask_(slots - 1),
              arena_(arena),
              initial_entropy_(cfg.map_tuning.initial_entropy),
              max_entropy_(cfg.map_tuning.max_entropy),
              harvest_sample_(cfg.map_tuning.harvest_sample),
              harvest_max_attempts_(cfg.map_tuning.harvest_max_attempts),
              max_key_inline_(cfg.map_tuning.max_key_inline)
        {
            assert((slots & (slots - 1)) == 0 && "slots must be power of two");
        }

        bool put(const char *key, size_t klen, const char *value, size_t vlen)
        {
            PomaiMetrics::puts.fetch_add(1, std::memory_order_relaxed);
            assert(klen <= max_key_inline_ && "Key too long");
            if (klen == 0)
                return false;

            uint64_t hash = fnv1a_hash(key, klen);
            uint64_t idx = hash & mask_;
            uint64_t start_idx = idx;

            while (table_[idx] != nullptr)
            {
                Seed *s = table_[idx];
                if (s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
                {
                    write_string_into_seed(s, key, static_cast<uint16_t>(klen), value, static_cast<uint32_t>(vlen));
                    return true;
                }
                idx = (idx + 1) & mask_;
                if (idx == start_idx)
                    return harvest_and_put_binary(key, klen, value, vlen);
            }

            Seed *s = arena_->alloc_seed();
            if (!s)
                return harvest_and_put_binary(key, klen, value, vlen);

            write_string_into_seed(s, key, static_cast<uint16_t>(klen), value, static_cast<uint32_t>(vlen));
            table_[idx] = s;
            return true;
        }

        const char *get(const char *key, size_t klen, uint32_t *out_len)
        {
            uint64_t hash = fnv1a_hash(key, klen);
            uint64_t idx = hash & mask_;
            uint64_t start_idx = idx;

            while (table_[idx] != nullptr)
            {
                Seed *s = table_[idx];
                if (s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
                {
                    if (s->entropy < max_entropy_)
                        ++s->entropy;
                    PomaiMetrics::hits.fetch_add(1, std::memory_order_relaxed);
                    return extract_string_from_seed(s, static_cast<uint16_t>(klen), out_len);
                }
                if (s->entropy > 0)
                    --s->entropy;
                idx = (idx + 1) & mask_;
                if (idx == start_idx)
                    break;
            }
            PomaiMetrics::misses.fetch_add(1, std::memory_order_relaxed);
            if (out_len)
                *out_len = 0;
            return nullptr;
        }

        bool erase(const char *key)
        {
            size_t klen = strlen(key);
            uint64_t hash = fnv1a_hash(key, klen);
            uint64_t idx = hash & mask_;
            uint64_t start_idx = idx;

            while (table_[idx] != nullptr)
            {
                Seed *s = table_[idx];
                if (s->get_klen() == klen && memcmp(s->payload, key, klen) == 0)
                {
                    if ((s->flags & Seed::FLAG_INDIRECT) != 0)
                        free_existing_blob_if_any(s);
                    s->header.store(0ULL, std::memory_order_release);
                    table_[idx] = nullptr;
                    arena_->free_seed(s);

                    uint64_t next = (idx + 1) & mask_;
                    while (table_[next] != nullptr)
                    {
                        Seed *m = table_[next];
                        table_[next] = nullptr;
                        uint64_t h = fnv1a_hash(m->payload, m->get_klen());
                        uint64_t d = h & mask_;
                        while (table_[d] != nullptr)
                            d = (d + 1) & mask_;
                        table_[d] = m;
                        next = (next + 1) & mask_;
                    }
                    return true;
                }
                idx = (idx + 1) & mask_;
                if (idx == start_idx)
                    break;
            }
            return false;
        }

        size_t size_count() const
        {
            size_t cnt = 0;
            for (auto p : table_)
                if (p && p->is_initialized())
                    ++cnt;
            return cnt;
        }

        void clear()
        {
            for (auto &p : table_)
            {
                if (p)
                {
                    if ((p->flags & Seed::FLAG_INDIRECT) != 0)
                        free_existing_blob_if_any(p);
                    p->header.store(0ULL, std::memory_order_release);
                    arena_->free_seed(p);
                }
                p = nullptr;
            }
        }

        void for_each_seed(const std::function<void(Seed *)> &cb) const
        {
            for (auto p : table_)
            {
                if (p != nullptr && p->is_initialized())
                    cb(p);
            }
        }

        template <typename Func>
        void scan_all(Func &&f) const
        {
            uint32_t now = static_cast<uint32_t>(std::time(nullptr));
            for (uint64_t i = 0; i < size_; ++i)
            {
                Seed *s = table_[i];
                if (s == nullptr || !s->is_initialized())
                    continue;
                uint32_t expiry = s->get_expiry();
                if (expiry != 0 && expiry < now)
                    continue;
                f(s);
            }
        }

        pomai::memory::PomaiArena *get_arena() const { return arena_; }
    };

} // namespace pomai::core