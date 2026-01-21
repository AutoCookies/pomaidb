/*
 * src/ai/pomai_orbit.cc
 *
 * Production-Grade Implementation: Safe Writes, Atomic IdsBlock, Zero-Copy Reads.
 * Optimized for robustness: Implements WAL-centric recovery for ephemeral arenas.
 */

#include "src/ai/pomai_orbit.h"

#include "src/ai/atomic_utils.h"
#include "src/ai/ids_block.h"
#include "src/memory/arena.h"
#include "src/memory/shard_arena.h"
#include "src/ai/eternalecho_quantizer.h"
#include "src/ai/whispergrain.h"
#include "src/core/cpu_kernels.h"
#include "src/core/metrics.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>
#include <unordered_set>
#include <unistd.h>
#include <sys/mman.h>

namespace pomai::ai::orbit
{
    using namespace pomai::ai::soa; // For IdEntry

    // Constants
    static constexpr size_t MAX_ECHO_BYTES = 64;
    static constexpr size_t DEFAULT_MAX_SUBBATCH = 4096;

    // --- Helpers ---

    static std::optional<const char *> resolve_bucket_base(const ArenaView &arena, uint64_t bucket_off, std::vector<char> &temp_buffer)
    {
        if (bucket_off == 0)
            return std::nullopt;

        // Try RAM access first (Fast Path)
        const char *ram_blob_ptr = arena.blob_ptr_from_offset_for_map(bucket_off);
        if (ram_blob_ptr)
        {
            // Skip the arena's length prefix (uint32_t) to point to user data (BucketHeader)
            return ram_blob_ptr + sizeof(uint32_t);
        }

        // Fallback: Read from remote/disk (Slow Path)
        temp_buffer = arena.read_remote_blob(bucket_off);
        if (temp_buffer.empty())
            return std::nullopt;

        return temp_buffer.data() + sizeof(uint32_t);
    }

    // --- ArenaView Methods ---
    char *ArenaView::alloc_blob(uint32_t len) const
    {
        if (pa)
            return pa->alloc_blob(len);
        if (sa)
            return sa->alloc_blob(len);
        return nullptr;
    }
    void ArenaView::demote_range(uint64_t offset, size_t len) const
    {
        if (sa)
            sa->demote_range(offset, len);
    }
    uint64_t ArenaView::offset_from_blob_ptr(const char *p) const noexcept
    {
        if (pa)
            return pa->offset_from_blob_ptr(p);
        if (sa)
            return sa->offset_from_blob_ptr(p);
        return UINT64_MAX;
    }
    const char *ArenaView::blob_ptr_from_offset_for_map(uint64_t offset) const
    {
        if (pa)
            return pa->blob_ptr_from_offset_for_map(offset);
        if (sa)
            return sa->blob_ptr_from_offset_for_map(offset);
        return nullptr;
    }
    std::vector<char> ArenaView::read_remote_blob(uint64_t remote_id) const
    {
        if (pa)
            return pa->read_remote_blob(remote_id);
        if (sa)
            return sa->read_remote_blob(remote_id);
        return {};
    }

    // --- Atomic Helper Wrappers ---
    static inline uint32_t bucket_atomic_load_count(const BucketHeader *hdr)
    {
        return pomai::ai::atomic_utils::atomic_load_u32(&hdr->count);
    }
    static inline uint32_t bucket_atomic_fetch_add_count(BucketHeader *hdr, uint32_t arg)
    {
        return pomai::ai::atomic_utils::atomic_fetch_add_u32(&hdr->count, arg);
    }
    static inline void bucket_atomic_sub_count(BucketHeader *hdr, uint32_t arg)
    {
        pomai::ai::atomic_utils::atomic_fetch_sub_u32(&hdr->count, arg);
    }
    static inline uint64_t bucket_atomic_load_next(const BucketHeader *hdr)
    {
        return pomai::ai::atomic_utils::atomic_load_u64(&hdr->next_bucket_offset);
    }
    static inline void bucket_atomic_store_next(BucketHeader *hdr, uint64_t val)
    {
        pomai::ai::atomic_utils::atomic_store_u64(&hdr->next_bucket_offset, val);
    }

    // --- Constructors ---
    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        dynamic_bucket_capacity_ = (cfg_.algo.initial_bucket_cap > 0) ? cfg_.algo.initial_bucket_cap : 128;
        if (!arena_.is_pomai_arena())
            throw std::invalid_argument("PomaiOrbit: arena null");
        if (cfg_.dim == 0)
            throw std::invalid_argument("PomaiOrbit: dim must be > 0");

        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
        schema_file_path_ = (p / "pomai_schema.bin").string();

        eeq_ = std::make_unique<pomai::ai::EternalEchoQuantizer>(cfg_.dim, cfg_.eeq_cfg);

        // 1. Load Schema (Centroids) if exists
        if (std::filesystem::exists(schema_file_path_))
            load_schema();
        else
            save_schema();

        if (cfg_.use_cortex)
        {
            try
            {
                cortex_ = std::make_unique<NetworkCortex>(cfg_.cortex_cfg);
                cortex_->start();
            }
            catch (...)
            {
            }
        }

        // 2. Open WAL
        wal_ = std::make_unique<pomai::memory::WalManager>();
        pomai::memory::WalManager::WalConfig wcfg;
        wcfg.sync_on_append = true;
        std::string wal_path = cfg_.data_path + "/orbit.wal";

        // 3. Replay WAL to restore data into Arena
        if (!wal_->open(wal_path, true, wcfg))
            std::cerr << "[Orbit] FATAL: Wal open failed\n";
        else
            recover_from_wal();

        if (!centroids_.empty())
            init_thermal_map(centroids_.size());
    }

    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::ShardArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        dynamic_bucket_capacity_ = (cfg_.algo.initial_bucket_cap > 0) ? cfg_.algo.initial_bucket_cap : 128;
        if (!arena_.is_shard_arena())
            throw std::invalid_argument("PomaiOrbit: shard arena null");
        if (cfg_.dim == 0)
            throw std::invalid_argument("PomaiOrbit: dim must be > 0");

        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
        schema_file_path_ = (p / "pomai_schema.bin").string();

        eeq_ = std::make_unique<pomai::ai::EternalEchoQuantizer>(cfg_.dim, cfg_.eeq_cfg);

        if (std::filesystem::exists(schema_file_path_))
            load_schema();
        else
            save_schema();

        if (cfg_.use_cortex)
        {
            try
            {
                cortex_ = std::make_unique<NetworkCortex>(cfg_.cortex_cfg);
                cortex_->start();
            }
            catch (...)
            {
            }
        }

        wal_ = std::make_unique<pomai::memory::WalManager>();
        pomai::memory::WalManager::WalConfig wcfg;
        wcfg.sync_on_append = true;
        std::string wal_path = cfg_.data_path + "/orbit.wal";
        if (!wal_->open(wal_path, true, wcfg))
            std::cerr << "[Orbit] FATAL: Wal open failed\n";
        else
            recover_from_wal();

        if (!centroids_.empty())
            init_thermal_map(centroids_.size());
    }

    PomaiOrbit::~PomaiOrbit()
    {
        if (cortex_)
            cortex_->stop();
    }

    void PomaiOrbit::recover_from_wal()
    {
        size_t replayed_count = 0;
        auto replayer = [&](uint16_t type, const void *payload, uint32_t len, uint64_t) -> bool
        {
            if (type == 20 && len >= 4)
            {
                const uint8_t *ptr = static_cast<const uint8_t *>(payload);
                uint32_t count = 0;
                std::memcpy(&count, ptr, 4);
                ptr += 4;
                std::vector<std::pair<uint64_t, std::vector<float>>> batch;
                size_t vec_bytes = cfg_.dim * sizeof(float);
                // Safe check len
                if (len < 4 + count * (8 + vec_bytes))
                    return true; // Malformed record, skip

                for (uint32_t i = 0; i < count; ++i)
                {
                    uint64_t label;
                    std::memcpy(&label, ptr, 8);
                    ptr += 8;
                    std::vector<float> vec(cfg_.dim);
                    std::memcpy(vec.data(), ptr, vec_bytes);
                    ptr += vec_bytes;
                    batch.emplace_back(label, std::move(vec));
                }
                if (!batch.empty())
                {
                    // Insert into memory only (do not write to WAL again during replay)
                    if (insert_batch_memory_only(batch))
                    {
                        replayed_count += batch.size();
                    }
                }
            }
            return true;
        };
        wal_->replay(replayer);
        if (replayed_count > 0)
            std::clog << "[Orbit] Recovered " << replayed_count << " vectors from WAL.\n";
    }

    // --- Thermal ---
    void PomaiOrbit::init_thermal_map(size_t num_centroids)
    {
        if (num_centroids == 0)
            return;
        if (thermal_map_.size() != num_centroids)
        {
            thermal_map_ = std::vector<std::atomic<uint8_t>>(num_centroids);
            last_access_epoch_ = std::vector<std::atomic<uint32_t>>(num_centroids);
        }
        uint32_t now = static_cast<uint32_t>(std::time(nullptr));
        for (size_t i = 0; i < num_centroids; ++i)
        {
            thermal_map_[i].store(100, std::memory_order_relaxed);
            last_access_epoch_[i].store(now, std::memory_order_relaxed);
        }
    }
    void PomaiOrbit::touch_centroid(uint32_t cid)
    {
        if (cid >= thermal_map_.size())
            return;
        thermal_map_[cid].store(255, std::memory_order_relaxed);
        last_access_epoch_[cid].store(static_cast<uint32_t>(std::time(nullptr)), std::memory_order_relaxed);
    }
    uint8_t PomaiOrbit::get_temperature(uint32_t cid) const
    {
        if (cid >= thermal_map_.size())
            return 0;
        return thermal_map_[cid].load(std::memory_order_relaxed);
    }
    void PomaiOrbit::apply_thermal_policy() { /* impl logic if needed */ }

    // --- Schema IO ---
    void PomaiOrbit::save_schema()
    {
        SchemaHeader header;
        header.magic_number = 0x504F4D41;
        header.version = 2;
        header.dim = cfg_.dim;
        header.num_centroids = centroids_.size();
        uint64_t total = 0;
        for (const auto &sh : label_shards_)
        {
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            total += sh.bucket.size();
        }
        header.total_vectors = total;

        std::string tmp = schema_file_path_ + ".tmp";
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (out.is_open())
        {
            out.write(reinterpret_cast<const char *>(&header), sizeof(header));
            for (const auto &c : centroids_)
            {
                out.write(reinterpret_cast<const char *>(c->vector.data()), cfg_.dim * sizeof(float));
                uint64_t off = c->bucket_offset.load(std::memory_order_acquire);
                out.write(reinterpret_cast<const char *>(&off), sizeof(off));
            }
            out.close();
            std::filesystem::rename(tmp, schema_file_path_);
        }
        else
        {
            std::cerr << "[Orbit] Failed to write schema to " << tmp << "\n";
        }
    }

    bool PomaiOrbit::load_schema()
    {
        std::ifstream in(schema_file_path_, std::ios::binary);
        if (!in.is_open())
            return false;
        SchemaHeader header;
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (header.magic_number != 0x504F4D41)
            return false;
        cfg_.dim = header.dim;
        cfg_.algo.num_centroids = static_cast<uint32_t>(header.num_centroids);
        centroids_.resize(header.num_centroids);
        for (size_t i = 0; i < header.num_centroids; ++i)
        {
            auto node = std::make_unique<OrbitNode>();
            node->vector.resize(cfg_.dim);
            in.read(reinterpret_cast<char *>(node->vector.data()), cfg_.dim * sizeof(float));
            uint64_t off_disk = 0;
            in.read(reinterpret_cast<char *>(&off_disk), sizeof(off_disk));

            // [CRITICAL FIX] Reset bucket offsets to 0.
            // In a crash/restart scenario with ephemeral Arena (like temp file mmap),
            // the old offsets point to garbage. We must force rebuild from WAL.
            // Ideally, we'd check if Arena is persistent, but assuming worst case is safer.
            node->bucket_offset.store(0, std::memory_order_relaxed);

            centroids_[i] = std::move(node);
        }
        rebuild_index(); // This will effectively do nothing if offsets are 0, which is correct.
        return true;
    }

    void PomaiOrbit::rebuild_index()
    {
        // std::clog << "[Orbit] Rebuilding index...\n";
        size_t recovered_count = 0;
        for (const auto &c : centroids_)
        {
            uint64_t curr = c->bucket_offset.load(std::memory_order_relaxed);
            while (curr != 0)
            {
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, curr, tmp);
                if (!base_opt)
                    break; // Should not happen if curr != 0 valid
                const char *ptr = *base_opt;
                BucketHeader hdr;
                std::memcpy(&hdr, ptr, sizeof(BucketHeader));
                uint32_t n = hdr.count;
                if (n > 1000000)
                    break;
                const char *ids_ptr = ptr + hdr.off_ids;
                for (uint32_t i = 0; i < n; ++i)
                {
                    uint64_t packed;
                    std::memcpy(&packed, ids_ptr + i * 8, 8);
                    if (IdEntry::tag_of(packed) == IdEntry::TAG_LABEL)
                    {
                        set_label_map(IdEntry::payload_of(packed), curr, i);
                        recovered_count++;
                    }
                }
                curr = hdr.next_bucket_offset;
            }
        }
        // std::clog << "[Orbit] Rebuilt index: " << recovered_count << " vectors.\n";

        // Re-init routing neighbors
        if (!centroids_.empty())
        {
            L2Func kern = get_pomai_l2sq_kernel();
            for (size_t i = 0; i < centroids_.size(); ++i)
            {
                std::vector<std::pair<float, uint32_t>> dists;
                for (size_t j = 0; j < centroids_.size(); ++j)
                {
                    if (i == j)
                        continue;
                    float d = kern(centroids_[i]->vector.data(), centroids_[j]->vector.data(), cfg_.dim);
                    dists.push_back({d, (uint32_t)j});
                }
                std::sort(dists.begin(), dists.end());
                std::lock_guard<std::shared_mutex> lk(centroids_[i]->mu);
                centroids_[i]->neighbors.clear();
                for (size_t k = 0; k < std::min(dists.size(), (size_t)32); ++k)
                    centroids_[i]->neighbors.push_back(dists[k].second);
            }
        }
    }

    // --- Core Logic ---
    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0)
            return false;
        if (cfg_.algo.num_centroids == 0)
            cfg_.algo.num_centroids = 64;
        size_t num_c = std::min(n, static_cast<size_t>(cfg_.algo.num_centroids));
        centroids_.resize(num_c);
        for (size_t i = 0; i < num_c; ++i)
        {
            centroids_[i] = std::make_unique<OrbitNode>();
            centroids_[i]->vector.resize(cfg_.dim);
            std::memcpy(centroids_[i]->vector.data(), data + i * cfg_.dim, cfg_.dim * sizeof(float));
            // Pre-allocate first bucket
            uint64_t off = alloc_new_bucket(i);
            if (off == 0)
            {
                std::cerr << "[Orbit] Train: Failed to allocate bucket for centroid " << i << "\n";
                return false;
            }
            centroids_[i]->bucket_offset.store(off, std::memory_order_release);
        }
        rebuild_index();
        save_schema();
        init_thermal_map(num_c);
        return true;
    }

    uint64_t PomaiOrbit::alloc_new_bucket(uint32_t cid)
    {
        size_t cap = dynamic_bucket_capacity_;
        auto align = [](size_t s)
        { return (s + 63) & ~63; };
        size_t head_sz = sizeof(BucketHeader);
        size_t code_sz = sizeof(uint16_t) * cap;
        size_t vec_sz = MAX_ECHO_BYTES * cap;
        size_t ids_sz = sizeof(uint64_t) * cap;
        uint32_t off_fp = align(head_sz);
        uint32_t off_pq = align(off_fp);
        uint32_t off_vec = align(off_pq + code_sz);
        uint32_t off_ids = align(off_vec + vec_sz);
        size_t total = off_ids + ids_sz;

        char *blob = arena_.alloc_blob(total);
        if (!blob)
            return 0;

        BucketHeader hdr;
        std::memset(&hdr, 0, sizeof(hdr));
        hdr.centroid_id = cid;
        hdr.off_pq_codes = off_pq;
        hdr.off_vectors = off_vec;
        hdr.off_ids = off_ids;
        // Memcpy + 4 to skip arena length prefix
        std::memcpy(blob + 4, &hdr, sizeof(hdr));
        return arena_.offset_from_blob_ptr(blob);
    }

    uint32_t PomaiOrbit::find_nearest_centroid(const float *vec)
    {
        uint32_t best = 0;
        float min_d = 1e30f;
        L2Func kern = get_pomai_l2sq_kernel();
        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            float d = kern(vec, centroids_[i]->vector.data(), cfg_.dim);
            if (d < min_d)
            {
                min_d = d;
                best = i;
            }
        }
        return best;
    }

    std::vector<uint32_t> PomaiOrbit::find_routing_centroids(const float *vec, size_t n)
    {
        if (centroids_.empty())
            return {};
        using P = std::pair<float, uint32_t>;
        std::vector<P> all;
        L2Func kern = get_pomai_l2sq_kernel();
        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            all.push_back({kern(vec, centroids_[i]->vector.data(), cfg_.dim), (uint32_t)i});
        }
        std::sort(all.begin(), all.end());
        std::vector<uint32_t> res;
        for (size_t i = 0; i < std::min(n, all.size()); ++i)
            res.push_back(all[i].second);
        return res;
    }

    // --- Insert ---
    void PomaiOrbit::set_label_map(uint64_t label, uint64_t bucket, uint32_t slot)
    {
        auto &sh = label_shards_[label_shard_index(label)];
        std::unique_lock<std::shared_mutex> lk(sh.mu);
        sh.bucket[label] = bucket;
        sh.slot[label] = slot;
    }
    bool PomaiOrbit::get_label_bucket(uint64_t label, uint64_t &b) const
    {
        auto &sh = label_shards_[label_shard_index(label)];
        std::shared_lock<std::shared_mutex> lk(sh.mu);
        auto it = sh.bucket.find(label);
        if (it == sh.bucket.end())
            return false;
        b = it->second;
        return true;
    }
    bool PomaiOrbit::get_label_slot(uint64_t label, uint32_t &s) const
    {
        auto &sh = label_shards_[label_shard_index(label)];
        std::shared_lock<std::shared_mutex> lk(sh.mu);
        auto it = sh.slot.find(label);
        if (it == sh.slot.end())
            return false;
        s = it->second;
        return true;
    }

    bool PomaiOrbit::insert(const float *vec, uint64_t label)
    {
        std::vector<float> v(vec, vec + cfg_.dim);
        std::vector<std::pair<uint64_t, std::vector<float>>> b;
        b.push_back({label, v});
        return insert_batch(b);
    }

    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (batch.empty())
            return false;
        std::shared_lock<std::shared_mutex> cp_lock(checkpoint_mu_);
        if (wal_)
        {
            std::vector<uint8_t> buffer;
            buffer.reserve(4 + batch.size() * (8 + cfg_.dim * 4));
            uint32_t cnt = static_cast<uint32_t>(batch.size());
            uint8_t b4[4];
            std::memcpy(b4, &cnt, 4);
            buffer.insert(buffer.end(), b4, b4 + 4);
            for (const auto &p : batch)
            {
                uint8_t id[8];
                std::memcpy(id, &p.first, 8);
                buffer.insert(buffer.end(), id, id + 8);
                const uint8_t *vptr = reinterpret_cast<const uint8_t *>(p.second.data());
                buffer.insert(buffer.end(), vptr, vptr + cfg_.dim * 4);
            }
            wal_->append_record(20, buffer.data(), buffer.size());
        }
        return insert_batch_memory_only(batch);
    }

    bool PomaiOrbit::insert_batch_memory_only(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (!eeq_)
            return false;

        if (centroids_.empty())
        {
            std::lock_guard<std::mutex> lk(train_mu_);
            if (centroids_.empty())
            {
                std::vector<float> training_data;
                for (const auto &item : batch)
                    if (item.second.size() == cfg_.dim)
                        training_data.insert(training_data.end(), item.second.begin(), item.second.end());
                if (!training_data.empty())
                    this->train(training_data.data(), training_data.size() / cfg_.dim);
            }
        }
        if (centroids_.empty())
            return false;

        struct Item
        {
            uint64_t label;
            uint8_t size;
            uint8_t bytes[MAX_ECHO_BYTES];
            uint32_t cid;
        };
        std::vector<Item> prepared;
        prepared.reserve(batch.size());

        for (const auto &p : batch)
        {
            if (p.second.size() != cfg_.dim)
                continue;
            uint32_t cid = find_nearest_centroid(p.second.data());
            pomai::ai::EchoCode code;
            try
            {
                code = eeq_->encode(p.second.data());
            }
            catch (...)
            {
                continue;
            }

            Item it;
            it.label = p.first;
            size_t pos = 0;
            it.bytes[pos++] = code.depth;
            for (auto s : code.scales_q)
                if (pos < MAX_ECHO_BYTES)
                    it.bytes[pos++] = s;
            for (const auto &sb : code.sign_bytes)
            {
                if (pos + sb.size() > MAX_ECHO_BYTES)
                {
                    pos = 0;
                    break;
                }
                std::memcpy(it.bytes + pos, sb.data(), sb.size());
                pos += sb.size();
            }
            if (pos > 0)
            {
                it.size = pos;
                it.cid = cid;
                prepared.push_back(it);
            }
        }
        if (prepared.empty())
            return true;

        std::sort(prepared.begin(), prepared.end(), [](auto &a, auto &b)
                  { return a.cid < b.cid; });

        size_t idx = 0;
        size_t total = prepared.size();
        while (idx < total)
        {
            uint32_t cid = prepared[idx].cid;
            size_t group_end = idx;
            while (group_end < total && prepared[group_end].cid == cid)
                ++group_end;

            OrbitNode &node = *centroids_[cid];
            std::unique_lock<std::shared_mutex> lk(node.mu);
            uint64_t off = node.bucket_offset.load(std::memory_order_acquire);
            if (off == 0)
            {
                off = alloc_new_bucket(cid);
                if (off == 0)
                {
                    std::cerr << "[Orbit] Failed alloc bucket for cid " << cid << "\n";
                    return false;
                }
                node.bucket_offset.store(off, std::memory_order_release);
            }

            size_t write_idx = idx;
            while (write_idx < group_end)
            {
                std::vector<char> tmp;
                auto base = resolve_bucket_base(arena_, off, tmp);
                if (!base)
                {
                    std::cerr << "[Orbit] FATAL: cannot resolve bucket " << off << "\n";
                    return false;
                }
                char *ptr = const_cast<char *>(*base);
                BucketHeader *hdr = reinterpret_cast<BucketHeader *>(ptr);

                uint32_t cur = bucket_atomic_load_count(hdr);
                if (cur >= dynamic_bucket_capacity_)
                {
                    uint64_t nxt = bucket_atomic_load_next(hdr);
                    if (nxt == 0)
                    {
                        nxt = alloc_new_bucket(cid);
                        if (nxt == 0)
                        {
                            std::cerr << "[Orbit] OOM alloc bucket\n";
                            return false;
                        }
                        bucket_atomic_store_next(hdr, nxt);
                    }
                    off = nxt;
                    continue;
                }

                uint32_t rem = dynamic_bucket_capacity_ - cur;
                uint32_t fit = std::min((uint32_t)(group_end - write_idx), rem);
                uint32_t slot = bucket_atomic_fetch_add_count(hdr, fit);

                if (slot + fit > dynamic_bucket_capacity_)
                {
                    bucket_atomic_sub_count(hdr, fit);
                    uint64_t nxt = bucket_atomic_load_next(hdr);
                    if (nxt == 0)
                    {
                        nxt = alloc_new_bucket(cid);
                        if (nxt == 0)
                        {
                            std::cerr << "[Orbit] OOM alloc bucket\n";
                            return false;
                        }
                        bucket_atomic_store_next(hdr, nxt);
                    }
                    off = nxt;
                    continue;
                }

                char *vbase = ptr + hdr->off_vectors;
                uint16_t *lbase = reinterpret_cast<uint16_t *>(ptr + hdr->off_pq_codes);
                uint64_t *ibase = reinterpret_cast<uint64_t *>(ptr + hdr->off_ids);

                for (uint32_t k = 0; k < fit; ++k)
                {
                    const Item &it = prepared[write_idx + k];
                    uint32_t s = slot + k;
                    std::memcpy(vbase + s * MAX_ECHO_BYTES, it.bytes, it.size);
                    __atomic_store_n(&lbase[s], static_cast<uint16_t>(it.size), __ATOMIC_RELEASE);
                    uint64_t packed = IdEntry::pack_label(it.label);
                    pomai::ai::atomic_utils::atomic_store_u64(ibase + s, packed);
                    set_label_map(it.label, off, s);
                }
                write_idx += fit;
            }
            idx = group_end;
        }
        return true;
    }

    bool PomaiOrbit::get(uint64_t label, std::vector<float> &out_vec)
    {
        {
            std::shared_lock<std::shared_mutex> lk(del_mu_);
            if (deleted_labels_.count(label))
                return false;
        }
        uint64_t b = 0;
        uint32_t s = 0;
        if (!get_label_bucket(label, b) || !get_label_slot(label, s))
            return false;

        std::vector<char> tmp;
        auto base = resolve_bucket_base(arena_, b, tmp);
        if (!base)
            return false;
        const char *ptr = *base;
        BucketHeader hdr;
        std::memcpy(&hdr, ptr, sizeof(hdr));

        uint16_t len;
        std::memcpy(&len, ptr + hdr.off_pq_codes + s * 2, 2);
        if (len == 0 || len > MAX_ECHO_BYTES)
            return false;

        const uint8_t *code_ptr = reinterpret_cast<const uint8_t *>(ptr + hdr.off_vectors + s * MAX_ECHO_BYTES);

        pomai::ai::EchoCode code;
        size_t pos = 0;
        code.depth = code_ptr[pos++];
        code.scales_q.resize(code.depth);
        code.bits_per_layer.resize(code.depth);
        code.sign_bytes.resize(code.depth);

        for (size_t k = 0; k < code.depth; ++k)
            code.scales_q[k] = code_ptr[pos++];
        for (size_t k = 0; k < code.depth; ++k)
        {
            uint32_t bits = cfg_.eeq_cfg.bits_per_layer[k];
            code.bits_per_layer[k] = bits;
            size_t nb = (bits + 7) / 8;
            code.sign_bytes[k].assign(code_ptr + pos, code_ptr + pos + nb);
            pos += nb;
        }
        out_vec.resize(cfg_.dim);
        eeq_->decode(code, out_vec.data());
        return true;
    }

    bool PomaiOrbit::remove(uint64_t label)
    {
        std::unique_lock<std::shared_mutex> lk(del_mu_);
        deleted_labels_.insert(label);
        return true;
    }

    // --- Search Implementations ---
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        pomai::ai::Budget budget;
        return search_with_budget(query, k, budget, nprobe);
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered(const float *query, size_t k, const std::vector<uint64_t> &candidates)
    {
        pomai::ai::Budget budget;
        return search_filtered_with_budget(query, k, candidates, budget);
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_with_budget(const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe)
    {
        if (!eeq_ || centroids_.empty())
            return {};

        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = pomai_dot(query, query, cfg_.dim);

        size_t np = (nprobe > 0) ? nprobe : cfg_.algo.m_neighbors;
        auto targets = find_routing_centroids(query, np);

        std::priority_queue<std::pair<float, uint64_t>> topk;

        for (uint32_t cid : targets)
        {
            uint64_t curr = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (curr != 0)
            {
                std::vector<char> tmp;
                auto base = resolve_bucket_base(arena_, curr, tmp);
                if (!base)
                    break;
                const char *ptr = *base;
                BucketHeader hdr;
                std::memcpy(&hdr, ptr, sizeof(hdr));
                uint32_t n = hdr.count;

                const uint64_t *id_ptr = reinterpret_cast<const uint64_t *>(ptr + hdr.off_ids);
                const uint16_t *len_ptr = reinterpret_cast<const uint16_t *>(ptr + hdr.off_pq_codes);
                const char *vec_base = ptr + hdr.off_vectors;

                for (uint32_t i = 0; i < n; ++i)
                {
                    uint64_t packed = pomai::ai::atomic_utils::atomic_load_u64(id_ptr + i);
                    if (IdEntry::tag_of(packed) != IdEntry::TAG_LABEL)
                        continue;
                    uint64_t lbl = IdEntry::payload_of(packed);

                    {
                        std::shared_lock<std::shared_mutex> dl(del_mu_);
                        if (deleted_labels_.count(lbl))
                            continue;
                    }

                    const uint8_t *code = reinterpret_cast<const uint8_t *>(vec_base + i * MAX_ECHO_BYTES);
                    float d = eeq_->approx_dist_code_bytes(qproj, qnorm2, code, len_ptr[i]);

                    if (topk.size() < k)
                        topk.push({d, lbl});
                    else if (d < topk.top().first)
                    {
                        topk.pop();
                        topk.push({d, lbl});
                    }
                }
                curr = hdr.next_bucket_offset;
            }
        }

        std::vector<std::pair<uint64_t, float>> res;
        while (!topk.empty())
        {
            res.push_back({topk.top().second, topk.top().first});
            topk.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered_with_budget(const float *query, size_t k, const std::vector<uint64_t> &candidates, const pomai::ai::Budget &)
    {
        if (candidates.empty())
            return {};
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = pomai_dot(query, query, cfg_.dim);

        std::priority_queue<std::pair<float, uint64_t>> topk;

        for (uint64_t id : candidates)
        {
            float dist = 0;
            if (const_cast<PomaiOrbit *>(this)->compute_distance_for_id_with_proj(qproj, qnorm2, id, dist))
            {
                if (topk.size() < k)
                    topk.push({dist, id});
                else if (dist < topk.top().first)
                {
                    topk.pop();
                    topk.push({dist, id});
                }
            }
        }
        std::vector<std::pair<uint64_t, float>> res;
        while (!topk.empty())
        {
            res.push_back({topk.top().second, topk.top().first});
            topk.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    bool PomaiOrbit::compute_distance_for_id_with_proj(const std::vector<std::vector<float>> &qproj, float qnorm2, uint64_t id, float &out_dist)
    {
        {
            std::shared_lock<std::shared_mutex> dl(del_mu_);
            if (deleted_labels_.count(id))
                return false;
        }
        uint64_t b = 0;
        uint32_t s = 0;
        if (!get_label_bucket(id, b) || !get_label_slot(id, s))
            return false;

        std::vector<char> tmp;
        auto base = resolve_bucket_base(arena_, b, tmp);
        if (!base)
            return false;
        const char *ptr = *base;
        BucketHeader hdr;
        std::memcpy(&hdr, ptr, sizeof(hdr));

        uint16_t len = 0;
        std::memcpy(&len, ptr + hdr.off_pq_codes + s * 2, 2);
        const uint8_t *code = reinterpret_cast<const uint8_t *>(ptr + hdr.off_vectors + s * MAX_ECHO_BYTES);

        out_dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, code, len);
        return true;
    }

    MembranceInfo PomaiOrbit::get_info() const
    {
        MembranceInfo info;
        info.dim = cfg_.dim;
        info.num_vectors = 0;
        for (const auto &sh : label_shards_)
        {
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            info.num_vectors += sh.bucket.size();
        }
        info.disk_bytes = 0;
        if (std::filesystem::exists(cfg_.data_path))
        {
            for (const auto &entry : std::filesystem::recursive_directory_iterator(cfg_.data_path))
            {
                if (entry.is_regular_file())
                    info.disk_bytes += entry.file_size();
            }
        }
        return info;
    }

    std::vector<uint64_t> PomaiOrbit::get_all_labels() const
    {
        std::vector<uint64_t> res;
        for (const auto &sh : label_shards_)
        {
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            for (auto &kv : sh.bucket)
                res.push_back(kv.first);
        }
        return res;
    }

    std::vector<uint64_t> PomaiOrbit::get_centroid_ids(uint32_t cid) const
    {
        return {};
    }
    bool PomaiOrbit::get_vectors_raw(const std::vector<uint64_t> &, std::vector<std::string> &) const
    {
        return false;
    }

    bool PomaiOrbit::checkpoint()
    {
        try
        {
            std::unique_lock<std::shared_mutex> cp_lock(checkpoint_mu_);
            std::lock_guard<std::mutex> lk(train_mu_);

            save_schema();

            // Fsync memory mapped regions
            long pg = sysconf(_SC_PAGESIZE);
            size_t page = (pg > 0) ? static_cast<size_t>(pg) : 4096;
            for (const auto &c : centroids_)
            {
                uint64_t off = c->bucket_offset.load(std::memory_order_acquire);
                while (off != 0)
                {
                    const char *ptr = arena_.blob_ptr_from_offset_for_map(off);
                    if (!ptr)
                        break;
                    uint32_t len;
                    std::memcpy(&len, ptr, 4);
                    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
                    uintptr_t base = addr & ~(page - 1);
                    uintptr_t end = (addr + 4 + len + page - 1) & ~(page - 1);
                    ::msync(reinterpret_cast<void *>(base), end - base, MS_SYNC);

                    BucketHeader hdr;
                    std::memcpy(&hdr, ptr + 4, sizeof(hdr));
                    off = hdr.next_bucket_offset;
                }
            }

            // [CRITICAL FIX] Do not truncate WAL.
            // Since the arena may be ephemeral (temp file), we rely on the WAL
            // for guaranteed recovery upon restart.
            if (wal_)
                wal_->fsync_log();
            ::sync();
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

} // namespace pomai::ai::orbit