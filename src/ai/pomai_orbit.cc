#include "src/ai/pomai_orbit.h"
#include "src/ai/atomic_utils.h"
#include "src/ai/ids_block.h"
#include "src/memory/arena.h"
#include "src/memory/shard_arena.h"
#include "src/ai/zeroharmony_pack.h"
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
#include <cstdlib>
#include "src/core/algo/blitz_kernels.h"

namespace pomai::ai::orbit
{
    using namespace pomai::ai::soa;

    static constexpr size_t MAX_ECHO_BYTES = 64 * 1024;
    static constexpr size_t DEFAULT_MAX_SUBBATCH = 4096;

    struct Item
    {
        uint64_t label;
        uint8_t size;
        uint8_t bytes[MAX_ECHO_BYTES];
        uint32_t cid;
    };

    static std::optional<const char *> resolve_bucket_base(const ArenaView &arena, uint64_t bucket_off, std::vector<char> &temp_buffer)
    {
        if (bucket_off == 0)
            return std::nullopt;
        const char *ram_blob_ptr = arena.blob_ptr_from_offset_for_map(bucket_off);
        if (ram_blob_ptr)
            return ram_blob_ptr + sizeof(uint32_t);
        temp_buffer = arena.read_remote_blob(bucket_off);
        if (temp_buffer.empty())
            return std::nullopt;
        return temp_buffer.data() + sizeof(uint32_t);
    }

    static inline uint32_t load_blob_len_from_body_ptr(const char *body_ptr) noexcept
    {
        const uint32_t *lenptr = reinterpret_cast<const uint32_t *>(body_ptr - sizeof(uint32_t));
        uint32_t len = 0;
        std::memcpy(&len, lenptr, sizeof(len));
        return len;
    }

    static inline bool safe_blob_range(uint32_t blob_len, uint32_t off, size_t need) noexcept
    {
        if (off > blob_len)
            return false;
        uint64_t end = static_cast<uint64_t>(off) + static_cast<uint64_t>(need);
        if (end > static_cast<uint64_t>(blob_len))
            return false;
        return true;
    }

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

    static inline size_t compute_packed_slot_size(size_t dim, bool use_half_nonzero)
    {
        const size_t min_slot = 512;
        const size_t max_slot = 64 * 1024;
        size_t multiplier = use_half_nonzero ? 3 : 5;
        size_t pad = 16;
        size_t calc = dim * multiplier + pad;
        size_t slot = (calc < min_slot) ? min_slot : (calc > max_slot ? max_slot : calc);
        if (const char *env = std::getenv("POMAI_PACKED_SLOT_BYTES"))
        {
            try
            {
                size_t ev = static_cast<size_t>(std::stoul(env));
                if (ev < min_slot)
                    ev = min_slot;
                if (ev > max_slot)
                    ev = max_slot;
                slot = ev;
            }
            catch (...)
            {
            }
        }
        return slot;
    }

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
        zeroharmony_ = std::make_unique<pomai::ai::ZeroHarmonyPacker>(cfg_.zero_harmony_cfg, cfg_.dim);

        packed_slot_size_ = compute_packed_slot_size(cfg_.dim, cfg_.zero_harmony_cfg.use_half_nonzero);
        std::clog << "[Orbit] packed_slot_size_=" << packed_slot_size_ << " (dim=" << cfg_.dim
                  << " half_nonzero=" << (cfg_.zero_harmony_cfg.use_half_nonzero ? "1" : "0") << ")\n";

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
        if (!wal_->open(wal_path, wcfg))
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
        zeroharmony_ = std::make_unique<pomai::ai::ZeroHarmonyPacker>(cfg_.zero_harmony_cfg, cfg_.dim);

        packed_slot_size_ = compute_packed_slot_size(cfg_.dim, cfg_.zero_harmony_cfg.use_half_nonzero);
        std::clog << "[Orbit] packed_slot_size_=" << packed_slot_size_ << " (dim=" << cfg_.dim
                  << " half_nonzero=" << (cfg_.zero_harmony_cfg.use_half_nonzero ? "1" : "0") << ")\n";

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
        if (!wal_->open(wal_path, wcfg))
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
        auto replayer = [&](uint64_t seq, uint16_t type, const std::vector<uint8_t> &data)
        {
            if (type == static_cast<uint16_t>(pomai::memory::WAL_REC_INSERT_BATCH) && data.size() >= 4)
            {
                const uint8_t *ptr = data.data();
                uint32_t count = 0;
                std::memcpy(&count, ptr, 4);
                ptr += 4;
                size_t vec_bytes = cfg_.dim * sizeof(float);
                if (data.size() < 4 + static_cast<size_t>(count) * (8 + vec_bytes))
                    return;
                std::vector<std::pair<uint64_t, std::vector<float>>> batch;
                batch.reserve(count);
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
                if (!batch.empty() && insert_batch_memory_only(batch))
                    replayed_count += batch.size();
            }
            else if (type == static_cast<uint16_t>(pomai::memory::WAL_REC_DELETE_LABEL) && data.size() >= sizeof(uint64_t))
            {
                uint64_t label = 0;
                std::memcpy(&label, data.data(), sizeof(label));
                try
                {
                    apply_persisted_delete(label);
                }
                catch (...)
                {
                }
            }
        };
        wal_->recover(replayer);
        if (replayed_count > 0)
            std::clog << "[Orbit] Recovered " << replayed_count << " vectors from WAL.\n";
    }

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

    void PomaiOrbit::apply_persisted_delete(uint64_t label)
    {
        {
            std::unique_lock<std::shared_mutex> dl(del_mu_);
            deleted_labels_.insert(label);
        }

        uint64_t b = 0;
        uint32_t s = 0;
        if (!get_label_bucket(label, b) || !get_label_slot(label, s))
            return;

        std::vector<char> tmp;
        auto base_opt = resolve_bucket_base(arena_, b, tmp);
        if (!base_opt)
        {
            auto &sh = label_shards_[label_shard_index(label)];
            std::unique_lock<std::shared_mutex> lk(sh.mu);
            sh.bucket.erase(label);
            sh.slot.erase(label);
            return;
        }

        char *mutable_base = const_cast<char *>(*base_opt);
        BucketHeader *hdr = reinterpret_cast<BucketHeader *>(mutable_base);

        if (s >= dynamic_bucket_capacity_)
        {
            auto &sh = label_shards_[label_shard_index(label)];
            std::unique_lock<std::shared_mutex> lk(sh.mu);
            sh.bucket.erase(label);
            sh.slot.erase(label);
            return;
        }

        uint64_t *ids_base = reinterpret_cast<uint64_t *>(mutable_base + hdr->off_ids);
        uint16_t *lens_base = reinterpret_cast<uint16_t *>(mutable_base + hdr->off_pq_codes);

        pomai::ai::atomic_utils::atomic_store_u64(ids_base + s, 0);
        __atomic_store_n(lens_base + s, static_cast<uint16_t>(0), __ATOMIC_RELEASE);

        pomai::ai::atomic_utils::atomic_fetch_sub_u32(&hdr->count, 1u);

        uint32_t cid = hdr->centroid_id;
        if (cid < bucket_sizes_.size())
            pomai::ai::atomic_utils::atomic_fetch_sub_u32(&bucket_sizes_[cid], 1u);

        auto &sh = label_shards_[label_shard_index(label)];
        {
            std::unique_lock<std::shared_mutex> lk(sh.mu);
            sh.bucket.erase(label);
            sh.slot.erase(label);
        }
    }

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
        if (!out.is_open())
            return;
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

    bool PomaiOrbit::load_schema()
    {
        std::ifstream in(schema_file_path_, std::ios::binary);
        if (!in.is_open())
            return false;
        SchemaHeader header;
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (!in || header.magic_number != 0x504F4D41)
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
            node->bucket_offset.store(off_disk, std::memory_order_relaxed);
            centroids_[i] = std::move(node);
        }
        rebuild_index();
        return true;
    }

    void PomaiOrbit::rebuild_index()
    {
        bucket_sizes_.assign(centroids_.size(), 0);

        for (const auto &c : centroids_)
        {
            uint64_t curr = c->bucket_offset.load(std::memory_order_relaxed);
            while (curr != 0)
            {
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, curr, tmp);
                if (!base_opt)
                    break;

                const BucketHeader *hdr = reinterpret_cast<const BucketHeader *>(*base_opt);
                uint32_t n = bucket_atomic_load_count(hdr);

                pomai::ai::atomic_utils::atomic_fetch_add_u32(&bucket_sizes_[hdr->centroid_id], n);

                const char *ids_ptr = (*base_opt) + hdr->off_ids;
                for (uint32_t i = 0; i < n; ++i)
                {
                    uint64_t packed;
                    std::memcpy(&packed, ids_ptr + i * 8, 8);
                    if (IdEntry::tag_of(packed) == IdEntry::TAG_LABEL)
                    {
                        set_label_map(IdEntry::payload_of(packed), curr, i);
                    }
                }
                curr = bucket_atomic_load_next(hdr);
            }
        }

        if (!centroids_.empty())
        {
            L2Func kern = get_pomai_l2sq_kernel();
            for (size_t i = 0; i < centroids_.size(); ++i)
            {
                std::vector<std::pair<float, uint32_t>> dists;
                dists.reserve(centroids_.size() - 1);
                for (size_t j = 0; j < centroids_.size(); ++j)
                    if (i != j)
                        dists.push_back({kern(centroids_[i]->vector.data(), centroids_[j]->vector.data(), cfg_.dim), static_cast<uint32_t>(j)});
                std::sort(dists.begin(), dists.end());
                std::lock_guard<std::shared_mutex> lk(centroids_[i]->mu);
                centroids_[i]->neighbors.clear();
                for (size_t k = 0; k < std::min(dists.size(), (size_t)32); ++k)
                    centroids_[i]->neighbors.push_back(dists[k].second);
            }
        }
    }

    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0)
            return false;

        size_t num_c = std::clamp(n / cfg_.algo.auto_scale_factor,
                                  (size_t)cfg_.algo.min_centroids,
                                  (size_t)cfg_.algo.max_centroids);

        std::clog << "[Autopilot] Initializing " << num_c << " centroids for " << n << " samples.\n";

        centroids_.clear();
        bucket_sizes_.assign(num_c, 0);

        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(1337));

        for (size_t i = 0; i < num_c; ++i)
        {
            size_t idx = indices[i % n];
            auto node = std::make_unique<OrbitNode>();
            node->vector.assign(data + idx * cfg_.dim, data + (idx + 1) * cfg_.dim);
            node->bucket_offset.store(alloc_new_bucket(static_cast<uint32_t>(i)), std::memory_order_release);
            centroids_.push_back(std::move(node));
        }

        rebuild_index();
        save_schema();
        return true;
    }

    uint64_t PomaiOrbit::alloc_new_bucket(uint32_t cid)
    {
        size_t cap = dynamic_bucket_capacity_;
        auto align = [](size_t s)
        { return (s + 63) & ~63; };
        size_t head_sz = sizeof(BucketHeader);
        size_t code_sz = sizeof(uint16_t) * cap;
        size_t vec_sz = packed_slot_size_ * cap;
        size_t ids_sz = sizeof(uint64_t) * cap;
        uint32_t off_fp = static_cast<uint32_t>(align(head_sz));
        uint32_t off_pq = static_cast<uint32_t>(align(off_fp));
        uint32_t off_vec = static_cast<uint32_t>(align(off_pq + code_sz));
        uint32_t off_ids = static_cast<uint32_t>(align(off_vec + vec_sz));
        size_t total = off_ids + ids_sz;
        if (total > std::numeric_limits<uint32_t>::max())
        {
            std::cerr << "[Orbit] alloc_new_bucket: requested bucket size too large\n";
            return 0;
        }
        char *blob = arena_.alloc_blob(static_cast<uint32_t>(total));
        if (!blob)
            return 0;
        BucketHeader hdr;
        std::memset(&hdr, 0, sizeof(hdr));
        hdr.centroid_id = cid;
        hdr.off_pq_codes = off_pq;
        hdr.off_vectors = off_vec;
        hdr.off_ids = off_ids;
        std::memcpy(blob + 4, &hdr, sizeof(hdr));
        return arena_.offset_from_blob_ptr(blob);
    }

    uint32_t PomaiOrbit::find_nearest_centroid(const float *vec)
    {
        uint32_t best = 0;
        float min_d = std::numeric_limits<float>::infinity();
        L2Func kern = get_pomai_l2sq_kernel();
        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            float d = kern(vec, centroids_[i]->vector.data(), cfg_.dim);
            if (d < min_d)
            {
                min_d = d;
                best = static_cast<uint32_t>(i);
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
        all.reserve(centroids_.size());
        L2Func kern = get_pomai_dot_kernel();
        for (size_t i = 0; i < centroids_.size(); ++i)
            all.push_back({kern(vec, centroids_[i]->vector.data(), cfg_.dim), static_cast<uint32_t>(i)});
        std::sort(all.begin(), all.end());
        std::vector<uint32_t> res;
        res.reserve(std::min(n, all.size()));
        for (size_t i = 0; i < std::min(n, all.size()); ++i)
            res.push_back(all[i].second);
        return res;
    }

    void PomaiOrbit::set_label_map(uint64_t label, uint64_t bucket_off, uint32_t slot)
    {
        auto &sh = label_shards_[label_shard_index(label)];
        std::unique_lock<std::shared_mutex> lk(sh.mu);
        sh.bucket[label] = bucket_off;
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
        if (!vec)
            return false;
        std::vector<float> v(cfg_.dim);
        std::memcpy(v.data(), vec, cfg_.dim * sizeof(float));
        std::vector<std::pair<uint64_t, std::vector<float>>> b;
        b.emplace_back(label, std::move(v));
        return insert_batch(b);
    }

    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (batch.empty())
            return false;

        if (wal_)
        {
            std::vector<uint8_t> buffer;
            buffer.reserve(4 + batch.size() * (8 + cfg_.dim * 4));
            uint32_t cnt = 0;
            for (const auto &p : batch)
                if (p.second.size() == cfg_.dim)
                    ++cnt;
            uint8_t b4[4];
            std::memcpy(b4, &cnt, 4);
            buffer.insert(buffer.end(), b4, b4 + 4);

            for (const auto &p : batch)
            {
                if (p.second.size() != cfg_.dim)
                    continue;
                uint8_t idbuf[8];
                std::memcpy(idbuf, &p.first, 8);
                buffer.insert(buffer.end(), idbuf, idbuf + 8);
                const uint8_t *vptr = reinterpret_cast<const uint8_t *>(p.second.data());
                buffer.insert(buffer.end(), vptr, vptr + cfg_.dim * sizeof(float));
            }

            try
            {
                uint64_t seq = wal_->append(static_cast<uint16_t>(pomai::memory::WAL_REC_INSERT_BATCH), buffer.data(), static_cast<size_t>(buffer.size()), 0);
                if (seq == 0)
                    std::cerr << "[Orbit] WAL append failed for insert batch\n";
            }
            catch (...)
            {
            }
        }

        return insert_batch_memory_only(batch);
    }

    bool PomaiOrbit::insert_batch_memory_only(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (batch.empty())
            return false;

        std::vector<uint32_t> split_candidates;
        {
            std::shared_lock<std::shared_mutex> cp_lock(checkpoint_mu_);

            if (centroids_.empty())
            {
                size_t seed_count = std::max<size_t>(cfg_.algo.min_centroids, std::thread::hardware_concurrency() / 2);
                for (size_t i = 0; i < std::min(batch.size(), seed_count); ++i)
                {
                    auto node = std::make_unique<OrbitNode>();
                    node->vector = batch[i].second;
                    node->bucket_offset.store(alloc_new_bucket(static_cast<uint32_t>(i)), std::memory_order_release);
                    centroids_.push_back(std::move(node));
                }
                rebuild_index();
                build_echo_graph(1.0f, 0.01f);
            }

            const size_t B = batch.size();
            const size_t C = centroids_.size();
            const size_t D = cfg_.dim;

            std::vector<float> qnorm2;
            qnorm2.resize(B);
            std::vector<float> trans;
            trans.assign(D * B, 0.0f);

            for (size_t v = 0; v < B; ++v)
            {
                const float *q = batch[v].second.data();
                float s = 0.0f;
                for (size_t d = 0; d < D; ++d)
                {
                    float val = q[d];
                    s += val * val;
                    trans[d * B + v] = val;
                }
                qnorm2[v] = s;
            }

            std::vector<float> cflat;
            cflat.resize(C * D);
            std::vector<float> cnorm2;
            cnorm2.resize(C);
            for (size_t c = 0; c < C; ++c)
            {
                const std::vector<float> &cv = centroids_[c]->vector;
                float s = 0.0f;
                for (size_t d = 0; d < D; ++d)
                {
                    float v = cv[d];
                    cflat[c * D + d] = v;
                    s += v * v;
                }
                cnorm2[c] = s;
            }

            std::vector<float> best_dist(B, std::numeric_limits<float>::infinity());
            std::vector<uint32_t> assigned_cid(B, 0);
            std::vector<float> dot;
            dot.assign(B, 0.0f);

            for (size_t c = 0; c < C; ++c)
            {
                std::fill(dot.begin(), dot.end(), 0.0f);

                const float *cptr = &cflat[c * D];

                for (size_t d = 0; d < D; ++d)
                {
                    float cd = cptr[d];
                    const float *trow = &trans[d * B];
                    size_t v = 0;
                    const size_t B8 = (B / 8) * 8;
                    for (; v < B8; v += 8)
                    {
                        dot[v + 0] += trow[v + 0] * cd;
                        dot[v + 1] += trow[v + 1] * cd;
                        dot[v + 2] += trow[v + 2] * cd;
                        dot[v + 3] += trow[v + 3] * cd;
                        dot[v + 4] += trow[v + 4] * cd;
                        dot[v + 5] += trow[v + 5] * cd;
                        dot[v + 6] += trow[v + 6] * cd;
                        dot[v + 7] += trow[v + 7] * cd;
                    }
                    for (; v < B; ++v)
                        dot[v] += trow[v] * cd;
                }

                float c2 = cnorm2[c];
                for (size_t v = 0; v < B; ++v)
                {
                    float dist = qnorm2[v] + c2 - 2.0f * dot[v];
                    if (dist < best_dist[v])
                    {
                        best_dist[v] = dist;
                        assigned_cid[v] = static_cast<uint32_t>(c);
                    }
                }
            }

            std::vector<Item> prepared;
            prepared.reserve(B);
            for (size_t i = 0; i < B; ++i)
            {
                Item it;
                it.label = batch[i].first;
                uint32_t cid = assigned_cid[i];
                it.cid = cid;
                std::vector<uint8_t> pk = zeroharmony_->pack_with_mean(batch[i].second.data(), centroids_[cid]->vector);
                if (pk.size() > MAX_ECHO_BYTES)
                    pk.resize(MAX_ECHO_BYTES);
                it.size = static_cast<uint8_t>(pk.size());
                std::memcpy(it.bytes, pk.data(), pk.size());
                prepared.push_back(it);
            }

            std::sort(prepared.begin(), prepared.end(), [](auto &a, auto &b)
                      { return a.cid < b.cid; });

            std::vector<uint32_t> added_counts;
            added_counts.assign(centroids_.size(), 0);

            for (size_t i = 0; i < prepared.size();)
            {
                uint32_t cid = prepared[i].cid;
                size_t end = i;
                while (end < prepared.size() && prepared[end].cid == cid)
                    ++end;

                OrbitNode &node = *centroids_[cid];
                size_t k = i;
                std::vector<std::tuple<uint64_t, uint32_t, uint64_t>> deferred_maps;
                while (k < end)
                {
                    uint64_t off = node.bucket_offset.load(std::memory_order_acquire);
                    std::vector<char> tmp;
                    auto base_opt = resolve_bucket_base(arena_, off, tmp);
                    if (!base_opt)
                    {
                        uint64_t cur = node.bucket_offset.load(std::memory_order_acquire);
                        if (cur == off)
                        {
                            uint64_t alloc = alloc_new_bucket(cid);
                            if (alloc == 0)
                                break;
                            uint64_t expected = off;
                            node.bucket_offset.compare_exchange_weak(expected, alloc, std::memory_order_acq_rel);
                            off = node.bucket_offset.load(std::memory_order_acquire);
                            base_opt = resolve_bucket_base(arena_, off, tmp);
                            if (!base_opt)
                                continue;
                        }
                        else
                        {
                            off = cur;
                            continue;
                        }
                    }
                    char *mutable_base = const_cast<char *>(*base_opt);
                    BucketHeader *hdr = reinterpret_cast<BucketHeader *>(mutable_base);
                    uint32_t cur_count = bucket_atomic_load_count(hdr);
                    if (cur_count >= dynamic_bucket_capacity_)
                    {
                        uint64_t nxt = bucket_atomic_load_next(hdr);
                        if (!nxt)
                        {
                            uint64_t alloc = alloc_new_bucket(cid);
                            if (alloc == 0)
                                break;
                            uint64_t expected = 0;
                            pomai::ai::atomic_utils::atomic_compare_exchange_u64(&hdr->next_bucket_offset, expected, alloc);
                            nxt = bucket_atomic_load_next(hdr);
                            if (!nxt)
                                break;
                        }
                        off = nxt;
                        continue;
                    }
                    uint32_t avail = dynamic_bucket_capacity_ - cur_count;
                    size_t want = std::min<size_t>(avail, end - k);
                    uint32_t old = bucket_atomic_fetch_add_count(hdr, static_cast<uint32_t>(want));
                    if (old >= dynamic_bucket_capacity_)
                    {
                        bucket_atomic_sub_count(hdr, static_cast<uint32_t>(want));
                        continue;
                    }
                    uint32_t start_slot = old;
                    size_t actual = std::min<size_t>(want, dynamic_bucket_capacity_ - old);
                    uint8_t *vec_base = reinterpret_cast<uint8_t *>(mutable_base + hdr->off_vectors);
                    uint16_t *lens_base = reinterpret_cast<uint16_t *>(mutable_base + hdr->off_pq_codes);
                    uint64_t *ids_base = reinterpret_cast<uint64_t *>(mutable_base + hdr->off_ids);
                    for (size_t t = 0; t < actual; ++t)
                    {
                        size_t idx = k + t;
                        std::memcpy(vec_base + static_cast<size_t>(start_slot + t) * packed_slot_size_, prepared[idx].bytes, prepared[idx].size);
                        __atomic_store_n(&lens_base[start_slot + t], static_cast<uint16_t>(prepared[idx].size), __ATOMIC_RELEASE);
                        uint64_t packed_id = IdEntry::pack_label(prepared[idx].label);
                        pomai::ai::atomic_utils::atomic_store_u64(ids_base + start_slot + t, packed_id);
                        deferred_maps.emplace_back(prepared[idx].label, off, static_cast<uint64_t>(start_slot + t));
                    }
                    added_counts[cid] = static_cast<uint32_t>(added_counts[cid] + actual);
                    k += actual;
                }
                for (auto &m : deferred_maps)
                {
                    uint64_t lbl = std::get<0>(m);
                    uint64_t boff = std::get<1>(m);
                    uint32_t slot = static_cast<uint32_t>(std::get<2>(m));
                    set_label_map(lbl, boff, slot);
                }
                i = end;
            }

            for (size_t cid = 0; cid < added_counts.size(); ++cid)
            {
                uint32_t added = added_counts[cid];
                if (added == 0)
                    continue;
                uint32_t prev = pomai::ai::atomic_utils::atomic_fetch_add_u32(&bucket_sizes_[cid], added);
                if (prev + added > K_SPLIT_THRESHOLD)
                    split_candidates.push_back(static_cast<uint32_t>(cid));
            }
        }

        for (uint32_t cid : split_candidates)
            check_and_split_bucket(cid);

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
        {
            bool found = false;
            for (size_t cid = 0; cid < centroids_.size() && !found; ++cid)
            {
                uint64_t curr = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
                while (curr != 0 && !found)
                {
                    std::vector<char> tmp;
                    auto base_opt = resolve_bucket_base(arena_, curr, tmp);
                    if (!base_opt)
                        break;
                    const char *ptr = *base_opt;
                    const BucketHeader *hdrptr = reinterpret_cast<const BucketHeader *>(ptr);
                    uint32_t n = bucket_atomic_load_count(hdrptr);
                    if (n > dynamic_bucket_capacity_)
                        break;
                    if (!safe_blob_range(load_blob_len_from_body_ptr(ptr), hdrptr->off_ids, static_cast<size_t>(n) * sizeof(uint64_t)))
                        break;
                    const uint64_t *id_ptr = reinterpret_cast<const uint64_t *>(ptr + hdrptr->off_ids);
                    for (uint32_t i = 0; i < n; ++i)
                    {
                        uint64_t packed = pomai::ai::atomic_utils::atomic_load_u64(id_ptr + i);
                        if (IdEntry::tag_of(packed) == IdEntry::TAG_LABEL && IdEntry::payload_of(packed) == label)
                        {
                            set_label_map(label, curr, i);
                            b = curr;
                            s = i;
                            found = true;
                            break;
                        }
                    }
                    curr = bucket_atomic_load_next(hdrptr);
                }
            }
            if (!found)
                return false;
        }
        std::vector<char> tmp;
        auto base = resolve_bucket_base(arena_, b, tmp);
        if (!base)
            return false;
        const char *ptr = *base;
        BucketHeader hdr;
        std::memcpy(&hdr, ptr, sizeof(hdr));
        uint32_t blob_len = load_blob_len_from_body_ptr(ptr);
        if (!safe_blob_range(blob_len, hdr.off_pq_codes, sizeof(uint16_t)))
            return false;
        const uint16_t *lbase = reinterpret_cast<const uint16_t *>(ptr + hdr.off_pq_codes);
        uint16_t len = __atomic_load_n(lbase + s, __ATOMIC_ACQUIRE);
        if (len == 0 || len > static_cast<uint16_t>(packed_slot_size_))
            return false;
        if (!safe_blob_range(blob_len, hdr.off_vectors + s * packed_slot_size_, len))
            return false;
        const uint8_t *code_ptr = reinterpret_cast<const uint8_t *>(ptr + hdr.off_vectors + s * packed_slot_size_);
        uint32_t cid = hdr.centroid_id;
        if (cid >= centroids_.size())
            return false;
        const std::vector<float> &mean = centroids_[cid]->vector;
        out_vec.resize(cfg_.dim);
        if (!zeroharmony_->unpack_to(code_ptr, len, mean, out_vec.data()))
            return false;
        return true;
    }

    bool PomaiOrbit::remove(uint64_t label)
    {
        {
            std::unique_lock<std::shared_mutex> lk(del_mu_);
            if (deleted_labels_.count(label))
                return false;
        }

        bool wal_ok = true;
        if (wal_)
        {
            uint64_t seq = wal_->append(static_cast<uint16_t>(pomai::memory::WAL_REC_DELETE_LABEL), &label, static_cast<size_t>(sizeof(label)), 0);
            if (seq == 0)
            {
                std::cerr << "[Orbit] WAL append (delete) failed for label " << label << "\n";
                wal_ok = false;
            }
        }

        try
        {
            apply_persisted_delete(label);
        }
        catch (...)
        {
        }
        return wal_ok;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        pomai::ai::Budget budget;
        if (whisper_ctrl_)
            budget = whisper_ctrl_->compute_budget(false);

        auto start = std::chrono::high_resolution_clock::now();
        auto results = search_with_budget(query, k, budget, nprobe);

        if (whisper_ctrl_)
        {
            float ms = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
            whisper_ctrl_->observe_latency(ms);
        }

        PomaiMetrics::total_searches.fetch_add(1, std::memory_order_relaxed);
        if (results.empty())
            PomaiMetrics::searches_empty.fetch_add(1, std::memory_order_relaxed);

        return results;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_with_budget(
        const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe)
    {
        if (centroids_.empty())
            return {};

        const size_t max_scan = (budget.ops_budget > 0) ? budget.ops_budget : 50000;
        uint32_t hops = (budget.bucket_budget > 0) ? budget.bucket_budget : 16;
        if (nprobe > 0)
            hops = nprobe;

        uint32_t start_cid = find_nearest_centroid(query);
        size_t scanned = 0;
        auto scan_cb = [this, query, &scanned, max_scan, k](uint32_t cid, auto &heap)
        {
            if (scanned >= max_scan)
                return;
            this->scan_bucket_blitz_avx2(query, cid, heap, scanned, max_scan, k);
        };

        return echo_graph_.auto_navigate(query, start_cid, k, hops, scan_cb);
    }

    void PomaiOrbit::scan_bucket_blitz_avx2(const float *query, uint32_t cid,
                                            std::priority_queue<std::pair<float, uint64_t>> &heap,
                                            size_t &scanned, size_t limit, size_t keep_k) const
    {
        uint64_t curr = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
        static thread_local std::vector<char> buf;
        buf.reserve(64 * 1024);

        while (curr != 0 && scanned < limit)
        {
            auto base = resolve_bucket_base(arena_, curr, buf);
            if (!base)
                break;
            const BucketHeader *hdr = reinterpret_cast<const BucketHeader *>(*base);
            uint32_t n = bucket_atomic_load_count(hdr);

            const uint64_t *ids = reinterpret_cast<const uint64_t *>(*base + hdr->off_ids);
            const uint16_t *lens = reinterpret_cast<const uint16_t *>(*base + hdr->off_pq_codes);
            const char *vecs = *base + hdr->off_vectors;

            for (uint32_t i = 0; i < n; ++i)
            {
                if (++scanned > limit)
                    return;

                float worst = heap.size() >= keep_k ? heap.top().first : std::numeric_limits<float>::infinity();
                const uint8_t *code = reinterpret_cast<const uint8_t *>(vecs + (size_t)i * packed_slot_size_);

                float d = zeroharmony_->approx_dist_with_cutoff(query, code, lens[i], centroids_[cid]->vector, worst);
                if (heap.size() < keep_k || d < worst)
                {
                    heap.push({d, IdEntry::payload_of(pomai::ai::atomic_utils::atomic_load_u64(ids + i))});
                    if (heap.size() > keep_k)
                        heap.pop();
                }
            }
            curr = hdr->next_bucket_offset;
        }
    }

    bool PomaiOrbit::compute_distance_for_id_with_proj(const std::vector<std::vector<float>> & /*qproj*/, float /*qnorm2*/, uint64_t id, float &out_dist)
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
        uint32_t blob_len = load_blob_len_from_body_ptr(ptr);
        if (!safe_blob_range(blob_len, 0, sizeof(BucketHeader)))
            return false;
        BucketHeader hdr;
        std::memcpy(&hdr, ptr, sizeof(hdr));
        if (!safe_blob_range(blob_len, hdr.off_pq_codes + s * sizeof(uint16_t), sizeof(uint16_t)))
            return false;
        const uint16_t *lbase = reinterpret_cast<const uint16_t *>(ptr + hdr.off_pq_codes);
        uint16_t len = __atomic_load_n(lbase + s, __ATOMIC_ACQUIRE);
        if (len == 0 || len > static_cast<uint16_t>(packed_slot_size_))
            return false;
        if (!safe_blob_range(blob_len, hdr.off_vectors + s * packed_slot_size_, len))
            return false;
        (void)ptr;
        (void)hdr;
        (void)len;
        (void)out_dist;
        return false;
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
                if (entry.is_regular_file())
                    info.disk_bytes += entry.file_size();
        }
        return info;
    }

    std::vector<uint64_t> PomaiOrbit::get_all_labels() const
    {
        std::vector<uint64_t> res;
        for (const auto &sh : label_shards_)
        {
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            for (const auto &kv : sh.bucket)
                res.push_back(kv.first);
        }
        return res;
    }

    std::vector<uint64_t> PomaiOrbit::get_centroid_ids(uint32_t) const { return {}; }
    bool PomaiOrbit::get_vectors_raw(const std::vector<uint64_t> &, std::vector<std::string> &) const { return false; }

    bool PomaiOrbit::checkpoint()
    {
        try
        {
            std::unique_lock<std::shared_mutex> cp_lock(checkpoint_mu_);
            std::lock_guard<std::mutex> lk(train_mu_);
            save_schema();
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
            ::sync();
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    void PomaiOrbit::build_echo_graph(float beta, float threshold)
    {
        std::unique_lock<std::mutex> bg_lk(echo_graph_bg_mu_, std::try_to_lock);
        if (!bg_lk)
            return;
        if (centroids_.empty() || (next_graph_snapshot_cid_ >= centroids_.size()))
            return;
        size_t begin = next_graph_snapshot_cid_;
        size_t end = std::min(begin + 32ul, centroids_.size());
        if (adj_snapshot_.size() != centroids_.size())
            adj_snapshot_.resize(centroids_.size());
        L2Func kern = get_pomai_l2sq_kernel();
        for (size_t i = begin; i < end; ++i)
        {
            std::vector<std::pair<float, uint32_t>> dists;
            dists.reserve(centroids_.size());
            for (size_t j = 0; j < centroids_.size(); ++j)
            {
                if (i == j)
                    continue;
                dists.push_back({kern(centroids_[i]->vector.data(), centroids_[j]->vector.data(), cfg_.dim), static_cast<uint32_t>(j)});
            }
            std::sort(dists.begin(), dists.end());
            adj_snapshot_[i].clear();
            for (size_t k = 0; k < std::min<size_t>(dists.size(), 16); ++k)
            {
                adj_snapshot_[i].push_back({dists[k].second, std::exp(-beta * dists[k].first)});
            }
        }
        next_graph_snapshot_cid_ = end;
        if (end == centroids_.size())
        {
            echo_graph_.build_from_adjacency(adj_snapshot_);
            next_graph_snapshot_cid_ = 0;
        }
    }

    void PomaiOrbit::check_and_split_bucket(uint32_t cid)
    {
        constexpr size_t K_SPLIT_THRESHOLD = PomaiOrbit::K_SPLIT_THRESHOLD;
        constexpr uint64_t MIN_SPLIT_INTERVAL_NS = 60ULL * 1000000000ULL;
        if (cid >= bucket_sizes_.size())
            return;
        if (centroids_.size() >= cfg_.algo.max_centroids)
            return;
        static std::atomic<bool> global_splitting_lock{false};
        if (global_splitting_lock.exchange(true))
            return;
        uint32_t sz = pomai::ai::atomic_utils::atomic_load_u32(&bucket_sizes_[cid]);
        if (sz < K_SPLIT_THRESHOLD)
        {
            global_splitting_lock.store(false);
            return;
        }
        uint64_t now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        static std::mutex split_ts_mu;
        static std::unordered_map<uint32_t, uint64_t> split_last_ts;
        {
            std::lock_guard<std::mutex> l(split_ts_mu);
            auto it = split_last_ts.find(cid);
            if (it != split_last_ts.end())
            {
                uint64_t last = it->second;
                if (now < last + MIN_SPLIT_INTERVAL_NS)
                {
                    global_splitting_lock.store(false);
                    return;
                }
            }
            split_last_ts[cid] = now;
        }
        auto new_node = std::make_unique<OrbitNode>();
        new_node->vector.resize(cfg_.dim);
        for (size_t i = 0; i < cfg_.dim; ++i)
            new_node->vector[i] = centroids_[cid]->vector[i] * 1.01f;
        uint32_t new_cid;
        {
            std::unique_lock<std::shared_mutex> lk(checkpoint_mu_);
            new_cid = static_cast<uint32_t>(centroids_.size());
            new_node->bucket_offset.store(alloc_new_bucket(new_cid), std::memory_order_release);
            centroids_.push_back(std::move(new_node));
            bucket_sizes_.push_back(0);
        }
        uint64_t old_bucket_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
        uint64_t new_bucket_off = centroids_[new_cid]->bucket_offset.load(std::memory_order_acquire);
        ArenaView &arena = arena_;
        std::vector<uint64_t> labels_to_move;
        std::vector<uint64_t> src_bucket_offs;
        std::vector<uint32_t> src_slots;
        {
            uint64_t curr = old_bucket_off;
            std::vector<char> tmp;
            while (curr != 0)
            {
                auto base_opt = resolve_bucket_base(arena, curr, tmp);
                if (!base_opt)
                    break;
                const char *body = *base_opt;
                const BucketHeader *hdr = reinterpret_cast<const BucketHeader *>(body);
                uint32_t n = bucket_atomic_load_count(hdr);
                const char *vecs = body + hdr->off_vectors;
                const uint16_t *lens = reinterpret_cast<const uint16_t *>(body + hdr->off_pq_codes);
                const uint64_t *ids = reinterpret_cast<const uint64_t *>(body + hdr->off_ids);
                for (uint32_t i = 0; i < n; ++i)
                {
                    uint64_t packed = pomai::ai::atomic_utils::atomic_load_u64(ids + i);
                    if (IdEntry::tag_of(packed) != IdEntry::TAG_LABEL)
                        continue;
                    uint64_t lbl = IdEntry::payload_of(packed);
                    uint16_t plen = lens[i];
                    if (plen == 0 || plen > static_cast<uint16_t>(packed_slot_size_))
                        continue;
                    std::vector<float> unpacked(cfg_.dim);
                    const uint8_t *pkt = reinterpret_cast<const uint8_t *>(vecs + static_cast<size_t>(i) * packed_slot_size_);
                    if (!zeroharmony_->unpack_to(pkt, plen, centroids_[cid]->vector, unpacked.data()))
                        continue;
                    float d_old = get_pomai_l2sq_kernel()(unpacked.data(), centroids_[cid]->vector.data(), cfg_.dim);
                    float d_new = get_pomai_l2sq_kernel()(unpacked.data(), centroids_[new_cid]->vector.data(), cfg_.dim);
                    if (d_new < d_old)
                    {
                        labels_to_move.push_back(lbl);
                        src_bucket_offs.push_back(curr);
                        src_slots.push_back(i);
                    }
                }
                curr = bucket_atomic_load_next(hdr);
            }
        }
        if (!labels_to_move.empty())
        {
            for (size_t idx = 0; idx < labels_to_move.size(); ++idx)
            {
                uint64_t lbl = labels_to_move[idx];
                uint64_t src_off = src_bucket_offs[idx];
                uint32_t src_slot = src_slots[idx];
                uint64_t cur_bucket = 0;
                uint32_t cur_slot = 0;
                if (!get_label_bucket(lbl, cur_bucket) || !get_label_slot(lbl, cur_slot))
                    continue;
                if (cur_bucket != src_off || cur_slot != src_slot)
                    continue;
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena, src_off, tmp);
                if (!base_opt)
                    continue;
                char *src_body = const_cast<char *>(*base_opt);
                BucketHeader *src_hdr = reinterpret_cast<BucketHeader *>(src_body);
                const char *src_vecs = src_body + src_hdr->off_vectors;
                const uint16_t *src_lens = reinterpret_cast<const uint16_t *>(src_body + src_hdr->off_pq_codes);
                const uint64_t *src_ids = reinterpret_cast<const uint64_t *>(src_body + src_hdr->off_ids);
                uint16_t plen = src_lens[src_slot];
                if (plen == 0 || plen > static_cast<uint16_t>(packed_slot_size_))
                    continue;
                const uint8_t *pkt = reinterpret_cast<const uint8_t *>(src_vecs + static_cast<size_t>(src_slot) * packed_slot_size_);
                uint64_t dst_off = new_bucket_off;
                bool inserted = false;
                while (!inserted)
                {
                    std::vector<char> tmp2;
                    auto base2_opt = resolve_bucket_base(arena, dst_off, tmp2);
                    if (!base2_opt)
                    {
                        break;
                    }
                    char *dst_body = const_cast<char *>(*base2_opt);
                    BucketHeader *dst_hdr = reinterpret_cast<BucketHeader *>(dst_body);
                    uint32_t cur = bucket_atomic_load_count(dst_hdr);
                    if (cur >= dynamic_bucket_capacity_)
                    {
                        uint64_t nxt = bucket_atomic_load_next(dst_hdr);
                        if (nxt == 0)
                        {
                            uint64_t alloc = alloc_new_bucket(new_cid);
                            if (alloc == 0)
                                break;
                            uint64_t expected = 0;
                            pomai::ai::atomic_utils::atomic_compare_exchange_u64(&dst_hdr->next_bucket_offset, expected, alloc);
                            dst_off = bucket_atomic_load_next(dst_hdr);
                            if (!dst_off)
                                break;
                            continue;
                        }
                        dst_off = nxt;
                        continue;
                    }
                    uint32_t slot = bucket_atomic_fetch_add_count(dst_hdr, 1);
                    if (slot + 1 > dynamic_bucket_capacity_)
                    {
                        bucket_atomic_sub_count(dst_hdr, 1);
                        uint64_t nxt = bucket_atomic_load_next(dst_hdr);
                        if (nxt == 0)
                        {
                            uint64_t alloc = alloc_new_bucket(new_cid);
                            if (alloc == 0)
                                break;
                            uint64_t expected = 0;
                            pomai::ai::atomic_utils::atomic_compare_exchange_u64(&dst_hdr->next_bucket_offset, expected, alloc);
                            dst_off = bucket_atomic_load_next(dst_hdr);
                            if (!dst_off)
                                break;
                            continue;
                        }
                        dst_off = nxt;
                        continue;
                    }
                    char *dst_vbase = dst_body + dst_hdr->off_vectors;
                    uint16_t *dst_lbase = reinterpret_cast<uint16_t *>(dst_body + dst_hdr->off_pq_codes);
                    uint64_t *dst_ibase = reinterpret_cast<uint64_t *>(dst_body + dst_hdr->off_ids);
                    std::memcpy(dst_vbase + static_cast<size_t>(slot) * packed_slot_size_, pkt, plen);
                    __atomic_store_n(&dst_lbase[slot], plen, __ATOMIC_RELEASE);
                    uint64_t packed_id = IdEntry::pack_label(lbl);
                    pomai::ai::atomic_utils::atomic_store_u64(dst_ibase + slot, packed_id);
                    set_label_map(lbl, dst_off, slot);
                    pomai::ai::atomic_utils::atomic_fetch_add_u32(&bucket_sizes_[new_cid], 1);
                    inserted = true;
                }
                if (!inserted)
                    continue;
                uint16_t *w_lens = reinterpret_cast<uint16_t *>(src_body + src_hdr->off_pq_codes);
                uint64_t *w_ids = reinterpret_cast<uint64_t *>(src_body + src_hdr->off_ids);
                __atomic_store_n(&w_lens[src_slot], static_cast<uint16_t>(0), __ATOMIC_RELEASE);
                pomai::ai::atomic_utils::atomic_store_u64(w_ids + src_slot, 0);
                pomai::ai::atomic_utils::atomic_fetch_sub_u32(&src_hdr->count, 1);
                pomai::ai::atomic_utils::atomic_fetch_sub_u32(&bucket_sizes_[cid], 1);
            }
        }
        build_echo_graph(1.0f, 0.01f);
        global_splitting_lock.store(false);
    }

} // namespace pomai::ai::orbit
