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

namespace pomai::ai::orbit
{
    using namespace pomai::ai::soa;

    // Absolute upper bound for static buffer sizing (Item.bytes). Must be >= runtime cap.
    static constexpr size_t MAX_ECHO_BYTES = 64 * 1024;
    static constexpr size_t DEFAULT_MAX_SUBBATCH = 4096;

    // Helper: small POD used by batch insertion pipeline to avoid hidden allocations
    // and provide predictable stack-local layout for writes into buckets.
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
                // ignore malformed env
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

        // compute runtime packed slot size (clamped); environment override supported
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
                if (len < 4 + count * (8 + vec_bytes))
                    return true;
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
            return true;
        };
        wal_->replay(replayer);
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

    void PomaiOrbit::apply_thermal_policy() {}

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
            node->bucket_offset.store(0, std::memory_order_relaxed);
            centroids_[i] = std::move(node);
        }
        rebuild_index();
        return true;
    }

    void PomaiOrbit::rebuild_index()
    {
        size_t recovered_count = 0;
        for (const auto &c : centroids_)
        {
            uint64_t curr = c->bucket_offset.load(std::memory_order_relaxed);
            while (curr != 0)
            {
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, curr, tmp);
                if (!base_opt)
                    break;
                const char *ptr = *base_opt;
                uint32_t blob_len = load_blob_len_from_body_ptr(ptr);
                if (!safe_blob_range(blob_len, 0, sizeof(BucketHeader)))
                    break;
                BucketHeader hdr;
                std::memcpy(&hdr, ptr, sizeof(hdr));
                uint32_t n = bucket_atomic_load_count(reinterpret_cast<const BucketHeader *>(ptr));
                if (n > dynamic_bucket_capacity_)
                    break;
                if (!safe_blob_range(blob_len, hdr.off_ids, static_cast<size_t>(n) * sizeof(uint64_t)))
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
                curr = bucket_atomic_load_next(reinterpret_cast<const BucketHeader *>(ptr));
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
        if (cfg_.algo.num_centroids == 0)
            cfg_.algo.num_centroids = 64;
        size_t num_c = std::min(n, static_cast<size_t>(cfg_.algo.num_centroids));
        centroids_.resize(num_c);
        for (size_t i = 0; i < num_c; ++i)
        {
            centroids_[i] = std::make_unique<OrbitNode>();
            centroids_[i]->vector.resize(cfg_.dim);
            std::memcpy(centroids_[i]->vector.data(), data + i * cfg_.dim, cfg_.dim * sizeof(float));
            uint64_t off = alloc_new_bucket(static_cast<uint32_t>(i));
            if (off == 0)
                return false;
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
        if (!vec)
            return false;
        std::vector<float> v(cfg_.dim);
        std::memcpy(v.data(), vec, cfg_.dim * sizeof(float));
        std::vector<std::pair<uint64_t, std::vector<float>>> b;
        b.emplace_back(label, std::move(v));
        return insert_batch(b);
    }

    // NOTE: WAL stores the raw float vectors (backups), while in-memory buckets store packed bytes
    // using ZeroHarmonyPacker. WAL append is done first (best-effort), then memory insert proceeds.
    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (batch.empty())
            return false;

        std::shared_lock<std::shared_mutex> cp_lock(checkpoint_mu_);

        // WAL: serialize as [uint32 count][(uint64 label)(float * dim)]...
        if (wal_)
        {
            std::vector<uint8_t> buffer;
            buffer.reserve(4 + batch.size() * (8 + cfg_.dim * 4));
            uint32_t cnt = 0;
            // We'll only write valid-dim items to WAL
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

            // Best-effort WAL append; failure doesn't abort insert into memory.
            try
            {
                wal_->append_record(20, buffer.data(), static_cast<uint32_t>(buffer.size()));
            }
            catch (...)
            {
                // swallow: WAL is best-effort for now
            }
        }

        return insert_batch_memory_only(batch);
    }

    // Memory-only insert path: pack vectors using ZeroHarmony and write into centroid buckets.
    bool PomaiOrbit::insert_batch_memory_only(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (!zeroharmony_)
            return false;

        // Auto-train centroids if empty (legacy support)
        if (centroids_.empty())
        {
            std::lock_guard<std::mutex> lk(train_mu_);
            if (centroids_.empty())
            {
                std::vector<float> training_data;
                training_data.reserve(batch.size() * cfg_.dim);
                for (const auto &item : batch)
                {
                    if (item.second.size() == cfg_.dim)
                        training_data.insert(training_data.end(), item.second.begin(), item.second.end());
                }
                if (!training_data.empty())
                    this->train(training_data.data(), training_data.size() / cfg_.dim);
            }
        }
        if (centroids_.empty())
            return false;

        // Prepare items: pack per-centroid with zero-allocation destination (Item.bytes)
        std::vector<Item> prepared;
        prepared.reserve(batch.size());

        const size_t safe_cap = packed_slot_size_;

        for (const auto &p : batch)
        {
            if (p.second.size() != cfg_.dim)
                continue;

            uint32_t cid = find_nearest_centroid(p.second.data());

            std::vector<uint8_t> packed = zeroharmony_->pack_with_mean(p.second.data(), centroids_[cid]->vector);
            if (packed.empty())
                continue;

            if (packed.size() > safe_cap)
            {
                std::cerr << "[Orbit] Error: packed size " << packed.size() << " exceeds packed_slot_size_ " << safe_cap << " for label=" << p.first << ". Rejecting batch.\n";
                return false;
            }

            Item it;
            it.label = p.first;
            it.cid = cid;
            std::memcpy(it.bytes, packed.data(), packed.size());
            it.size = static_cast<uint8_t>(packed.size());
            prepared.push_back(it);
        }

        if (prepared.empty())
            return true;

        // Sort by centroid to increase locality of writes
        std::sort(prepared.begin(), prepared.end(), [](const Item &a, const Item &b)
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
                    return false;
                node.bucket_offset.store(off, std::memory_order_release);
            }

            size_t write_idx = idx;
            while (write_idx < group_end)
            {
                std::vector<char> tmp;
                auto base = resolve_bucket_base(arena_, off, tmp);
                if (!base)
                    return false;

                char *ptr = const_cast<char *>(*base);
                BucketHeader *hdr = reinterpret_cast<BucketHeader *>(ptr);

                // Atomic reservation of 'fit' slots
                uint32_t cur = bucket_atomic_load_count(hdr);
                if (cur >= dynamic_bucket_capacity_)
                {
                    uint64_t nxt = bucket_atomic_load_next(hdr);
                    if (nxt == 0)
                    {
                        nxt = alloc_new_bucket(cid);
                        if (nxt == 0)
                            return false;
                        bucket_atomic_store_next(hdr, nxt);
                    }
                    off = nxt;
                    continue;
                }

                uint32_t rem = dynamic_bucket_capacity_ - cur;
                uint32_t fit = static_cast<uint32_t>(std::min((size_t)rem, group_end - write_idx));
                uint32_t slot = bucket_atomic_fetch_add_count(hdr, fit);

                // Overshoot check and rollback
                if (slot + fit > dynamic_bucket_capacity_)
                {
                    bucket_atomic_sub_count(hdr, fit);
                    uint64_t nxt = bucket_atomic_load_next(hdr);
                    if (nxt == 0)
                    {
                        nxt = alloc_new_bucket(cid);
                        if (nxt == 0)
                            return false;
                        bucket_atomic_store_next(hdr, nxt);
                    }
                    off = nxt;
                    continue;
                }

                // Pointers into bucket payload (vectors, lengths, ids)
                char *vbase = ptr + hdr->off_vectors;
                uint16_t *lbase = reinterpret_cast<uint16_t *>(ptr + hdr->off_pq_codes);
                uint64_t *ibase = reinterpret_cast<uint64_t *>(ptr + hdr->off_ids);

                // Write the batch of 'fit' items
                for (uint32_t k = 0; k < fit; ++k)
                {
                    const Item &it = prepared[write_idx + k];
                    uint32_t s = slot + k;

                    // Write packed bytes with fixed stride packed_slot_size_
                    std::memcpy(vbase + static_cast<size_t>(s) * packed_slot_size_, it.bytes, it.size);

                    // Publish length with release semantics to signal readers that data is valid
                    __atomic_store_n(&lbase[s], static_cast<uint16_t>(it.size), __ATOMIC_RELEASE);

                    // Store label as packed IdEntry::TAG_LABEL
                    uint64_t packed_id = IdEntry::pack_label(it.label);
                    pomai::ai::atomic_utils::atomic_store_u64(ibase + s, packed_id);

                    // Update label shard map for fast lookup
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
        std::unique_lock<std::shared_mutex> lk(del_mu_);
        deleted_labels_.insert(label);
        return true;
    }

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
        (void)budget;
        if (!zeroharmony_ || centroids_.empty())
            return {};
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
                uint32_t blob_len = load_blob_len_from_body_ptr(ptr);
                if (!safe_blob_range(blob_len, 0, sizeof(BucketHeader)))
                    break;
                BucketHeader hdr;
                std::memcpy(&hdr, ptr, sizeof(hdr));
                uint32_t n = bucket_atomic_load_count(reinterpret_cast<const BucketHeader *>(ptr));
                if (n > dynamic_bucket_capacity_)
                    break;
                if (!safe_blob_range(blob_len, hdr.off_ids, static_cast<size_t>(n) * sizeof(uint64_t)))
                    break;
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
                    uint16_t v_len = __atomic_load_n(&len_ptr[i], __ATOMIC_ACQUIRE);
                    if (v_len == 0 || v_len > static_cast<uint16_t>(packed_slot_size_))
                        continue;
                    if (!safe_blob_range(blob_len, hdr.off_vectors + i * packed_slot_size_, v_len))
                        continue;
                    const uint8_t *code = reinterpret_cast<const uint8_t *>(vec_base + i * packed_slot_size_);
                    const std::vector<float> &mean = centroids_[cid]->vector;
                    float d = zeroharmony_->approx_dist(query, code, v_len, mean);
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
        res.reserve(topk.size());
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
        std::priority_queue<std::pair<float, uint64_t>> topk;
        for (uint64_t id : candidates)
        {
            {
                std::shared_lock<std::shared_mutex> dl(del_mu_);
                if (deleted_labels_.count(id))
                    continue;
            }
            uint64_t b = 0;
            uint32_t s = 0;
            if (!get_label_bucket(id, b) || !get_label_slot(id, s))
                continue;
            std::vector<char> tmp;
            auto base = resolve_bucket_base(arena_, b, tmp);
            if (!base)
                continue;
            const char *ptr = *base;
            uint32_t blob_len = load_blob_len_from_body_ptr(ptr);
            if (!safe_blob_range(blob_len, 0, sizeof(BucketHeader)))
                continue;
            BucketHeader hdr;
            std::memcpy(&hdr, ptr, sizeof(hdr));
            if (!safe_blob_range(blob_len, hdr.off_pq_codes + s * sizeof(uint16_t), sizeof(uint16_t)))
                continue;
            const uint16_t *lbase = reinterpret_cast<const uint16_t *>(ptr + hdr.off_pq_codes);
            uint16_t len = __atomic_load_n(lbase + s, __ATOMIC_ACQUIRE);
            if (len == 0 || len > static_cast<uint16_t>(packed_slot_size_))
                continue;
            if (!safe_blob_range(blob_len, hdr.off_vectors + s * packed_slot_size_, len))
                continue;
            const uint8_t *code = reinterpret_cast<const uint8_t *>(ptr + hdr.off_vectors + s * packed_slot_size_);
            uint32_t cid = hdr.centroid_id;
            if (cid >= centroids_.size())
                continue;
            const std::vector<float> &mean = centroids_[cid]->vector;
            float d = zeroharmony_->approx_dist(query, code, len, mean);
            if (topk.size() < k)
                topk.push({d, id});
            else if (d < topk.top().first)
            {
                topk.pop();
                topk.push({d, id});
            }
        }
        std::vector<std::pair<uint64_t, float>> res;
        res.reserve(topk.size());
        while (!topk.empty())
        {
            res.push_back({topk.top().second, topk.top().first});
            topk.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
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