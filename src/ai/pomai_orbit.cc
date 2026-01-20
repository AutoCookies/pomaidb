/*
 * src/ai/pomai_orbit.cc
 *
 * PomaiOrbit implementation (Corrected WalManager integration).
 */

#include "src/ai/pomai_orbit.h"

#include "src/ai/atomic_utils.h"
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
    // Constants
    static constexpr size_t MAX_ECHO_BYTES = 64;
    static constexpr uint32_t RESERVE_CHUNK = 16;
    static constexpr size_t DEFAULT_MAX_SUBBATCH = 4096;

    static constexpr uint32_t kCostSetup = 200;
    static constexpr uint32_t kCostRoute = 50;
    static constexpr uint32_t kCostBucket = 20;
    static constexpr uint32_t kCostItem = 5;
    static constexpr uint32_t kCostCheck = 2;

    struct ThreadReserve
    {
        uint32_t base = 0;
        uint32_t remain = 0;
        uint64_t bucket_off = 0;
    };

    static inline float *thread_local_recon(size_t dim)
    {
        static thread_local std::vector<float> recon_buf;
        if (recon_buf.size() < dim)
            recon_buf.resize(dim);
        return recon_buf.data();
    }

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

    // ------------------- Constructors -------------------
    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        dynamic_bucket_capacity_ = cfg_.algo.initial_bucket_cap;
        if (dynamic_bucket_capacity_ == 0)
            dynamic_bucket_capacity_ = 128;

        if (!arena_.is_pomai_arena())
            throw std::invalid_argument("PomaiOrbit: arena null or wrong type");
        if (cfg_.dim == 0)
            throw std::invalid_argument("PomaiOrbit: cfg.dim must be > 0");

        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
        schema_file_path_ = (p / "pomai_schema.bin").string();

        eeq_ = std::make_unique<pomai::ai::EternalEchoQuantizer>(cfg_.dim, cfg_.eeq_cfg);

        if (std::filesystem::exists(schema_file_path_))
        {
            std::clog << "[Orbit] Found existing DB. Loading schema...\n";
            if (!load_schema())
                std::cerr << "[Orbit] Failed to load schema!\n";
        }
        else
        {
            std::clog << "[Orbit] Initializing NEW Database\n";
            save_schema();
        }

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

        // [FIX] Initialize WAL with correct config struct
        wal_ = std::make_unique<pomai::memory::WalManager>();
        std::string wal_path = cfg_.data_path + "/orbit.wal";

        pomai::memory::WalManager::WalConfig wcfg;
        wcfg.sync_on_append = true; // Ensure durability

        if (!wal_->open(wal_path, true, wcfg))
        {
            std::cerr << "[Orbit] FATAL: Cannot open WAL at " << wal_path << "\n";
        }
        else
        {
            std::clog << "[Orbit] Replaying WAL from " << wal_path << "...\n";
            recover_from_wal();
        }

        if (!centroids_.empty())
        {
            init_thermal_map(centroids_.size());
        }
    }

    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::ShardArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        dynamic_bucket_capacity_ = cfg_.algo.initial_bucket_cap;
        if (dynamic_bucket_capacity_ == 0)
            dynamic_bucket_capacity_ = 128;

        if (!arena_.is_shard_arena())
            throw std::invalid_argument("PomaiOrbit: shard arena null or wrong type");
        if (cfg_.dim == 0)
            throw std::invalid_argument("PomaiOrbit: cfg.dim must be > 0");

        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
        schema_file_path_ = (p / "pomai_schema.bin").string();

        eeq_ = std::make_unique<pomai::ai::EternalEchoQuantizer>(cfg_.dim, cfg_.eeq_cfg);

        if (std::filesystem::exists(schema_file_path_))
        {
            std::clog << "[Orbit] Found existing DB. Loading schema...\n";
            if (!load_schema())
                std::cerr << "[Orbit] Failed to load schema!\n";
        }
        else
        {
            std::clog << "[Orbit] Initializing NEW Database\n";
            save_schema();
        }

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

        // [FIX] Initialize WAL with correct config struct
        wal_ = std::make_unique<pomai::memory::WalManager>();
        std::string wal_path = cfg_.data_path + "/orbit.wal";

        pomai::memory::WalManager::WalConfig wcfg;
        wcfg.sync_on_append = true;

        if (!wal_->open(wal_path, true, wcfg))
        {
            std::cerr << "[Orbit] FATAL: Cannot open WAL at " << wal_path << "\n";
        }
        else
        {
            std::clog << "[Orbit] Replaying WAL from " << wal_path << "...\n";
            recover_from_wal();
        }

        if (!centroids_.empty())
        {
            init_thermal_map(centroids_.size());
        }
    }

    PomaiOrbit::~PomaiOrbit()
    {
        if (cortex_)
            cortex_->stop();
    }

    // ------------------- Recovery Logic -------------------
    void PomaiOrbit::recover_from_wal()
    {
        size_t replayed_count = 0;
        size_t batches_count = 0;

        auto replayer = [&](uint16_t type, const void *payload, uint32_t len, uint64_t seq) -> bool
        {
            if (type == 20) // WAL_REC_INSERT_BATCH
            {
                if (len < 4)
                    return true;
                const uint8_t *ptr = static_cast<const uint8_t *>(payload);
                const uint8_t *end = ptr + len;

                uint32_t count = 0;
                std::memcpy(&count, ptr, 4);
                ptr += 4;

                std::vector<std::pair<uint64_t, std::vector<float>>> batch;
                batch.reserve(count);
                size_t vec_bytes = cfg_.dim * sizeof(float);

                for (uint32_t i = 0; i < count; ++i)
                {
                    if (ptr + 8 + vec_bytes > end)
                        break;
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
                    insert_batch_memory_only(batch);
                    replayed_count += batch.size();
                    batches_count++;
                }
            }
            return true;
        };

        wal_->replay(replayer);
        if (replayed_count > 0)
        {
            std::clog << "[Orbit] Recovery complete. Replayed " << replayed_count
                      << " vectors from " << batches_count << " batches.\n";
        }
    }

    // ------------------- Thermal / Helper Methods -------------------
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
        uint8_t old_temp = thermal_map_[cid].load(std::memory_order_relaxed);
        if (old_temp < 255)
        {
            uint8_t boost = (old_temp < 50) ? 50 : 5;
            int new_val = old_temp + boost;
            thermal_map_[cid].store((new_val > 255) ? 255 : static_cast<uint8_t>(new_val), std::memory_order_relaxed);
        }
        uint32_t now = static_cast<uint32_t>(std::time(nullptr));
        last_access_epoch_[cid].store(now, std::memory_order_relaxed);
    }

    uint8_t PomaiOrbit::get_temperature(uint32_t cid) const
    {
        if (cid >= thermal_map_.size())
            return 0;
        uint8_t temp = thermal_map_[cid].load(std::memory_order_relaxed);
        if (temp == 0)
            return 0;
        uint32_t last = last_access_epoch_[cid].load(std::memory_order_relaxed);
        uint32_t now = static_cast<uint32_t>(std::time(nullptr));
        if (now > last)
        {
            uint32_t delta = now - last;
            uint32_t cooldown = (delta / 60) * 10;
            if (cooldown >= temp)
                return 0;
            return temp - cooldown;
        }
        return temp;
    }

    void PomaiOrbit::apply_thermal_policy()
    {
        if (!arena_.is_shard_arena())
            return;
        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            uint8_t temp = get_temperature(static_cast<uint32_t>(i));
            if (temp < 5)
            {
                uint64_t off = centroids_[i]->bucket_offset.load(std::memory_order_relaxed);
                if (off != 0)
                    arena_.demote_range(off, 4096);
            }
        }
    }

    void PomaiOrbit::save_schema()
    {
        SchemaHeader header;
        header.magic_number = 0x504F4D41;
        header.dim = cfg_.dim;
        header.num_centroids = centroids_.size(); // Lưu số lượng centroid hiện tại

        // Tính tổng vector (để thống kê)
        uint64_t total = 0;
        for (size_t si = 0; si < kLabelShardCount; ++si)
        {
            std::shared_lock<std::shared_mutex> lk(label_shards_[si].mu);
            total += label_shards_[si].bucket.size();
        }
        header.total_vectors = total;

        std::ofstream out(schema_file_path_, std::ios::binary | std::ios::trunc);
        if (!out.is_open())
        {
            std::clog << "[Orbit] ERROR: Could not save schema to " << schema_file_path_ << "\n";
            return;
        }

        // 1. Write Header
        out.write(reinterpret_cast<const char *>(&header), sizeof(header));

        // 2. Write Centroids (Vector + Bucket Offset)
        // Đây là "Neo" để móc vào dữ liệu trên đĩa
        for (const auto &c : centroids_)
        {
            // Write vector info
            out.write(reinterpret_cast<const char *>(c->vector.data()), cfg_.dim * sizeof(float));
            // Write bucket offset (Head pointer)
            uint64_t off = c->bucket_offset.load(std::memory_order_acquire);
            out.write(reinterpret_cast<const char *>(&off), sizeof(off));
        }

        out.close();
        // std::clog << "[Orbit] Schema saved. Centroids: " << header.num_centroids << "\n";
    }

    bool PomaiOrbit::load_schema()
    {
        std::ifstream in(schema_file_path_, std::ios::binary);
        if (!in.is_open())
            return false;

        // 1. Read Header
        SchemaHeader header;
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        if (header.magic_number != 0x504F4D41)
            return false;

        cfg_.dim = header.dim;
        cfg_.algo.num_centroids = static_cast<uint32_t>(header.num_centroids);

        // 2. Load Centroids
        centroids_.clear();
        centroids_.resize(header.num_centroids);

        for (size_t i = 0; i < header.num_centroids; ++i)
        {
            auto node = std::make_unique<OrbitNode>();
            node->vector.resize(cfg_.dim);

            // Read vector
            in.read(reinterpret_cast<char *>(node->vector.data()), cfg_.dim * sizeof(float));

            // Read bucket offset
            uint64_t off = 0;
            in.read(reinterpret_cast<char *>(&off), sizeof(off));
            node->bucket_offset.store(off, std::memory_order_relaxed);

            // Rebuild neighbors (simplified: just init empty, or rebuild properly if needed)
            // Trong thực tế cần lưu cả neighbors list, nhưng ở đây ta có thể lười (lazy) hoặc train lại routing.
            // Để đơn giản, ta để neighbors trống, hệ thống vẫn chạy (nhưng search chậm hơn chút).

            centroids_[i] = std::move(node);
        }

        in.close();
        std::clog << "[Orbit] Schema loaded. Restored " << header.num_centroids << " centroids.\n";

        // 3. Rebuild In-Memory Index (Label -> Bucket Map)
        // Vì Label Map chỉ nằm trên RAM, ta phải quét lại Buckets để xây dựng lại nó.
        rebuild_index();

        return true;
    }

    void PomaiOrbit::rebuild_index()
    {
        std::clog << "[Orbit] Rebuilding index from disk data...\n";
        size_t count = 0;

        for (const auto &c : centroids_)
        {
            uint64_t curr = c->bucket_offset.load(std::memory_order_relaxed);
            while (curr != 0)
            {
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, curr, tmp);
                if (!base_opt)
                    break;

                // [FIX] Lấy con trỏ thô (char*)
                const char *ptr = *base_opt;

                // [FIX] KHÔNG ép kiểu trực tiếp. Dùng memcpy để copy sang biến stack đã căn chỉnh.
                BucketHeader hdr_copy;
                std::memcpy(&hdr_copy, ptr, sizeof(BucketHeader));

                // Đọc count từ bản copy an toàn
                uint32_t n = hdr_copy.count.load(std::memory_order_relaxed);

                // Sanity check: Nếu file lỗi, n có thể là số rác cực lớn -> treo máy
                if (n > 1000000)
                {
                    std::cerr << "[Orbit] Warning: Corrupt bucket count (" << n << "). Stopping chain.\n";
                    break;
                }

                // Tính vị trí mảng ID (cũng là pointer thô)
                const char *ids_ptr_raw = ptr + hdr_copy.off_ids;

                for (uint32_t i = 0; i < n; ++i)
                {
                    // [FIX] Copy từng ID (8 bytes) an toàn
                    uint64_t label;
                    std::memcpy(&label, ids_ptr_raw + i * sizeof(uint64_t), sizeof(uint64_t));

                    set_label_map(label, curr, i);
                    count++;
                }

                // Chuyển sang bucket tiếp theo
                curr = hdr_copy.next_bucket_offset.load(std::memory_order_relaxed);
            }
        }
        std::clog << "[Orbit] Index rebuilt. Total vectors: " << count << "\n";

        // Re-init routing graph (Giữ nguyên logic cũ)
        if (!centroids_.empty())
        {
            L2Func kern = get_pomai_l2sq_kernel();
            size_t num_c = centroids_.size();
#pragma omp parallel for
            for (size_t i = 0; i < num_c; ++i)
            {
                std::vector<std::pair<float, uint32_t>> dists;
                dists.reserve(num_c);
                for (size_t j = 0; j < num_c; ++j)
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
                {
                    centroids_[i]->neighbors.push_back(dists[k].second);
                }
            }
        }
    }

    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0)
            return false;
        if (cfg_.algo.num_centroids == 0)
        {
            size_t suggest = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
            cfg_.algo.num_centroids = static_cast<uint32_t>(std::clamp(suggest, static_cast<size_t>(64), static_cast<size_t>(4096)));
        }
        size_t avg_density = (n / cfg_.algo.num_centroids) + 1;
        dynamic_bucket_capacity_ = static_cast<uint32_t>(avg_density * 1.5);
        dynamic_bucket_capacity_ = std::clamp(dynamic_bucket_capacity_, static_cast<uint32_t>(32), static_cast<uint32_t>(512));

        std::clog << "[Orbit Autopilot] Training Config: Centroids=" << cfg_.algo.num_centroids
                  << ", BucketCap=" << dynamic_bucket_capacity_ << " (N=" << n << ")\n";

        size_t num_c = std::min(n, static_cast<size_t>(cfg_.algo.num_centroids));
        centroids_.resize(num_c);
        for (size_t i = 0; i < num_c; ++i)
            centroids_[i] = std::make_unique<OrbitNode>();

        std::mt19937 rng(42);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(num_c);

        for (size_t i = 0; i < num_c; ++i)
        {
            centroids_[i]->vector.resize(cfg_.dim);
            std::memcpy(centroids_[i]->vector.data(), data + indices[i] * cfg_.dim, cfg_.dim * sizeof(float));
            centroids_[i]->neighbors.reserve(cfg_.algo.m_neighbors * 2);
        }

        {
            L2Func kern = get_pomai_l2sq_kernel();
            for (size_t i = 0; i < num_c; ++i)
            {
                std::vector<std::pair<float, uint32_t>> dists;
                for (size_t j = 0; j < num_c; ++j)
                {
                    if (i == j)
                        continue;
                    float d = kern(centroids_[i]->vector.data(), centroids_[j]->vector.data(), cfg_.dim);
                    dists.push_back({d, static_cast<uint32_t>(j)});
                }
                std::sort(dists.begin(), dists.end());
                for (size_t k = 0; k < std::min(dists.size(), static_cast<size_t>(cfg_.algo.m_neighbors)); ++k)
                    centroids_[i]->neighbors.push_back(dists[k].second);
            }
        }

        for (size_t i = 0; i < num_c; ++i)
        {
            uint64_t off = alloc_new_bucket(static_cast<uint32_t>(i));
            centroids_[i]->bucket_offset.store(off, std::memory_order_release);
        }
        save_schema();
        init_thermal_map(centroids_.size());
        return true;
    }

    uint64_t PomaiOrbit::alloc_new_bucket(uint32_t centroid_id)
    {
        size_t cap = dynamic_bucket_capacity_;
        size_t head_sz = sizeof(BucketHeader);
        size_t code_len_sz = sizeof(uint16_t) * cap;
        size_t vec_sz = MAX_ECHO_BYTES * cap;
        size_t ids_sz = sizeof(uint64_t) * cap;
        auto align64 = [](size_t s)
        { return (s + 63) & ~63; };

        uint32_t off_fp = static_cast<uint32_t>(align64(head_sz));
        uint32_t off_pq = static_cast<uint32_t>(align64(off_fp + 0));
        uint32_t off_vec = static_cast<uint32_t>(align64(off_pq + code_len_sz));
        uint32_t off_ids = static_cast<uint32_t>(align64(off_vec + vec_sz));
        size_t total_bytes = off_ids + ids_sz;

        char *blob_ptr = arena_.alloc_blob(static_cast<uint32_t>(total_bytes));
        if (!blob_ptr)
            return 0;

        uint64_t offset = arena_.offset_from_blob_ptr(blob_ptr);
        BucketHeader *hdr = reinterpret_cast<BucketHeader *>(blob_ptr + sizeof(uint32_t));
        new (hdr) BucketHeader();

        hdr->centroid_id = centroid_id;
        hdr->count.store(0, std::memory_order_relaxed);
        hdr->next_bucket_offset.store(0, std::memory_order_relaxed);
        hdr->off_fingerprints = off_fp;
        hdr->off_pq_codes = off_pq;
        hdr->off_vectors = off_vec;
        hdr->off_ids = off_ids;
        hdr->synapse_scale = 1.0f;
        hdr->is_frozen = false;
        hdr->disk_offset = 0;
        hdr->last_access_ms = 0;

        uint16_t *len_base = reinterpret_cast<uint16_t *>(blob_ptr + sizeof(uint32_t) + off_pq);
        for (size_t i = 0; i < cap; ++i)
            __atomic_store_n(len_base + i, static_cast<uint16_t>(0), __ATOMIC_RELAXED);

        return offset;
    }

    uint32_t PomaiOrbit::find_nearest_centroid(const float *vec)
    {
        uint32_t best = 0;
        float min_d = std::numeric_limits<float>::max();
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
        using NodeDist = std::pair<float, uint32_t>;
        std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> pq;
        std::unordered_set<uint32_t> visited;

        uint32_t entry = 0;
        L2Func kern = get_pomai_l2sq_kernel();
        pq.push({kern(vec, centroids_[entry]->vector.data(), cfg_.dim), entry});
        visited.insert(entry);

        std::vector<uint32_t> res;
        while (!pq.empty())
        {
            auto cur = pq.top();
            pq.pop();
            res.push_back(cur.second);
            if (res.size() >= n * 2)
                break;
            const auto &node = *centroids_[cur.second];
            for (uint32_t nb : node.neighbors)
            {
                if (visited.insert(nb).second)
                {
                    float d = kern(vec, centroids_[nb]->vector.data(), cfg_.dim);
                    pq.push({d, nb});
                }
            }
        }
        if (res.size() > n)
            res.resize(n);
        return res;
    }

    void PomaiOrbit::set_label_map(uint64_t label, uint64_t bucket_off, uint32_t slot)
    {
        size_t si = label_shard_index(label);
        LabelShard &sh = label_shards_[si];
        std::unique_lock<std::shared_mutex> lk(sh.mu);
        sh.bucket[label] = bucket_off;
        sh.slot[label] = slot;
    }

    bool PomaiOrbit::get_label_bucket(uint64_t label, uint64_t &out_bucket) const
    {
        size_t si = label_shard_index(label);
        const LabelShard &sh = label_shards_[si];
        std::shared_lock<std::shared_mutex> lk(sh.mu);
        auto it = sh.bucket.find(label);
        if (it == sh.bucket.end())
            return false;
        out_bucket = it->second;
        return true;
    }

    bool PomaiOrbit::get_label_slot(uint64_t label, uint32_t &out_slot) const
    {
        size_t si = label_shard_index(label);
        const LabelShard &sh = label_shards_[si];
        std::shared_lock<std::shared_mutex> lk(sh.mu);
        auto it = sh.slot.find(label);
        if (it == sh.slot.end())
            return false;
        out_slot = it->second;
        return true;
    }

    bool PomaiOrbit::insert(const float *vec, uint64_t label)
    {
        std::vector<float> v(vec, vec + cfg_.dim);
        std::vector<std::pair<uint64_t, std::vector<float>>> batch;
        batch.push_back({label, std::move(v)});
        return insert_batch(batch);
    }

    // ------------------- insert_batch (WAL + Memory) -------------------
    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (batch.empty())
            return false;

        // [THÊM] Shared Lock: Cho phép nhiều luồng insert, nhưng chặn Checkpoint
        std::shared_lock<std::shared_mutex> cp_lock(checkpoint_mu_);

        // 1. [WAL STEP] Serialize & Append to Disk
        if (wal_)
        {
            // ... (Code ghi WAL cũ giữ nguyên) ...
            // Copy đoạn tính size và wal_->append_record(...) vào đây
            size_t vec_size = cfg_.dim * sizeof(float);
            size_t total_size = 4 + batch.size() * (8 + vec_size);
            std::vector<uint8_t> buffer(total_size);
            uint8_t *ptr = buffer.data();
            uint32_t count = static_cast<uint32_t>(batch.size());
            std::memcpy(ptr, &count, 4);
            ptr += 4;
            for (const auto &item : batch)
            {
                std::memcpy(ptr, &item.first, 8);
                ptr += 8;
                if (item.second.size() == cfg_.dim)
                    std::memcpy(ptr, item.second.data(), vec_size);
                else
                    std::memset(ptr, 0, vec_size);
                ptr += vec_size;
            }
            wal_->append_record(20, buffer.data(), static_cast<uint32_t>(total_size));
        }

        // 2. [MEMORY STEP]
        return insert_batch_memory_only(batch);
    }

    // ------------------- insert_batch_memory_only (Original Logic) -------------------
    bool PomaiOrbit::insert_batch_memory_only(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (!eeq_)
            return false;

        if (centroids_.empty())
        {
            std::lock_guard<std::mutex> lk(train_mu_);
            if (centroids_.empty())
            {
                if (batch.empty())
                    return false;
                std::vector<float> training_data;
                training_data.reserve(batch.size() * cfg_.dim);
                size_t valid_count = 0;
                for (const auto &item : batch)
                {
                    if (item.second.size() == cfg_.dim)
                    {
                        training_data.insert(training_data.end(), item.second.begin(), item.second.end());
                        valid_count++;
                    }
                }
                if (valid_count > 0)
                {
                    size_t original_k = cfg_.algo.num_centroids;
                    if (original_k == 0 || original_k > valid_count / 2)
                    {
                        size_t suggest = static_cast<size_t>(std::sqrt(valid_count));
                        cfg_.algo.num_centroids = static_cast<uint32_t>(std::max<size_t>(1, suggest));
                        std::clog << "[Orbit] Auto-adjusted num_centroids to " << cfg_.algo.num_centroids << " based on batch size.\n";
                    }
                    this->train(training_data.data(), valid_count);
                }
            }
        }

        if (centroids_.empty() || batch.empty())
            return false;

        size_t max_subbatch = DEFAULT_MAX_SUBBATCH;
        const char *env = std::getenv("POMAI_MAX_BATCH_INSERT");
        if (env)
        {
            try
            {
                max_subbatch = static_cast<size_t>(std::stoul(env));
            }
            catch (...)
            {
            }
            if (max_subbatch == 0)
                max_subbatch = DEFAULT_MAX_SUBBATCH;
        }

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
            uint64_t label = p.first;
            const std::vector<float> &vec = p.second;
            if (vec.size() != cfg_.dim)
                continue;

            uint32_t cid = find_nearest_centroid(vec.data());
            pomai::ai::EchoCode code;
            try
            {
                code = eeq_->encode(vec.data());
            }
            catch (...)
            {
                continue;
            }

            Item it;
            it.label = label;
            size_t pos = 0;
            it.bytes[pos++] = code.depth;
            for (uint8_t q : code.scales_q)
                if (pos < MAX_ECHO_BYTES)
                    it.bytes[pos++] = q;
            bool overflow = false;
            for (const auto &sb : code.sign_bytes)
            {
                if (pos + sb.size() > MAX_ECHO_BYTES)
                {
                    overflow = true;
                    break;
                }
                std::memcpy(it.bytes + pos, sb.data(), sb.size());
                pos += sb.size();
            }
            if (overflow || pos == 0)
                continue;
            it.size = static_cast<uint8_t>(pos);
            it.cid = cid;
            prepared.push_back(it);
        }

        if (prepared.empty())
            return true;

        std::sort(prepared.begin(), prepared.end(), [](auto &a, auto &b)
                  { return a.cid < b.cid; });

        size_t idx = 0, total = prepared.size();
        while (idx < total)
        {
            uint32_t cid = prepared[idx].cid;
            size_t group_end = idx;
            while (group_end < total && prepared[group_end].cid == cid && (group_end - idx) < max_subbatch)
                ++group_end;

            OrbitNode &node = *centroids_[cid];
            std::unique_lock<std::shared_mutex> lock(node.mu);

            uint64_t current_off = node.bucket_offset.load(std::memory_order_acquire);
            if (current_off == 0)
            {
                current_off = alloc_new_bucket(cid);
                if (current_off == 0)
                    return false;
                node.bucket_offset.store(current_off, std::memory_order_release);
            }

            size_t write_idx = idx;
            while (write_idx < group_end)
            {
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, current_off, tmp);
                if (!base_opt)
                {
                    uint64_t nb = alloc_new_bucket(cid);
                    if (nb == 0)
                        return false;
                    const char *prev_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                    if (prev_ptr)
                    {
                        BucketHeader *hdr_prev = reinterpret_cast<BucketHeader *>(const_cast<char *>(prev_ptr) + sizeof(uint32_t));
                        hdr_prev->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    continue;
                }

                char *bucket_ptr = const_cast<char *>(*base_opt);
                BucketHeader *hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);
                uint32_t cur_count = hdr->count.load(std::memory_order_relaxed);

                if (cur_count >= dynamic_bucket_capacity_)
                {
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(cid);
                        if (nb == 0)
                            return false;
                        hdr->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    continue;
                }

                uint32_t remaining = dynamic_bucket_capacity_ - cur_count;
                uint32_t fit = 0;
                size_t probe = write_idx;
                while (probe < group_end && fit < remaining)
                {
                    ++fit;
                    ++probe;
                }

                if (fit == 0)
                {
                    current_off = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (current_off == 0)
                    {
                        current_off = alloc_new_bucket(cid);
                        if (!current_off)
                            return false;
                        hdr->next_bucket_offset.store(current_off, std::memory_order_release);
                    }
                    continue;
                }

                uint32_t slot_base = hdr->count.fetch_add(fit, std::memory_order_acq_rel);
                if (slot_base + fit > dynamic_bucket_capacity_)
                {
                    hdr->count.fetch_sub(fit, std::memory_order_acq_rel);
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(cid);
                        if (nb == 0)
                            return false;
                        hdr->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    continue;
                }

                char *vec_area = bucket_ptr + hdr->off_vectors;
                uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
                uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);

                for (uint32_t j = 0; j < fit; ++j)
                {
                    const Item &it = prepared[write_idx + j];
                    uint32_t slot = slot_base + j;
                    char *slot_ptr = vec_area + static_cast<size_t>(slot) * MAX_ECHO_BYTES;
                    std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
                    std::memcpy(slot_ptr, it.bytes, it.size);
                    pomai::ai::atomic_utils::atomic_store_u64(id_base + slot, it.label);
                    __atomic_store_n(&len_base[slot], static_cast<uint16_t>(it.size), __ATOMIC_RELEASE);

                    size_t si = label_shard_index(it.label);
                    LabelShard &sh = label_shards_[si];
                    std::unique_lock<std::shared_mutex> lm(sh.mu);
                    sh.bucket[it.label] = current_off;
                    sh.slot[it.label] = slot;
                }
                write_idx += fit;
            }
            idx = group_end;
            PomaiMetrics::batch_subbatches_processed.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }

    bool PomaiOrbit::get(uint64_t label, std::vector<float> &out_vec)
    {
        // 1. Check deleted
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.count(label))
                return false;
        }

        // 2. Find location
        uint64_t bucket_off = 0;
        // Lưu ý: get_label_bucket cần được định nghĩa const hoặc dùng const_cast an toàn
        // Giả sử hàm này thread-safe
        if (!get_label_bucket(label, bucket_off) || bucket_off == 0)
            return false;

        // 3. Resolve Pointer
        std::vector<char> temp_buffer;
        auto base_opt = resolve_bucket_base(arena_, bucket_off, temp_buffer);
        if (!base_opt)
            return false;

        const char *ptr = *base_opt;

        // [FIX] Copy Header an toàn (tránh lỗi alignment)
        BucketHeader hdr;
        std::memcpy(&hdr, ptr, sizeof(BucketHeader));

        uint32_t count = hdr.count.load(std::memory_order_relaxed);
        if (count == 0)
            return false;

        // 4. Scan IDs để tìm vị trí (Index)
        // Offset tới mảng IDs
        const char *ids_ptr_raw = ptr + hdr.off_ids;
        int32_t found_idx = -1;

        for (uint32_t i = 0; i < count; ++i)
        {
            uint64_t v;
            // [FIX] Copy ID ra biến local (8 bytes)
            std::memcpy(&v, ids_ptr_raw + i * sizeof(uint64_t), sizeof(uint64_t));

            if (v == label)
            {
                found_idx = static_cast<int32_t>(i);
                break;
            }
        }

        if (found_idx < 0)
            return false;

        // 5. Lấy độ dài dữ liệu (Echo/Vector length)
        const char *lens_ptr_raw = ptr + hdr.off_pq_codes;
        uint16_t len = 0;
        // [FIX] Copy length (2 bytes)
        std::memcpy(&len, lens_ptr_raw + found_idx * sizeof(uint16_t), sizeof(uint16_t));

        if (len == 0 || len > MAX_ECHO_BYTES)
            return false;

        // 6. Lấy dữ liệu Vector
        const char *vec_data_ptr = ptr + hdr.off_vectors + static_cast<size_t>(found_idx) * MAX_ECHO_BYTES;

        // Decode (Giữ nguyên logic decode của bạn, chỉ thay đổi input pointer)
        const uint8_t *ub = reinterpret_cast<const uint8_t *>(vec_data_ptr);

        pomai::ai::EchoCode code;
        size_t pos = 0;
        code.depth = (pos < len) ? ub[pos++] : 0;
        size_t depth = code.depth;

        code.scales_q.resize(depth);
        for (size_t k = 0; k < depth; ++k)
            code.scales_q[k] = (pos < len) ? ub[pos++] : 0;

        code.bits_per_layer.resize(depth);
        code.sign_bytes.resize(depth);
        for (size_t k = 0; k < depth; ++k)
        {
            uint32_t b = cfg_.eeq_cfg.bits_per_layer[k];
            code.bits_per_layer[k] = b;
            size_t bytes = (b + 7) / 8;
            if (pos + bytes <= len)
            {
                code.sign_bytes[k].assign(ub + pos, ub + pos + bytes);
                pos += bytes;
            }
            else
            {
                return false;
            }
        }

        out_vec.assign(cfg_.dim, 0.0f);
        if (eeq_)
        {
            eeq_->decode(code, out_vec.data());
        }
        else
        {
            // Fallback nếu không có Quantizer (Raw float store)
            // Nếu lưu raw float thì len phải == dim * 4
            if (len == cfg_.dim * sizeof(float))
            {
                std::memcpy(out_vec.data(), ub, len);
            }
        }
        return true;
    }

    bool PomaiOrbit::remove(uint64_t label)
    {
        std::unique_lock<std::shared_mutex> dm(del_mu_);
        deleted_labels_.insert(label);
        return true;
    }

    bool PomaiOrbit::compute_distance_for_id_with_proj(const std::vector<std::vector<float>> &qproj, float qnorm2, uint64_t id, float &out_dist)
    {
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.count(id))
                return false;
        }

        uint64_t bucket_off = 0;
        uint32_t cached_slot = UINT32_MAX;
        if (!get_label_bucket(id, bucket_off))
            return false;
        get_label_slot(id, cached_slot);

        if (bucket_off == 0)
            return false;

        std::vector<char> temp_buffer;
        auto base_opt = resolve_bucket_base(arena_, bucket_off, temp_buffer);
        if (!base_opt)
            return false;
        const char *data_base_ptr = *base_opt;

        const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(data_base_ptr);
        uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
        if (count == 0)
            return false;

        const uint64_t *id_base = reinterpret_cast<const uint64_t *>(data_base_ptr + hdr_ptr->off_ids);
        int32_t found = -1;

        if (cached_slot != UINT32_MAX && cached_slot < count)
        {
            uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + cached_slot);
            if (v == id)
                found = static_cast<int32_t>(cached_slot);
            else
            {
                size_t si = label_shard_index(id);
                LabelShard &sh = label_shards_[si];
                std::unique_lock<std::shared_mutex> lm(sh.mu);
                auto sit = sh.slot.find(id);
                if (sit != sh.slot.end() && sit->second == cached_slot)
                    sh.slot.erase(sit);
                cached_slot = UINT32_MAX;
            }
        }

        if (found < 0)
        {
            for (uint32_t i = 0; i < count; ++i)
            {
                uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                if (v == id)
                {
                    found = static_cast<int32_t>(i);
                    size_t si = label_shard_index(id);
                    LabelShard &sh = label_shards_[si];
                    std::unique_lock<std::shared_mutex> lm(sh.mu);
                    sh.slot[id] = static_cast<uint32_t>(i);
                    break;
                }
            }
        }
        if (found < 0)
            return false;

        const uint16_t *len_base = reinterpret_cast<const uint16_t *>(data_base_ptr + hdr_ptr->off_pq_codes);
        uint16_t len = __atomic_load_n(&len_base[found], __ATOMIC_ACQUIRE);
        if (len == 0 || len > MAX_ECHO_BYTES)
            return false;

        const char *slot_ptr = data_base_ptr + hdr_ptr->off_vectors + static_cast<size_t>(found) * MAX_ECHO_BYTES;
        const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

        out_dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);
        return true;
    }

    bool PomaiOrbit::compute_distance_for_id(const float *query, uint64_t id, float &out_dist)
    {
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);
        return compute_distance_for_id_with_proj(qproj, qnorm2, id, out_dist);
    }

    MembranceInfo PomaiOrbit::get_info() const
    {
        MembranceInfo info;
        info.dim = cfg_.dim;
        uint64_t total_labels = 0;
        for (size_t si = 0; si < kLabelShardCount; ++si)
        {
            const LabelShard &sh = label_shards_[si];
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            total_labels += sh.bucket.size();
        }
        {
            std::shared_lock<std::shared_mutex> lk(del_mu_);
            if (total_labels > deleted_labels_.size())
                total_labels -= deleted_labels_.size();
            else
                total_labels = 0;
        }
        info.num_vectors = static_cast<size_t>(total_labels);
        uint64_t bytes = 0;
        try
        {
            std::filesystem::path p(cfg_.data_path);
            if (std::filesystem::exists(p))
            {
                for (auto it = std::filesystem::recursive_directory_iterator(p, std::filesystem::directory_options::skip_permission_denied);
                     it != std::filesystem::recursive_directory_iterator(); ++it)
                {
                    std::error_code ec;
                    if (it->is_regular_file(ec))
                    {
                        uint64_t fsz = static_cast<uint64_t>(it->file_size(ec));
                        if (!ec)
                            bytes += fsz;
                    }
                }
            }
        }
        catch (...)
        {
        }
        info.disk_bytes = bytes;
        return info;
    }

    std::vector<uint64_t> PomaiOrbit::get_centroid_ids(uint32_t cid) const
    {
        if (cid >= centroids_.size())
            return {};
        std::vector<uint64_t> ids;
        const OrbitNode &node = *centroids_[cid];
        uint64_t curr = node.bucket_offset.load(std::memory_order_acquire);
        std::vector<char> temp_buf;

        while (curr != 0)
        {
            auto base_opt = resolve_bucket_base(arena_, curr, temp_buf);
            if (!base_opt)
                break;

            const char *ptr = *base_opt;

            // [FIX] Dùng memcpy để tránh lỗi alignment
            BucketHeader hdr_copy;
            std::memcpy(&hdr_copy, ptr, sizeof(BucketHeader));

            uint32_t count = hdr_copy.count.load(std::memory_order_acquire);
            if (count > 0)
            {
                const char *ids_raw = ptr + hdr_copy.off_ids;
                for (uint32_t i = 0; i < count; ++i)
                {
                    uint64_t val;
                    std::memcpy(&val, ids_raw + i * sizeof(uint64_t), sizeof(uint64_t));
                    ids.push_back(val);
                }
            }
            curr = hdr_copy.next_bucket_offset.load(std::memory_order_acquire);
        }
        return ids;
    }

    std::vector<uint64_t> PomaiOrbit::get_all_labels() const
    {
        std::vector<uint64_t> out;
        out.reserve(1024);
        for (size_t si = 0; si < kLabelShardCount; ++si)
        {
            const LabelShard &sh = label_shards_[si];
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            for (const auto &kv : sh.bucket)
                out.push_back(kv.first);
        }
        return out;
    }

    bool PomaiOrbit::get_vectors_raw(const std::vector<uint64_t> &ids, std::vector<std::string> &outs) const
    {
        outs.clear();
        outs.resize(ids.size());
        if (ids.empty())
            return true;

        // 1. Group IDs by Bucket để tối ưu I/O (tránh resolve pointer nhiều lần)
        struct Task
        {
            size_t out_idx;
            uint64_t id;
        };
        std::unordered_map<uint64_t, std::vector<Task>> bucket_tasks;

        // Cần const_cast vì get_label_bucket không const (nhưng nó chỉ đọc map, nên OK)
        PomaiOrbit *mutable_this = const_cast<PomaiOrbit *>(this);

        for (size_t i = 0; i < ids.size(); ++i)
        {
            uint64_t bucket_off = 0;
            if (mutable_this->get_label_bucket(ids[i], bucket_off) && bucket_off != 0)
            {
                bucket_tasks[bucket_off].push_back({i, ids[i]});
            }
        }

        // 2. Process each bucket
        std::vector<char> temp_buffer;

        for (const auto &kv : bucket_tasks)
        {
            uint64_t bucket_off = kv.first;
            const auto &tasks = kv.second;

            auto base_opt = resolve_bucket_base(arena_, bucket_off, temp_buffer);
            if (!base_opt)
                continue;

            const char *ptr = *base_opt;

            // [FIX] Copy Header an toàn
            BucketHeader hdr;
            std::memcpy(&hdr, ptr, sizeof(BucketHeader));

            uint32_t count = hdr.count.load(std::memory_order_relaxed);
            if (count == 0)
                continue;

            const char *ids_ptr_raw = ptr + hdr.off_ids;
            const char *lens_ptr_raw = ptr + hdr.off_pq_codes;
            const char *vec_base_ptr = ptr + hdr.off_vectors;

            // Xây dựng map tạm: ID -> Index trong bucket
            std::unordered_map<uint64_t, uint32_t> id_to_idx;
            id_to_idx.reserve(count);

            for (uint32_t i = 0; i < count; ++i)
            {
                uint64_t v;
                // [FIX] Copy ID an toàn
                std::memcpy(&v, ids_ptr_raw + i * sizeof(uint64_t), sizeof(uint64_t));
                id_to_idx[v] = i;
            }

            // Lấy dữ liệu cho từng task
            for (const auto &task : tasks)
            {
                auto it = id_to_idx.find(task.id);
                if (it == id_to_idx.end())
                    continue;

                uint32_t idx = it->second;

                // [FIX] Copy Length an toàn
                uint16_t len = 0;
                std::memcpy(&len, lens_ptr_raw + idx * sizeof(uint16_t), sizeof(uint16_t));

                if (len == 0 || len > MAX_ECHO_BYTES)
                    continue;

                const char *slot_ptr = vec_base_ptr + static_cast<size_t>(idx) * MAX_ECHO_BYTES;

                // Copy raw bytes ra output string
                outs[task.out_idx].assign(slot_ptr, len);
            }
        }
        return true;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        // Wrapper gọi search_with_budget với ngân sách mặc định (Vô cực)
        pomai::ai::Budget unlimited;
        unlimited.ops_budget = 100'000'000; // 100M ops ~ vô tận cho 1 query

        return search_with_budget(query, k, unlimited, nprobe);
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered(const float *query, size_t k, const std::vector<uint64_t> &candidates)
    {
        // Wrapper gọi search_filtered_with_budget với ngân sách mặc định
        pomai::ai::Budget unlimited;
        unlimited.ops_budget = 100'000'000;

        return search_filtered_with_budget(query, k, candidates, unlimited);
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_with_budget(
        const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe)
    {
        if (!query || k == 0)
            return {};

        // 1. Budget check
        uint32_t ops_left = budget.ops_budget;
        auto pay_ops = [&](uint32_t cost) -> bool
        {
            if (ops_left < cost)
                return false;
            ops_left -= cost;
            return true;
        };

        if (!pay_ops(kCostSetup))
            return {};
        if (centroids_.empty())
            return {};

        // 2. Projections
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);

        // 3. Bloom Filter for Deleted items
        std::optional<DeletedBloom> bloom;
        std::vector<uint64_t> small_deleted;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256)
            {
                bloom.emplace();
                for (uint64_t v : deleted_labels_)
                    bloom->add(v);
            }
            else if (!deleted_labels_.empty())
            {
                small_deleted.assign(deleted_labels_.begin(), deleted_labels_.end());
            }
        }

        // 4. Routing
        if (nprobe == 0)
        {
            nprobe = std::max(1UL, static_cast<size_t>(cfg_.algo.num_centroids) / 50);
            if (k > 50)
                nprobe *= 2;
        }
        if (!pay_ops(nprobe * kCostRoute))
            return {};

        auto targets = find_routing_centroids(query, nprobe);
        // Thermal Sort: Ưu tiên centroid nóng
        std::sort(targets.begin(), targets.end(), [&](uint32_t a, uint32_t b)
                  { return get_temperature(a) > get_temperature(b); });

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        for (uint32_t cid : targets)
        {
            if (ops_left == 0)
                break;

            // Thermal Gating: Bỏ qua centroid lạnh nếu budget thấp
            uint8_t temp = get_temperature(cid);
            if (temp < 10 && ops_left < 2000)
                continue;
            touch_centroid(cid);

            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (current_off != 0)
            {
                if (ops_left == 0)
                    break;
                if (!pay_ops(kCostBucket))
                {
                    ops_left = 0;
                    break;
                }

                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, current_off, tmp);
                if (!base_opt)
                    break;

                const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(*base_opt);
                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);

                if (count == 0)
                {
                    current_off = next_bucket_offset;
                    continue;
                }

                const char *vec_area = reinterpret_cast<const char *>(hdr_ptr) + hdr_ptr->off_vectors;
                const uint16_t *len_base = reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(hdr_ptr) + hdr_ptr->off_pq_codes);
                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(reinterpret_cast<const char *>(hdr_ptr) + hdr_ptr->off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    if (ops_left == 0)
                        break;
                    if (!pay_ops(kCostItem))
                    {
                        ops_left = 0;
                        break;
                    }

                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);

                    // Filter deleted
                    if (bloom)
                    {
                        if (bloom->maybe_contains(id))
                            continue;
                    }
                    else if (!small_deleted.empty())
                    {
                        bool found = false;
                        for (uint64_t d : small_deleted)
                            if (d == id)
                            {
                                found = true;
                                break;
                            }
                        if (found)
                            continue;
                    }

                    uint16_t len = __atomic_load_n(&len_base[i], __ATOMIC_ACQUIRE);
                    if (len == 0 || len > MAX_ECHO_BYTES)
                        continue;

                    const uint8_t *ub = reinterpret_cast<const uint8_t *>(vec_area + static_cast<size_t>(i) * MAX_ECHO_BYTES);
                    float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);

                    if (topk.size() < k)
                    {
                        topk.push({dist, id});
                    }
                    else if (dist < topk.top().first)
                    {
                        topk.pop();
                        topk.push({dist, id});
                    }
                }
                current_off = next_bucket_offset;
            }
        }

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(topk.size());
        while (!topk.empty())
        {
            out.emplace_back(topk.top().second, topk.top().first);
            topk.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered_with_budget(
        const float *query, size_t k, const std::vector<uint64_t> &candidates, const pomai::ai::Budget &budget)
    {
        if (!query || k == 0 || candidates.empty())
            return {};

        uint32_t ops_left = budget.ops_budget;
        auto pay_ops = [&](uint32_t cost) -> bool
        {
            if (ops_left < cost)
                return false;
            ops_left -= cost;
            return true;
        };

        if (!pay_ops(kCostSetup))
            return {};

        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);

        std::optional<DeletedBloom> bloom;
        std::vector<uint64_t> small_deleted;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256)
            {
                bloom.emplace();
                for (uint64_t v : deleted_labels_)
                    bloom->add(v);
            }
            else if (!deleted_labels_.empty())
            {
                small_deleted.assign(deleted_labels_.begin(), deleted_labels_.end());
            }
        }

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        for (uint64_t id : candidates)
        {
            if (ops_left == 0)
                break;

            // Check deleted
            if (bloom && bloom->maybe_contains(id))
                continue;
            else if (!small_deleted.empty())
            {
                bool found = false;
                for (uint64_t d : small_deleted)
                    if (d == id)
                    {
                        found = true;
                        break;
                    }
                if (found)
                    continue;
            }

            if (!pay_ops(kCostCheck + kCostBucket + kCostItem))
            {
                ops_left = 0;
                break;
            }

            uint64_t bucket_off = 0;
            if (!get_label_bucket(id, bucket_off) || bucket_off == 0)
                continue;

            std::vector<char> tmp;
            auto base_opt = resolve_bucket_base(arena_, bucket_off, tmp);
            if (!base_opt)
                continue;
            const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(*base_opt);

            // Thermal Touch
            touch_centroid(hdr_ptr->centroid_id);

            uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
            if (count == 0)
                continue;

            const uint64_t *id_base = reinterpret_cast<const uint64_t *>(reinterpret_cast<const char *>(hdr_ptr) + hdr_ptr->off_ids);

            // Linear scan in bucket to find slot (since we only have bucket_off from label map)
            int32_t found = -1;
            for (uint32_t j = 0; j < count; ++j)
            {
                uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + j);
                if (v == id)
                {
                    found = static_cast<int32_t>(j);
                    break;
                }
            }
            if (found < 0)
                continue;

            const uint16_t *len_base = reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(hdr_ptr) + hdr_ptr->off_pq_codes);
            uint16_t len = __atomic_load_n(&len_base[found], __ATOMIC_ACQUIRE);
            if (len == 0 || len > MAX_ECHO_BYTES)
                continue;

            const char *vec_area = reinterpret_cast<const char *>(hdr_ptr) + hdr_ptr->off_vectors;
            const uint8_t *ub = reinterpret_cast<const uint8_t *>(vec_area + static_cast<size_t>(found) * MAX_ECHO_BYTES);

            float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);
            if (topk.size() < k)
            {
                topk.push({dist, id});
            }
            else if (dist < topk.top().first)
            {
                topk.pop();
                topk.push({dist, id});
            }
        }

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(topk.size());
        while (!topk.empty())
        {
            out.emplace_back(topk.top().second, topk.top().first);
            topk.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    bool PomaiOrbit::checkpoint()
    {
        // [DEFENSIVE] Bọc Try-Catch để server KHÔNG BAO GIỜ SẬP
        try
        {
            // [Lock] Chặn đứng mọi thao tác Insert mới
            std::unique_lock<std::shared_mutex> cp_lock(checkpoint_mu_);

            // [Lock] Chặn auto-train
            std::lock_guard<std::mutex> lk(train_mu_);

            std::clog << "[Orbit] Checkpointing...\n";

            // 1. Lưu Metadata/Schema (schema contains centroid/bucket layout)
            save_schema();

            // 2. Flush all mapped bucket blobs from arena(s) to stable storage.
            // We iterate all centroids and their bucket chains; for each blob we
            // msync the exact mapped range covering the blob (header + payload).
            // This avoids relying solely on global sync() and reduces unnecessary IO.
            long pg = sysconf(_SC_PAGESIZE);
            size_t page = (pg > 0) ? static_cast<size_t>(pg) : 4096;

            auto align_down = [&](uintptr_t a) -> uintptr_t
            { return a & ~(static_cast<uintptr_t>(page - 1)); };
            auto align_up = [&](uintptr_t a) -> uintptr_t
            { return (a + page - 1) & ~(static_cast<uintptr_t>(page - 1)); };

            // For each centroid, walk its bucket chain and msync each published blob.
            for (const auto &cptr : centroids_)
            {
                if (!cptr)
                    continue;

                uint64_t curr_off = cptr->bucket_offset.load(std::memory_order_acquire);
                while (curr_off != 0)
                {
                    // Try to get in-memory pointer for this bucket (may return nullptr if remote/not mapped)
                    const char *blob_hdr = arena_.blob_ptr_from_offset_for_map(curr_off);
                    if (!blob_hdr)
                    {
                        // Can't access this bucket in RAM right now (might be remote / demoted),
                        // skip (persisted to remote file already or not mapped).
                        break;
                    }

                    // blob_hdr points to the 4-byte length header (uint32_t)
                    // Calculate total bytes stored in this blob (header + payload)
                    uint32_t stored_len = 0;
                    // Safe read of the stored_len (we're in checkpoint, single-writer synchronization held)
                    std::memcpy(&stored_len, blob_hdr, sizeof(stored_len));
                    size_t total_bytes = static_cast<size_t>(sizeof(uint32_t)) + static_cast<size_t>(stored_len);

                    // Page-align region and msync
                    uintptr_t start_addr = reinterpret_cast<uintptr_t>(blob_hdr);
                    uintptr_t page_start = align_down(start_addr);
                    uintptr_t page_end = align_up(start_addr + total_bytes);
                    size_t msync_len = (page_end > page_start) ? static_cast<size_t>(page_end - page_start) : 0;

                    if (msync_len > 0)
                    {
                        int rc = ::msync(reinterpret_cast<void *>(page_start), msync_len, MS_SYNC);
                        if (rc != 0)
                        {
                            std::cerr << "[Orbit] Warning: msync failed for bucket at off=" << curr_off
                                      << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
                        }
                    }

                    // Advance to next bucket
                    // resolve_bucket_base in other code returns (ram_blob_ptr + sizeof(uint32_t)),
                    // so here bucket header struct starts at (blob_hdr + sizeof(uint32_t))
                    const BucketHeader *hdr = reinterpret_cast<const BucketHeader *>(blob_hdr + sizeof(uint32_t));
                    curr_off = hdr->next_bucket_offset.load(std::memory_order_acquire);
                }
            }

            // 3. Ensure WAL is safely on disk (fsync) before truncating it.
            // This makes truncation safe: either all data is on disk or WAL still contains missing ops.
            if (wal_)
            {
                if (!wal_->fsync_log())
                {
                    std::cerr << "[Orbit] Checkpoint Warning: wal->fsync_log() failed. Aborting truncate.\n";
                    return false;
                }
            }

            // 4. (Defensive global sync) issue a global sync to ensure file system metadata
            // has been flushed; this is slower but gives extra durability on exotic filesystems.
            // Keep this call optional on platforms where it's appropriate.
            ::sync();

            // 5. Truncate WAL (An toàn tuyệt đối vì đĩa đã có dữ liệu)
            if (wal_)
            {
                if (!wal_->truncate_to_zero())
                {
                    std::cerr << "[Orbit] Checkpoint Error: Could not truncate WAL.\n";
                    return false;
                }
            }

            std::clog << "[Orbit] Checkpoint completed successfully.\n";
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Orbit] CRITICAL: Checkpoint exception: " << e.what() << "\n";
            return false;
        }
        catch (...)
        {
            std::cerr << "[Orbit] CRITICAL: Checkpoint crashed with unknown error.\n";
            return false;
        }
    }

} // namespace pomai::ai::orbit