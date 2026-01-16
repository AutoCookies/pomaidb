/*
 * src/ai/pomai_orbit.cc
 *
 * PomaiOrbit implementation (cleaned, clarified, and safe).
 *
 * Responsibilities:
 *  - insert / insert_batch
 *  - get / remove
 *  - search / search_with_budget / search_filtered_with_budget
 *  - helpers: compute_distance_for_id_with_proj, compute_distance_for_id
 *  - introspection: MembranceInfo + get_info() (best-effort)
 *
 * Goals:
 *  - Clear structure and comments.
 *  - Conservative, thread-safe access to shared structures.
 *  - Best-effort, I/O-safe disk accounting for the membrance data directory.
 *
 * Note: This translation keeps previous behavior but cleans up structure and error
 * handling. Performance-critical loops left intact; micro-optimizations should
 * be applied in targeted hot paths with profiling data.
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

namespace pomai::ai::orbit
{

    // Constants
    static constexpr size_t MAX_ECHO_BYTES = 64;
    static constexpr uint32_t RESERVE_CHUNK = 16; // per-thread reservation
    static constexpr size_t DEFAULT_MAX_SUBBATCH = 4096;

    static constexpr uint32_t kCostSetup  = 200; // Matrix proj + norm
    static constexpr uint32_t kCostRoute  = 50;  // Per probe routing step
    static constexpr uint32_t kCostBucket = 20;  // Bucket lookup + IO overhead check
    static constexpr uint32_t kCostItem   = 5;   // Decode + Distance calc (AVX optimized)
    static constexpr uint32_t kCostCheck  = 2;   // Filter check (Bloom/Set)

    struct ThreadReserve
    {
        uint32_t base = 0;
        uint32_t remain = 0;
        uint64_t bucket_off = 0;
    };

    // Helper: thread-local reconstruction buffer
    static inline float *thread_local_recon(size_t dim)
    {
        static thread_local std::vector<float> recon_buf;
        if (recon_buf.size() < dim)
            recon_buf.resize(dim);
        return recon_buf.data();
    }

    // Resolve a bucket offset to a stable pointer that points at the BucketHeader.
    // If bucket is remote (demoted) and arena returns nullptr, attempt to read into a temporary
    // buffer (read_remote_blob) and return pointer to temp memory + sizeof(uint32_t).
    // Returns nullopt on failure.
    static std::optional<const char *> resolve_bucket_base(const ArenaView &arena, uint64_t bucket_off, std::vector<char> &temp_buffer)
    {
        if (bucket_off == 0)
            return std::nullopt;

        const char *ram_blob_ptr = arena.blob_ptr_from_offset_for_map(bucket_off);
        if (ram_blob_ptr)
            return ram_blob_ptr + sizeof(uint32_t);

        // fallback: read remote blob into temp buffer (one-shot)
        temp_buffer = arena.read_remote_blob(bucket_off);
        if (temp_buffer.empty())
            return std::nullopt;
        return temp_buffer.data() + sizeof(uint32_t);
    }

    // ----------------- ArenaView dispatch impl (thin wrappers) -----------------
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

    // ------------------- Constructors / Destructors -------------------
    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
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
                cortex_ = std::make_unique<NetworkCortex>(7777);
                cortex_->start();
            }
            catch (...)
            {
                // non-fatal
            }
        }

        if (!centroids_.empty())
        {
            init_thermal_map(centroids_.size());
        }
    }

    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::ShardArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
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
                cortex_ = std::make_unique<NetworkCortex>(7777);
                cortex_->start();
            }
            catch (...)
            {
                // non-fatal
            }
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

    // ------------------- Thermal Thermodynamics Core -------------------

    void PomaiOrbit::init_thermal_map(size_t num_centroids)
    {
        if (num_centroids == 0)
            return;
        // Resize atomic vectors (lưu ý: atomic không copy được, phải resize rồi init từng cái hoặc loop)
        // std::vector::resize mặc định sẽ gọi default constructor của atomic (value 0).
        if (thermal_map_.size() != num_centroids)
        {
            thermal_map_ = std::vector<std::atomic<uint8_t>>(num_centroids);
            last_access_epoch_ = std::vector<std::atomic<uint32_t>>(num_centroids);
        }

        uint32_t now = static_cast<uint32_t>(std::time(nullptr));
        for (size_t i = 0; i < num_centroids; ++i)
        {
            // Mặc định "Warm" (100) để không bị evicted ngay khi khởi động
            thermal_map_[i].store(100, std::memory_order_relaxed);
            last_access_epoch_[i].store(now, std::memory_order_relaxed);
        }
    }

    void PomaiOrbit::touch_centroid(uint32_t cid)
    {
        if (cid >= thermal_map_.size())
            return;

        // 1. Heat up
        uint8_t old_temp = thermal_map_[cid].load(std::memory_order_relaxed);
        if (old_temp < 255)
        {
            // Boost mạnh nếu đang lạnh để "rã đông", boost nhẹ nếu đang nóng
            uint8_t boost = (old_temp < 50) ? 50 : 5;
            int new_val = old_temp + boost;
            thermal_map_[cid].store((new_val > 255) ? 255 : static_cast<uint8_t>(new_val), std::memory_order_relaxed);
        }

        // 2. Mark timestamp
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

        // Lazy Decay: Tính toán độ nguội dựa trên thời gian trôi qua
        uint32_t last = last_access_epoch_[cid].load(std::memory_order_relaxed);
        uint32_t now = static_cast<uint32_t>(std::time(nullptr));

        if (now > last)
        {
            uint32_t delta = now - last;
            // Config: 60s giảm 10 độ
            uint32_t cooldown = (delta / 60) * 10;
            if (cooldown >= temp)
                return 0;
            return temp - cooldown;
        }
        return temp;
    }

    void PomaiOrbit::apply_thermal_policy()
    {
        // Chỉ áp dụng với ShardArena (Disk-backed)
        if (!arena_.is_shard_arena())
            return;

        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            uint8_t temp = get_temperature(static_cast<uint32_t>(i));

            // Policy: Quá lạnh (< 5) -> Đá ra khỏi RAM (Free Memory)
            if (temp < 5)
            {
                uint64_t off = centroids_[i]->bucket_offset.load(std::memory_order_relaxed);
                // Demote bucket header & content.
                // Size hint 4KB hoặc dynamic_bucket_capacity * size
                if (off != 0)
                {
                    // Ước lượng size bucket (đơn giản hoá 4KB page)
                    arena_.demote_range(off, 4096);
                }
            }
        }
    }

    // ------------------- Persistence (schema) -------------------
    void PomaiOrbit::save_schema()
    {
        SchemaHeader header;
        header.dim = cfg_.dim;
        header.num_centroids = cfg_.num_centroids;

        // compute approximate total_vectors from label shards (exclude deleted labels)
        {
            uint64_t total = 0;
            for (size_t si = 0; si < kLabelShardCount; ++si)
            {
                const LabelShard &sh = label_shards_[si];
                std::shared_lock<std::shared_mutex> lk(sh.mu);
                total += sh.bucket.size();
            }
            {
                std::shared_lock<std::shared_mutex> lk(del_mu_);
                if (total > deleted_labels_.size())
                    total -= deleted_labels_.size();
                else
                    total = 0;
            }
            header.total_vectors = static_cast<uint64_t>(total);
        }

        std::ofstream out(schema_file_path_, std::ios::binary);
        if (out.is_open())
        {
            out.write(reinterpret_cast<const char *>(&header), sizeof(header));
            out.close();
        }
        else
        {
            std::clog << "[Orbit] ERROR: Could not save schema to " << schema_file_path_ << "\n";
        }
    }

    bool PomaiOrbit::load_schema()
    {
        std::ifstream in(schema_file_path_, std::ios::binary);
        if (!in.is_open())
            return false;

        SchemaHeader header;
        in.read(reinterpret_cast<char *>(&header), sizeof(header));
        in.close();

        if (header.magic_number != 0x504F4D41)
        {
            std::clog << "[Orbit] Invalid schema file signature\n";
            return false;
        }

        cfg_.dim = header.dim;
        if (header.num_centroids > 0)
            cfg_.num_centroids = header.num_centroids;
        return true;
    }

    // ------------------- Train / routing / alloc -------------------
    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0)
            return false;

        // 1. Autopilot: Calculate optimal number of centroids if not set
        if (cfg_.num_centroids == 0)
        {
            cfg_.num_centroids = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
            cfg_.num_centroids = std::clamp(cfg_.num_centroids, static_cast<size_t>(64), static_cast<size_t>(4096));
        }

        // 2. Autopilot: Calculate dynamic bucket capacity based on density
        size_t avg_density = (n / cfg_.num_centroids) + 1;
        dynamic_bucket_capacity_ = static_cast<uint32_t>(avg_density * 1.5);
        dynamic_bucket_capacity_ = std::clamp(dynamic_bucket_capacity_, static_cast<uint32_t>(32), static_cast<uint32_t>(512));

        std::clog << "[Orbit Autopilot] Training Config: Centroids=" << cfg_.num_centroids
                  << ", BucketCap=" << dynamic_bucket_capacity_
                  << " (N=" << n << ")\n";

        // 3. Initialize Centroids (Skeleton)
        size_t num_c = std::min(n, cfg_.num_centroids);
        centroids_.resize(num_c);
        for (size_t i = 0; i < num_c; ++i)
            centroids_[i] = std::make_unique<OrbitNode>();

        // 4. KMeans++ Initialization (Simplified: Random Shuffle)
        std::mt19937 rng(42);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(num_c);

        for (size_t i = 0; i < num_c; ++i)
        {
            centroids_[i]->vector.resize(cfg_.dim);
            std::memcpy(centroids_[i]->vector.data(), data + indices[i] * cfg_.dim, cfg_.dim * sizeof(float));
            centroids_[i]->neighbors.reserve(cfg_.m_neighbors * 2);
        }

        // 5. Build Routing Graph (HNSW/NGT style neighbors)
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
                for (size_t k = 0; k < std::min(dists.size(), cfg_.m_neighbors); ++k)
                    centroids_[i]->neighbors.push_back(dists[k].second);
            }
        }

        // 6. Allocate Initial Buckets (Flesh)
        for (size_t i = 0; i < num_c; ++i)
        {
            uint64_t off = alloc_new_bucket(static_cast<uint32_t>(i));
            centroids_[i]->bucket_offset.store(off, std::memory_order_release);
        }

        // 7. Persist Schema
        save_schema();

        // 8. [THERMAL ACTIVATION] Initialize the thermal map for the new centroids.
        // This sets their initial temperature to "Warm" (100) so they are ready for action.
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

    // ------------------- Label map shard helpers -------------------
    void PomaiOrbit::set_label_map(uint64_t label, uint64_t bucket_off, uint32_t slot)
    {
        size_t si = label_shard_index(label);
        LabelShard &sh = label_shards_[si];

        // Dùng unique_lock để ghi an toàn
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

    // ------------------- insert / get / remove -------------------
    bool PomaiOrbit::insert(const float *vec, uint64_t label)
    {
        if (!eeq_ || centroids_.empty())
            return false;

        static thread_local ThreadReserve tres;

        uint32_t cid = find_nearest_centroid(vec);
        OrbitNode &node = *centroids_[cid];

        uint64_t current_off = node.bucket_offset.load(std::memory_order_acquire);
        if (current_off == 0)
        {
            std::unique_lock<std::shared_mutex> lock(node.mu);
            if ((current_off = node.bucket_offset.load(std::memory_order_acquire)) == 0)
            {
                current_off = alloc_new_bucket(cid);
                if (current_off == 0)
                    return false;
                node.bucket_offset.store(current_off, std::memory_order_release);
            }
        }

        pomai::ai::EchoCode code;
        try
        {
            code = eeq_->encode(vec);
        }
        catch (...)
        {
            return false;
        }

        while (true)
        {
            const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
            if (!ram_blob_ptr)
            {
                std::unique_lock<std::shared_mutex> lock(node.mu);
                current_off = node.bucket_offset.load(std::memory_order_acquire);
                if (current_off == 0)
                    return false;
                ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!ram_blob_ptr)
                    return false;
                tres.remain = 0;
                tres.bucket_off = 0;
            }

            char *bucket_ptr = const_cast<char *>(ram_blob_ptr) + sizeof(uint32_t);
            BucketHeader *hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);
            uint32_t cap = dynamic_bucket_capacity_;

            if (tres.remain == 0 || tres.bucket_off != current_off)
            {
                uint32_t chunk = RESERVE_CHUNK;
                uint32_t slot_base = hdr->count.fetch_add(chunk, std::memory_order_acq_rel);

                if (slot_base >= cap)
                {
                    hdr->count.fetch_sub(chunk, std::memory_order_acq_rel);
                    std::unique_lock<std::shared_mutex> lock(node.mu);
                    const char *blob_ptr2 = arena_.blob_ptr_from_offset_for_map(current_off);
                    if (!blob_ptr2)
                        return false;
                    char *bucket_ptr2 = const_cast<char *>(blob_ptr2) + sizeof(uint32_t);
                    BucketHeader *hdr2 = reinterpret_cast<BucketHeader *>(bucket_ptr2);

                    uint64_t nb = hdr2->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(cid);
                        if (nb == 0)
                        {
                            std::clog << "[Orbit] insert: alloc_new_bucket failed\n";
                            return false;
                        }
                        hdr2->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    tres.remain = 0;
                    tres.bucket_off = 0;
                    continue;
                }

                uint32_t usable = chunk;
                if (slot_base + chunk > cap)
                {
                    usable = (slot_base < cap) ? (cap - slot_base) : 0;
                    if (usable < chunk)
                        hdr->count.fetch_sub(chunk - usable, std::memory_order_acq_rel);
                }
                if (usable == 0)
                {
                    std::unique_lock<std::shared_mutex> lock(node.mu);
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(cid);
                        if (nb == 0)
                            return false;
                        hdr->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    tres.remain = 0;
                    tres.bucket_off = 0;
                    continue;
                }
                tres.base = slot_base;
                tres.remain = usable;
                tres.bucket_off = current_off;
            }

            uint32_t idx = tres.base + (RESERVE_CHUNK - tres.remain);
            tres.remain--;
            if (idx >= dynamic_bucket_capacity_)
            {
                tres.remain = 0;
                continue;
            }

            uint8_t tmpbuf[MAX_ECHO_BYTES];
            size_t pos = 0;
            tmpbuf[pos++] = code.depth;
            for (uint8_t q : code.scales_q)
                if (pos < MAX_ECHO_BYTES)
                    tmpbuf[pos++] = q;

            bool overflow = false;
            for (const auto &sb : code.sign_bytes)
            {
                if (pos + sb.size() > MAX_ECHO_BYTES)
                {
                    overflow = true;
                    break;
                }
                std::memcpy(tmpbuf + pos, sb.data(), sb.size());
                pos += sb.size();
            }
            if (overflow)
            {
                hdr->count.fetch_sub(1, std::memory_order_acq_rel);
                tres.remain = 0;
                return false;
            }

            char *vec_area = bucket_ptr + hdr->off_vectors;
            char *slot_ptr = vec_area + static_cast<size_t>(idx) * MAX_ECHO_BYTES;
            std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
            std::memcpy(slot_ptr, tmpbuf, pos);

            uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);
            pomai::ai::atomic_utils::atomic_store_u64(id_base + idx, label);

            uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
            __atomic_store_n(&len_base[idx], static_cast<uint16_t>(pos), __ATOMIC_RELEASE);

            set_label_map(label, current_off, idx);
            return true;
        }

        return false;
    }

    bool PomaiOrbit::get(uint64_t label, std::vector<float> &out_vec)
    {
        std::shared_lock<std::shared_mutex> dm(del_mu_);
        if (deleted_labels_.count(label))
            return false;

        uint64_t bucket_off = 0;
        if (!get_label_bucket(label, bucket_off))
            return false;
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
        for (uint32_t i = 0; i < count; ++i)
        {
            uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
            if (v == label)
            {
                found = static_cast<int32_t>(i);
                break;
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

        pomai::ai::EchoCode code;
        size_t pos = 0;
        code.depth = (pos < len) ? ub[pos++] : 0;
        size_t depth = code.depth;
        code.scales_q.resize(depth);
        for (size_t k = 0; k < depth; ++k)
        {
            code.scales_q[k] = (pos < len) ? ub[pos++] : 0;
        }
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
                return false;
        }

        out_vec.assign(cfg_.dim, 0.0f);
        eeq_->decode(code, out_vec.data());
        return true;
    }

    bool PomaiOrbit::remove(uint64_t label)
    {
        std::unique_lock<std::shared_mutex> dm(del_mu_);
        deleted_labels_.insert(label);
        return true;
    }

    // ------------------- compute_distance helpers -------------------
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
        get_label_slot(id, cached_slot); // optional

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

    // ------------------- search / budget-aware search -------------------
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_with_budget(
        const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe)
    {
        if (!query || k == 0) return {};

        // 1. Setup Budget Tracker
        uint32_t ops_left = budget.ops_budget;
        auto pay_ops = [&](uint32_t cost) -> bool {
            if (ops_left < cost) return false;
            ops_left -= cost;
            return true;
        };

        // [COST] Trừ phí khởi tạo (Setup Cost)
        if (!pay_ops(kCostSetup)) return {};
        
        if (centroids_.empty()) return {};

        // 2. Prepare Query Projection
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);

        // 3. Prepare Deleted Filter
        std::optional<DeletedBloom> bloom;
        std::vector<uint64_t> small_deleted;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256) {
                bloom.emplace();
                for (uint64_t v : deleted_labels_) bloom->add(v);
            } else if (!deleted_labels_.empty()) {
                small_deleted.assign(deleted_labels_.begin(), deleted_labels_.end());
            }
        }

        // 4. Routing
        if (nprobe == 0) {
            nprobe = std::max(1UL, cfg_.num_centroids / 50);
            if (k > 50) nprobe *= 2;
        }

        // [COST] Trừ phí tìm đường (Routing Cost)
        // Ước lượng: số bước probe * chi phí mỗi bước
        if (!pay_ops(nprobe * kCostRoute)) return {}; 

        auto targets = find_routing_centroids(query, nprobe);

        // Sorting by temperature (Thermal Optimization)
        std::sort(targets.begin(), targets.end(), [&](uint32_t a, uint32_t b) {
            return get_temperature(a) > get_temperature(b);
        });

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        for (uint32_t cid : targets)
        {
            // Kiểm tra budget vòng ngoài
            if (ops_left == 0) break;

            // Thermal Gating
            uint8_t temp = get_temperature(cid);
            if (temp < 10 && ops_left < 2000) continue; // Skip cold data if low budget

            touch_centroid(cid);

            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (current_off != 0)
            {
                if (ops_left == 0) break;

                // [COST] Trừ phí truy cập Bucket (Bucket Access Cost)
                // Nếu không đủ tiền mở bucket -> Dừng ngay
                if (!pay_ops(kCostBucket)) { ops_left = 0; break; }

                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, current_off, tmp);
                if (!base_opt) break;

                const char *bucket_base = *base_opt;
                const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(bucket_base);
                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);

                if (count == 0) {
                    current_off = next_bucket_offset;
                    continue;
                }

                const uint16_t *len_base = reinterpret_cast<const uint16_t *>(bucket_base + hdr_ptr->off_pq_codes);
                const char *vec_area = bucket_base + hdr_ptr->off_vectors;
                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(bucket_base + hdr_ptr->off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    if (ops_left == 0) break;

                    // [COST] Trừ phí xử lý Item (Item Cost)
                    // Đây là vòng lặp tốn kém nhất, cần check chặt chẽ
                    if (!pay_ops(kCostItem)) { ops_left = 0; break; }

                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);

                    // Check filter cost is implicitly covered by kCostItem/kCostCheck overhead
                    if (bloom) {
                        if (bloom->maybe_contains(id)) continue;
                    } else if (!small_deleted.empty()) {
                        bool found = false;
                        for (uint64_t d : small_deleted) if (d == id) { found = true; break; }
                        if (found) continue;
                    }

                    uint16_t len = __atomic_load_n(&len_base[i], __ATOMIC_ACQUIRE);
                    if (len == 0 || len > MAX_ECHO_BYTES) continue;

                    const char *slot_ptr = vec_area + static_cast<size_t>(i) * MAX_ECHO_BYTES;
                    const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

                    float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);

                    if (topk.size() < k) {
                        topk.push({dist, id});
                    } else if (dist < topk.top().first) {
                        topk.pop();
                        topk.push({dist, id});
                    }
                }
                current_off = next_bucket_offset;
            }
        }

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(topk.size());
        while (!topk.empty()) {
            out.emplace_back(topk.top().second, topk.top().first);
            topk.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered_with_budget(
        const float *query, size_t k, const std::vector<uint64_t> &candidates, const pomai::ai::Budget &budget)
    {
        if (!query || k == 0 || candidates.empty()) return {};

        uint32_t ops_left = budget.ops_budget;
        auto pay_ops = [&](uint32_t cost) -> bool {
            if (ops_left < cost) return false;
            ops_left -= cost;
            return true;
        };

        // [COST] Trừ phí Setup
        if (!pay_ops(kCostSetup)) return {};

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);

        std::optional<DeletedBloom> bloom;
        std::vector<uint64_t> small_deleted;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256) {
                bloom.emplace();
                for (uint64_t v : deleted_labels_) bloom->add(v);
            } else if (!deleted_labels_.empty()) {
                small_deleted.assign(deleted_labels_.begin(), deleted_labels_.end());
            }
        }

        for (uint64_t id : candidates)
        {
            if (ops_left == 0) break;

            if (bloom && bloom->maybe_contains(id)) continue;
            else if (!small_deleted.empty()) {
                bool found = false;
                for (uint64_t d : small_deleted) if (d == id) { found = true; break; }
                if (found) continue;
            }

            // [COST] Trừ phí Truy cập ngẫu nhiên (Random Access Cost)
            // Filtered search nhảy cóc (random access) nên tốn kém hơn search thường.
            // Ta tính gộp: Check + Bucket + Decode
            if (!pay_ops(kCostCheck + kCostBucket + kCostItem)) { ops_left = 0; break; }

            uint64_t bucket_off = 0;
            if (!get_label_bucket(id, bucket_off) || bucket_off == 0) continue;

            std::vector<char> tmp;
            auto base_opt = resolve_bucket_base(arena_, bucket_off, tmp);
            if (!base_opt) continue;
            const char *data_base_ptr = *base_opt;
            
            const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(data_base_ptr);
            // Thermal logic: Keep filtered items hot
            touch_centroid(hdr_ptr->centroid_id);

            uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
            if (count == 0) continue;

            const uint64_t *id_base = reinterpret_cast<const uint64_t *>(data_base_ptr + hdr_ptr->off_ids);
            int32_t found = -1;
            for (uint32_t j = 0; j < count; ++j) {
                uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + j);
                if (v == id) { found = static_cast<int32_t>(j); break; }
            }
            if (found < 0) continue;

            const uint16_t *len_base = reinterpret_cast<const uint16_t *>(data_base_ptr + hdr_ptr->off_pq_codes);
            uint16_t len = __atomic_load_n(&len_base[found], __ATOMIC_ACQUIRE);
            if (len == 0 || len > MAX_ECHO_BYTES) continue;

            const char *vec_area = data_base_ptr + hdr_ptr->off_vectors;
            const char *slot_ptr = vec_area + static_cast<size_t>(found) * MAX_ECHO_BYTES;
            const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

            float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);

            if (topk.size() < k) {
                topk.push({dist, id});
            } else if (dist < topk.top().first) {
                topk.pop();
                topk.push({dist, id});
            }
        }

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(topk.size());
        while (!topk.empty()) {
            out.emplace_back(topk.top().second, topk.top().first);
            topk.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    // ------------------- insert_batch (with Auto-Train) -------------------
    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        // 1. Kiểm tra Quantizer (bắt buộc phải có)
        if (!eeq_)
            return false;

        // ---------------------------------------------------------
        // [NEW] AUTO-TRAIN LOGIC
        // Nếu DB chưa có centroids (chưa train), dùng batch này để train ngay lập tức.
        // ---------------------------------------------------------
        if (centroids_.empty()) // Check 1: Optimistic (Không lock để nhanh)
        {
            // Lock lại để chỉ 1 thằng được vào train
            std::lock_guard<std::mutex> lk(train_mu_);

            if (centroids_.empty()) // Check 2: Authoritative (Sau khi lock)
            {
                if (batch.empty())
                    return false;

                std::clog << "[Orbit] Database is empty. Auto-training centroids using first batch (" << batch.size() << " vectors)...\n";

                // Gom dữ liệu từ batch ra mảng liền kề
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
                    size_t original_k = cfg_.num_centroids;
                    if (original_k == 0 || original_k > valid_count / 2)
                    {
                        size_t suggest = static_cast<size_t>(std::sqrt(valid_count));
                        const_cast<Config &>(cfg_).num_centroids = std::max<size_t>(1, suggest);
                        std::clog << "[Orbit] Auto-adjusted num_centroids to " << const_cast<Config &>(cfg_).num_centroids << " based on batch size.\n";
                    }

                    this->train(training_data.data(), valid_count);
                }
            }
        }
        // ---------------------------------------------------------

        // 2. Kiểm tra lại: Nếu train thất bại hoặc vẫn rỗng thì dừng
        if (centroids_.empty() || batch.empty())
            return false;

        // 3. Chuẩn bị tham số Batch Insert
        size_t max_subbatch = DEFAULT_MAX_SUBBATCH;
        const char *env = std::getenv("POMAI_MAX_BATCH_INSERT");
        if (env)
        {
            try
            {
                max_subbatch = static_cast<size_t>(std::stoul(env));
            }
            catch (...)
            { /* ignore */
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

        // 4. Encode toàn bộ vectors trong batch (Parallelizable nếu cần)
        for (const auto &p : batch)
        {
            uint64_t label = p.first;
            const std::vector<float> &vec = p.second;
            if (vec.size() != cfg_.dim)
                continue;

            // Tìm cụm gần nhất (Routing)
            uint32_t cid = find_nearest_centroid(vec.data());

            // Nén vector (Quantization)
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
            // Pack metadata & scales
            it.bytes[pos++] = code.depth;
            for (uint8_t q : code.scales_q)
                if (pos < MAX_ECHO_BYTES)
                    it.bytes[pos++] = q;

            // Pack sign bits
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

        // 5. Sắp xếp theo Centroid ID để gom nhóm ghi (giảm lock contention)
        std::sort(prepared.begin(), prepared.end(), [](auto &a, auto &b)
                  { return a.cid < b.cid; });

        // 6. Ghi vào buckets
        size_t idx = 0, total = prepared.size();
        while (idx < total)
        {
            // size_t start = idx;
            uint32_t cid = prepared[idx].cid;

            // Gom nhóm các item cùng CID (tối đa max_subbatch)
            size_t group_end = idx;
            while (group_end < total && prepared[group_end].cid == cid && (group_end - idx) < max_subbatch)
                ++group_end;

            OrbitNode &node = *centroids_[cid];
            std::unique_lock<std::shared_mutex> lock(node.mu); // LOCK CENTROID

            // Lấy bucket hiện tại hoặc tạo mới
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
                // Resolve pointer tới vùng nhớ bucket
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, current_off, tmp);

                // Nếu bucket lỗi (ví dụ file remote không đọc được), tạo bucket mới
                if (!base_opt)
                {
                    uint64_t nb = alloc_new_bucket(cid);
                    if (nb == 0)
                        return false;

                    // Cố gắng link từ bucket cũ (nếu nó còn trong RAM map)
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

                // Nếu bucket đầy, nhảy sang bucket kế tiếp
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

                // Tính số lượng ghi được vào bucket này
                uint32_t remaining = dynamic_bucket_capacity_ - cur_count;
                uint32_t fit = 0;
                size_t probe = write_idx;
                while (probe < group_end && fit < remaining)
                {
                    ++fit;
                    ++probe;
                }

                // Race condition check: nếu trong lúc tính toán bucket bị đầy bởi thread khác
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

                // Reserve chỗ (Atomic fetch_add)
                uint32_t slot_base = hdr->count.fetch_add(fit, std::memory_order_acq_rel);

                // Double check overflow sau khi fetch_add
                if (slot_base + fit > dynamic_bucket_capacity_)
                {
                    // Rollback (để đơn giản ta không rollback count mà chỉ chuyển bucket mới, count dư kệ nó)
                    // Hoặc chính xác hơn:
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

                // Ghi dữ liệu
                char *vec_area = bucket_ptr + hdr->off_vectors;
                uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
                uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);

                for (uint32_t j = 0; j < fit; ++j)
                {
                    const Item &it = prepared[write_idx + j];
                    uint32_t slot = slot_base + j;

                    // Copy Compressed Vector
                    char *slot_ptr = vec_area + static_cast<size_t>(slot) * MAX_ECHO_BYTES;
                    std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
                    std::memcpy(slot_ptr, it.bytes, it.size);

                    // Store ID
                    pomai::ai::atomic_utils::atomic_store_u64(id_base + slot, it.label);

                    // Store Length (Commit write)
                    __atomic_store_n(&len_base[slot], static_cast<uint16_t>(it.size), __ATOMIC_RELEASE);

                    // Update Label Map (Inverted Index for Get/Delete)
                    size_t si = label_shard_index(it.label);
                    LabelShard &sh = label_shards_[si];
                    std::unique_lock<std::shared_mutex> lm(sh.mu);
                    sh.bucket[it.label] = current_off;
                    sh.slot[it.label] = slot;
                }

                write_idx += fit;
            }

            idx = group_end;
            // Nhường CPU một chút để các luồng Search chen vào
            std::this_thread::yield();
            PomaiMetrics::batch_subbatches_processed.fetch_add(1, std::memory_order_relaxed);
        }

        return true;
    }

    // ------------------- Membrance introspection (best-effort snapshot) -------------------
    MembranceInfo PomaiOrbit::get_info() const
    {
        MembranceInfo info;
        info.dim = cfg_.dim;

        // 1) Count labels present in in-memory label shards
        uint64_t total_labels = 0;
        for (size_t si = 0; si < kLabelShardCount; ++si)
        {
            const LabelShard &sh = label_shards_[si];
            std::shared_lock<std::shared_mutex> lk(sh.mu);
            total_labels += sh.bucket.size();
        }

        // 2) exclude soft-deleted labels
        {
            std::shared_lock<std::shared_mutex> lk(del_mu_);
            if (total_labels > deleted_labels_.size())
                total_labels -= deleted_labels_.size();
            else
                total_labels = 0;
        }
        info.num_vectors = static_cast<size_t>(total_labels);

        // 3) Compute on-disk footprint for membrance directory (best-effort)
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
        { /* best-effort: swallow errors */
        }

        info.disk_bytes = bytes;
        return info;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        // Search thường = Search với ngân sách "Vô cực" (100 triệu ops)
        // Lợi ích: Vẫn được hưởng Thermal Sorting (ưu tiên data nóng) và Thermal Touch.
        pomai::ai::Budget unlimited;
        unlimited.ops_budget = 100'000'000;

        return search_with_budget(query, k, unlimited, nprobe);
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered(const float *query, size_t k, const std::vector<uint64_t> &candidates)
    {
        // Filter search thường = Filter search với ngân sách "Vô cực"
        // Quan trọng: Nó sẽ kích hoạt touch_centroid() bên dưới để giữ data nóng.
        pomai::ai::Budget unlimited;
        unlimited.ops_budget = 100'000'000;

        return search_filtered_with_budget(query, k, candidates, unlimited);
    }

} // namespace pomai::ai::orbit