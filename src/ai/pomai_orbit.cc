/*
 * src/ai/pomai_orbit.cc
 *
 * PomaiOrbit implementation with pragmatic performance upgrades:
 *  - Deleted-label snapshot -> lightweight Bloom snapshot for hot loops
 *  - Safer & faster publish for inserts: reserve slots via fetch_add, write
 *    data+id, then publish `len` with release semantics so readers skip
 *    unwritten slots. This avoids readers seeing partially-written vectors.
 *  - insert_batch uses reservation (fetch_add) to reduce atomics and short-lock
 *    window; pre-encoding done before locking. Also group label_map updates
 *    to a single lock per bucket.
 *  - insert() uses per-thread chunk reservation to amortize atomic fetch_add.
 *
 * Notes:
 *  - This avoids ABI changes, keeps on-disk layout, and aims for safety (readers
 *    only process slots where len != 0). It purposely avoids adding new on-disk
 *    header fields.
 *  - Requires that blob allocations initialize len array to 0 (kept in alloc_new_bucket).
 */

#include "src/ai/pomai_orbit.h"

#include "src/ai/atomic_utils.h"
#include "src/memory/arena.h"
#include "src/memory/shard_arena.h"
#include "src/ai/eternalecho_quantizer.h"
#include "src/ai/whispergrain.h"
#include "src/core/cpu_kernels.h"

#include <new>
#include <algorithm>
#include <random>
#include <limits>
#include <cstring>
#include <iostream>
#include <queue>
#include <numeric>
#include <stdexcept>
#include <cstddef> // offsetof
#include <thread>  // thread_local
#include <optional>
#include <chrono>
#include <vector>
#include <type_traits>

namespace pomai::ai::orbit
{

    static constexpr size_t MAX_ECHO_BYTES = 64;

    // Per-thread reservation chunk for insert()
    static constexpr uint32_t RESERVE_CHUNK = 16; // reserve 16 slots at a time

    struct ThreadReserve
    {
        uint32_t base = 0;
        uint32_t remain = 0;
        uint64_t bucket_off = 0; // the bucket this reservation belongs to
    };

    // Thread-local recon buffer helper: avoid allocating per-candidate during searches.
    static inline float *thread_local_recon(size_t dim)
    {
        static thread_local std::vector<float> recon_buf;
        if (recon_buf.size() < dim)
            recon_buf.resize(dim);
        return recon_buf.data();
    }

    // Resolve bucket offset to a pointer to payload (after the uint32_t arena header).
    static std::optional<const char *> resolve_bucket_base(const ArenaView &arena, uint64_t bucket_off, std::vector<char> &temp_buffer)
    {
        if (bucket_off == 0) return std::nullopt;
        const char *ram_blob_ptr = arena.blob_ptr_from_offset_for_map(bucket_off);
        if (!ram_blob_ptr)
        {
            temp_buffer = arena.read_remote_blob(bucket_off);
            if (temp_buffer.empty()) return std::nullopt;
            return temp_buffer.data() + sizeof(uint32_t);
        }
        return ram_blob_ptr + sizeof(uint32_t);
    }

    // ----------------- ArenaView dispatch impl -----------------
    char *ArenaView::alloc_blob(uint32_t len) const
    {
        if (pa) return pa->alloc_blob(len);
        if (sa) return sa->alloc_blob(len);
        return nullptr;
    }
    uint64_t ArenaView::offset_from_blob_ptr(const char *p) const noexcept
    {
        if (pa) return pa->offset_from_blob_ptr(p);
        if (sa) return sa->offset_from_blob_ptr(p);
        return UINT64_MAX;
    }
    const char *ArenaView::blob_ptr_from_offset_for_map(uint64_t offset) const
    {
        if (pa) return pa->blob_ptr_from_offset_for_map(offset);
        if (sa) return sa->blob_ptr_from_offset_for_map(offset);
        return nullptr;
    }
    std::vector<char> ArenaView::read_remote_blob(uint64_t remote_id) const
    {
        if (pa) return pa->read_remote_blob(remote_id);
        if (sa) return sa->read_remote_blob(remote_id);
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
    }

    PomaiOrbit::~PomaiOrbit()
    {
        if (cortex_) cortex_->stop();
    }

    // ------------------- Persistence (schema) -------------------
    void PomaiOrbit::save_schema()
    {
        SchemaHeader header;
        header.dim = cfg_.dim;
        header.num_centroids = cfg_.num_centroids;

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
        if (header.num_centroids > 0) cfg_.num_centroids = header.num_centroids;
        return true;
    }

    // ------------------- Train / routing / alloc -------------------
    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0)
            return false;

        if (cfg_.num_centroids == 0)
        {
            cfg_.num_centroids = static_cast<size_t>(std::sqrt(static_cast<double>(n)));
            cfg_.num_centroids = std::clamp(cfg_.num_centroids, static_cast<size_t>(64), static_cast<size_t>(4096));
        }

        size_t avg_density = (n / cfg_.num_centroids) + 1;
        dynamic_bucket_capacity_ = static_cast<uint32_t>(avg_density * 1.5);
        dynamic_bucket_capacity_ = std::clamp(dynamic_bucket_capacity_, static_cast<uint32_t>(32), static_cast<uint32_t>(512));

        std::clog << "[Orbit Autopilot] Training Config: Centroids=" << cfg_.num_centroids
                  << ", BucketCap=" << dynamic_bucket_capacity_
                  << " (N=" << n << ")\n";

        size_t num_c = std::min(n, cfg_.num_centroids);
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
            centroids_[i]->neighbors.reserve(cfg_.m_neighbors * 2);
        }

        // Build neighbor lists (simple nearest centroids)
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

        // allocate initial buckets
        for (size_t i = 0; i < num_c; ++i)
        {
            uint64_t off = alloc_new_bucket(static_cast<uint32_t>(i));
            centroids_[i]->bucket_offset.store(off, std::memory_order_release);
        }

        save_schema();
        return true;
    }

    uint64_t PomaiOrbit::alloc_new_bucket(uint32_t centroid_id)
    {
        size_t cap = dynamic_bucket_capacity_;

        size_t head_sz = sizeof(BucketHeader);
        size_t code_len_sz = sizeof(uint16_t) * cap; // per-slot length
        size_t vec_sz = MAX_ECHO_BYTES * cap;
        size_t ids_sz = sizeof(uint64_t) * cap;
        auto align64 = [](size_t s)
        { return (s + 63) & ~63; };

        uint32_t off_fp = static_cast<uint32_t>(align64(head_sz));    // unused
        uint32_t off_pq = static_cast<uint32_t>(align64(off_fp + 0)); // place lengths here
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

        // zero lengths so readers skip until writers publish
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

    // ------------------- insert / get / remove -------------------

    bool PomaiOrbit::insert(const float *vec, uint64_t label)
    {
        if (!eeq_ || centroids_.empty())
            return false;

        // thread-local reservation state
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
                if (current_off == 0) return false;
                node.bucket_offset.store(current_off, std::memory_order_release);
            }
        }

        pomai::ai::EchoCode code;
        try { code = eeq_->encode(vec); } catch (...) { return false; }

        while (true)
        {
            const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
            if (!ram_blob_ptr)
            {
                // slow path: lock and refresh
                std::unique_lock<std::shared_mutex> lock(node.mu);
                current_off = node.bucket_offset.load(std::memory_order_acquire);
                if (current_off == 0) return false;
                ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!ram_blob_ptr) return false;
                // invalidate thread-local reserve if it belonged to another bucket
                tres.remain = 0;
                tres.bucket_off = 0;
            }
            char *bucket_ptr = const_cast<char *>(ram_blob_ptr) + sizeof(uint32_t);
            BucketHeader *hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);

            uint32_t cap = dynamic_bucket_capacity_;

            // Try to use thread-local reserve (fast path)
            if (tres.remain == 0 || tres.bucket_off != current_off)
            {
                // attempt to reserve a chunk
                uint32_t chunk = RESERVE_CHUNK;
                uint32_t slot_base = hdr->count.fetch_add(chunk, std::memory_order_acq_rel);

                if (slot_base >= cap)
                {
                    // no room: rollback and go to slow path
                    hdr->count.fetch_sub(chunk, std::memory_order_acq_rel);

                    // acquire node.mu and allocate next bucket if needed
                    std::unique_lock<std::shared_mutex> lock(node.mu);
                    const char *blob_ptr2 = arena_.blob_ptr_from_offset_for_map(current_off);
                    if (!blob_ptr2) return false;
                    char *bucket_ptr2 = const_cast<char *>(blob_ptr2) + sizeof(uint32_t);
                    BucketHeader *hdr2 = reinterpret_cast<BucketHeader *>(bucket_ptr2);

                    uint64_t nb = hdr2->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(cid);
                        if (nb == 0) { std::clog << "[Orbit] insert: alloc_new_bucket failed\n"; return false; }
                        hdr2->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    // reset thread local
                    tres.remain = 0;
                    tres.bucket_off = 0;
                    continue;
                }
                // compute usable (in case chunk partially exceeds cap)
                uint32_t usable = chunk;
                if (slot_base + chunk > cap)
                {
                    usable = (slot_base < cap) ? (cap - slot_base) : 0;
                    // adjust count to reflect only usable
                    if (usable < chunk)
                        hdr->count.fetch_sub(chunk - usable, std::memory_order_acq_rel);
                }
                if (usable == 0)
                {
                    // nothing usable, go to next bucket
                    std::unique_lock<std::shared_mutex> lock(node.mu);
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(cid);
                        if (nb == 0) return false;
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

            // consume one from thread-local reserve
            uint32_t idx = tres.base + (RESERVE_CHUNK - tres.remain);
            tres.remain--;

            if (idx >= dynamic_bucket_capacity_)
            {
                // Shouldn't happen but guard: fallback to loop.
                tres.remain = 0;
                continue;
            }

            // serialize locally
            uint8_t tmpbuf[MAX_ECHO_BYTES];
            size_t pos = 0;
            tmpbuf[pos++] = code.depth;
            for (uint8_t q : code.scales_q) if (pos < MAX_ECHO_BYTES) tmpbuf[pos++] = q;
            bool overflow = false;
            for (const auto &sb : code.sign_bytes)
            {
                if (pos + sb.size() > MAX_ECHO_BYTES) { overflow = true; break; }
                std::memcpy(tmpbuf + pos, sb.data(), sb.size());
                pos += sb.size();
            }
            if (overflow)
            {
                // rollback single slot
                hdr->count.fetch_sub(1, std::memory_order_acq_rel);
                tres.remain = 0;
                return false;
            }

            // write vector bytes then publish id then publish len (len is release)
            char *vec_area = bucket_ptr + hdr->off_vectors;
            char *slot_ptr = vec_area + static_cast<size_t>(idx) * MAX_ECHO_BYTES;
            std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
            std::memcpy(slot_ptr, tmpbuf, pos);

            uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);
            pomai::ai::atomic_utils::atomic_store_u64(id_base + idx, label);

            uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
            __atomic_store_n(&len_base[idx], static_cast<uint16_t>(pos), __ATOMIC_RELEASE);

            // Update label maps with a single lock (per-insert this is one lock; batches will group further)
            {
                std::unique_lock<std::shared_mutex> lm(label_map_mu_);
                label_to_bucket_[label] = current_off;
                label_to_slot_[label] = idx;
            }

            return true;
        }

        return false; // unreachable
    }

    bool PomaiOrbit::get(uint64_t label, std::vector<float> &out_vec)
    {
        std::shared_lock<std::shared_mutex> dm(del_mu_);
        if (deleted_labels_.count(label))
            return false;

        uint64_t bucket_off = 0;
        {
            std::shared_lock<std::shared_mutex> lm(label_map_mu_);
            auto it = label_to_bucket_.find(label);
            if (it == label_to_bucket_.end())
                return false;
            bucket_off = it->second;
        }
        if (bucket_off == 0)
            return false;

        const char *data_base_ptr = nullptr;
        std::vector<char> temp_buffer;
        auto base_opt = resolve_bucket_base(arena_, bucket_off, temp_buffer);
        if (!base_opt) return false;
        data_base_ptr = *base_opt;

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
        size_t pos = 0;

        pomai::ai::EchoCode code;
        code.depth = (pos < len) ? ub[pos++] : 0;
        size_t depth = code.depth;
        code.scales_q.resize(depth);
        for (size_t k = 0; k < depth; ++k)
        {
            if (pos < len)
                code.scales_q[k] = ub[pos++];
            else
                code.scales_q[k] = 0;
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

    // ------------------- Budget helpers & search implementations -------------------

    static inline bool pay_ops(uint32_t &ops_left, uint32_t cost)
    {
        if (ops_left < cost)
            return false;
        ops_left -= cost;
        return true;
    }

    // compute_distance_for_id_with_proj: use precomputed qproj/qnorm
    bool PomaiOrbit::compute_distance_for_id_with_proj(const std::vector<std::vector<float>> &qproj, float qnorm2, uint64_t id, float &out_dist)
    {
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.count(id)) return false;
        }

        uint64_t bucket_off = 0;
        uint32_t cached_slot = UINT32_MAX;
        {
            std::shared_lock<std::shared_mutex> lm(label_map_mu_);
            auto it = label_to_bucket_.find(id);
            if (it == label_to_bucket_.end()) return false;
            bucket_off = it->second;

            auto sit = label_to_slot_.find(id);
            if (sit != label_to_slot_.end())
                cached_slot = sit->second;
        }
        if (bucket_off == 0) return false;

        std::vector<char> temp_buffer;
        auto base_opt = resolve_bucket_base(arena_, bucket_off, temp_buffer);
        if (!base_opt) return false;
        const char *data_base_ptr = *base_opt;

        const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(data_base_ptr);
        uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
        if (count == 0) return false;

        const uint64_t *id_base = reinterpret_cast<const uint64_t *>(data_base_ptr + hdr_ptr->off_ids);
        int32_t found = -1;

        if (cached_slot != UINT32_MAX && cached_slot < count)
        {
            uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + cached_slot);
            if (v == id)
                found = static_cast<int32_t>(cached_slot);
            else
            {
                // cache mismatch -> erase
                std::unique_lock<std::shared_mutex> lm(label_map_mu_);
                auto sit = label_to_slot_.find(id);
                if (sit != label_to_slot_.end() && sit->second == cached_slot)
                    label_to_slot_.erase(sit);
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
                    std::unique_lock<std::shared_mutex> lm(label_map_mu_);
                    label_to_slot_[id] = static_cast<uint32_t>(i);
                    break;
                }
            }
        }

        if (found < 0) return false;

        const uint16_t *len_base = reinterpret_cast<const uint16_t *>(data_base_ptr + hdr_ptr->off_pq_codes);
        uint16_t len = __atomic_load_n(&len_base[found], __ATOMIC_ACQUIRE);
        if (len == 0 || len > MAX_ECHO_BYTES) return false;

        const char *slot_ptr = data_base_ptr + hdr_ptr->off_vectors + static_cast<size_t>(found) * MAX_ECHO_BYTES;
        const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

        out_dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);
        return true;
    }

    // Legacy compute_distance_for_id kept (computes qproj on its own)
    bool PomaiOrbit::compute_distance_for_id(const float *query, uint64_t id, float &out_dist)
    {
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);
        return compute_distance_for_id_with_proj(qproj, qnorm2, id, out_dist);
    }

    // ------------------- search_with_budget & filtered_with_budget -------------------

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_with_budget(
        const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe)
    {
        if (!query || k == 0) return {};
        uint32_t ops_left = budget.ops_budget;
        if (!pay_ops(ops_left, 1)) return {};
        if (centroids_.empty()) return {};

        // Precompute qproj + qnorm once for budget search
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);

        // Build small Bloom snapshot if deleted set is large enough
        std::optional<PomaiOrbit::DeletedBloom> bloom;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256)
            {
                bloom.emplace();
                for (uint64_t v : deleted_labels_) bloom->add(v);
            }
        }

        if (nprobe == 0)
        {
            nprobe = std::max(1UL, cfg_.num_centroids / 50);
            if (k > 50) nprobe *= 2;
        }
        auto targets = find_routing_centroids(query, nprobe);

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        const uint32_t cost_check = 1;
        const uint32_t cost_decode = 5;
        const uint32_t cost_exact = 100;

        for (uint32_t cid : targets)
        {
            if (ops_left == 0) break;
            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (current_off != 0)
            {
                if (ops_left == 0) break;
                const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!ram_blob_ptr) break;
                const char *bucket_base = ram_blob_ptr + sizeof(uint32_t);
                const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(bucket_base);

                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);
                if (count == 0) { current_off = next_bucket_offset; continue; }

                const uint16_t *len_base = reinterpret_cast<const uint16_t *>(bucket_base + hdr_ptr->off_pq_codes);
                const char *vec_area = bucket_base + hdr_ptr->off_vectors;
                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(bucket_base + hdr_ptr->off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    if (ops_left == 0) break;
                    if (!pay_ops(ops_left, cost_check)) { ops_left = 0; break; }

                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                    if (bloom && bloom->maybe_contains(id)) continue;

                    if (!pay_ops(ops_left, cost_decode)) continue;

                    uint16_t len = __atomic_load_n(&len_base[i], __ATOMIC_ACQUIRE);
                    if (len == 0 || len > MAX_ECHO_BYTES) continue;

                    const char *slot_ptr = vec_area + static_cast<size_t>(i) * MAX_ECHO_BYTES;
                    const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

                    // Fast in-place approximate distance on packed bytes (no EchoCode construction)
                    float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);

                    if (topk.size() < k) topk.push({dist, id});
                    else if (dist < topk.top().first) { topk.pop(); topk.push({dist, id}); }

                    if (budget.allow_exact_refine && ops_left >= cost_exact)
                    {
                        // optional refine: decode top candidates later
                    }
                }

                current_off = next_bucket_offset;
            }
        }

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(topk.size());
        while (!topk.empty()) { auto p = topk.top(); topk.pop(); out.emplace_back(p.second, p.first); }
        std::reverse(out.begin(), out.end());
        return out;
    }

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered_with_budget(
        const float *query, size_t k, const std::vector<uint64_t> &candidates, const pomai::ai::Budget &budget)
    {
        if (!query || k == 0 || candidates.empty()) return {};

        uint32_t ops_left = budget.ops_budget;
        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        const uint32_t cost_check = 1;
        const uint32_t cost_decode = 5;

        // Precompute qproj + qnorm once for budget filtered search
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);

        // Bloom snapshot for deleted labels if large
        std::optional<PomaiOrbit::DeletedBloom> bloom;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256)
            {
                bloom.emplace();
                for (uint64_t v : deleted_labels_) bloom->add(v);
            }
        }

        for (uint64_t id : candidates)
        {
            if (ops_left == 0) break;
            if (bloom && bloom->maybe_contains(id)) continue;
            if (!pay_ops(ops_left, cost_check)) break;

            // locate bucket quickly (cheap)
            uint64_t bucket_off = 0;
            {
                std::shared_lock<std::shared_mutex> lm(label_map_mu_);
                auto it = label_to_bucket_.find(id);
                if (it == label_to_bucket_.end())
                    continue;
                bucket_off = it->second;
            }
            if (bucket_off == 0)
                continue;

            if (!pay_ops(ops_left, cost_decode)) continue;

            // resolve bucket
            std::vector<char> temp;
            auto base_opt = resolve_bucket_base(arena_, bucket_off, temp);
            if (!base_opt) continue;
            const char *data_base_ptr = *base_opt;

            const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(data_base_ptr);
            uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
            if (count == 0) continue;

            const uint64_t *id_base = reinterpret_cast<const uint64_t *>(data_base_ptr + hdr_ptr->off_ids);
            const uint16_t *len_base = reinterpret_cast<const uint16_t *>(data_base_ptr + hdr_ptr->off_pq_codes);
            const char *vec_area = data_base_ptr + hdr_ptr->off_vectors;

            // find slot in this bucket
            int32_t found = -1;
            for (uint32_t j = 0; j < count; ++j)
            {
                uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + j);
                if (v == id) { found = static_cast<int32_t>(j); break; }
            }
            if (found < 0) continue;

            uint16_t len = __atomic_load_n(&len_base[found], __ATOMIC_ACQUIRE);
            if (len == 0 || len > MAX_ECHO_BYTES) continue;

            const char *slot_ptr = vec_area + static_cast<size_t>(found) * MAX_ECHO_BYTES;
            const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

            // fast in-place approx using precomputed qproj
            float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);

            if (topk.size() < k)
                topk.push({dist, id});
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
            auto p = topk.top();
            topk.pop();
            out.emplace_back(p.second, p.first);
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    // ------------------- Search (public, hot path uses ADC on-code) -------------------
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        if (!query || k == 0) return {};

        // Precompute projections and qnorm once for the query
        std::vector<std::vector<float>> qproj;
        eeq_->project_query(query, qproj);
        float qnorm2 = ::pomai_dot(query, query, cfg_.dim);
        const std::vector<float> &layer_energy = eeq_->layer_col_energy();
        (void)layer_energy; // kept for reference; not used directly here

        // Build Bloom snapshot if deleted_labels_ large
        std::optional<PomaiOrbit::DeletedBloom> bloom;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.size() > 256)
            {
                bloom.emplace();
                for (uint64_t v : deleted_labels_) bloom->add(v);
            }
        }

        if (centroids_.empty()) return {};

        if (nprobe == 0)
        {
            nprobe = std::max(1UL, cfg_.num_centroids / 50);
            if (k > 50) nprobe *= 2;
        }
        auto targets = find_routing_centroids(query, nprobe);

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        for (uint32_t cid : targets)
        {
            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (current_off != 0)
            {
                std::vector<char> tmp;
                auto base_opt = resolve_bucket_base(arena_, current_off, tmp);
                if (!base_opt) break;
                const char *bucket_base = *base_opt;
                const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(bucket_base);

                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);
                if (count == 0) { current_off = next_bucket_offset; continue; }

                const uint16_t *len_base = reinterpret_cast<const uint16_t *>(bucket_base + hdr_ptr->off_pq_codes);
                const char *vec_area = bucket_base + hdr_ptr->off_vectors;
                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(bucket_base + hdr_ptr->off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                    if (bloom && bloom->maybe_contains(id)) continue;

                    uint16_t len = __atomic_load_n(&len_base[i], __ATOMIC_ACQUIRE);
                    if (len == 0 || len > MAX_ECHO_BYTES) continue;

                    const char *slot_ptr = vec_area + static_cast<size_t>(i) * MAX_ECHO_BYTES;
                    const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);

                    // Fast in-place approximate distance on packed bytes (no EchoCode construction)
                    float dist = eeq_->approx_dist_code_bytes(qproj, qnorm2, ub, len);

                    if (topk.size() < k) topk.push({dist, id});
                    else if (dist < topk.top().first) { topk.pop(); topk.push({dist, id}); }
                }

                current_off = next_bucket_offset;
            }
        }

        std::vector<std::pair<uint64_t, float>> out;
        out.reserve(topk.size());
        while (!topk.empty()) { auto p = topk.top(); topk.pop(); out.emplace_back(p.second, p.first); }
        std::reverse(out.begin(), out.end());
        return out;
    }

    // ------------------- insert_batch: pre-serialize (outside lock) + reserve via fetch_add (short lock)
    //                   + grouped label_map updates (single lock per bucket) -------------------
    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (!eeq_ || centroids_.empty() || batch.empty())
            return false;

        struct SerializedItem {
            uint64_t label;
            uint8_t size;
            uint8_t bytes[MAX_ECHO_BYTES];
        };
        struct Prepared {
            uint32_t cid;
            SerializedItem it;
        };

        // Phase 1: prepare (no locks) - encode and serialize into fixed buffer
        std::vector<Prepared> prep;
        prep.reserve(batch.size());

        for (const auto &p : batch)
        {
            uint32_t cid = find_nearest_centroid(p.second.data());
            pomai::ai::EchoCode code;
            try { code = eeq_->encode(p.second.data()); } catch (...) { continue; }

            SerializedItem si;
            si.label = p.first;
            size_t pos = 0;
            if (pos < MAX_ECHO_BYTES) si.bytes[pos++] = code.depth;
            for (uint8_t q : code.scales_q) { if (pos < MAX_ECHO_BYTES) si.bytes[pos++] = q; else break; }
            bool overflow = false;
            for (const auto &sb : code.sign_bytes)
            {
                if (pos + sb.size() > MAX_ECHO_BYTES) { overflow = true; break; }
                std::memcpy(si.bytes + pos, sb.data(), sb.size());
                pos += sb.size();
            }
            if (overflow || pos == 0) continue;
            si.size = static_cast<uint8_t>(pos);
            prep.push_back({cid, si});
        }

        if (prep.empty()) return true;

        std::sort(prep.begin(), prep.end(), [](auto &a, auto &b){ return a.cid < b.cid; });

        // Phase 2: per-centroid short-lock + reserve via fetch_add
        size_t idx = 0, total = prep.size();
        while (idx < total)
        {
            uint32_t current_cid = prep[idx].cid;
            OrbitNode &node = *centroids_[current_cid];

            // we will hold node.mu only when allocating/chaining buckets; but we keep it
            // while filling slots here to keep code simple (short duration)
            std::unique_lock<std::shared_mutex> lock(node.mu);

            uint64_t current_off = node.bucket_offset.load(std::memory_order_acquire);
            if (current_off == 0)
            {
                current_off = alloc_new_bucket(current_cid);
                if (current_off == 0) return false;
                node.bucket_offset.store(current_off, std::memory_order_release);
            }

            // For each bucket chain, fill as many as possible, grouping label_map updates
            while (idx < total && prep[idx].cid == current_cid)
            {
                std::vector<std::pair<uint64_t, uint32_t>> label_updates; // collect (label, slot)

                const char *blob = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!blob)
                {
                    // if mapping failed, try to allocate next bucket and continue
                    uint64_t nb = alloc_new_bucket(current_cid);
                    if (nb == 0) return false;
                    BucketHeader *hdr_prev = reinterpret_cast<BucketHeader *>(const_cast<char *>(arena_.blob_ptr_from_offset_for_map(current_off)) + sizeof(uint32_t));
                    hdr_prev->next_bucket_offset.store(nb, std::memory_order_release);
                    current_off = nb;
                    continue;
                }

                char *bucket_ptr = const_cast<char *>(blob) + sizeof(uint32_t);
                BucketHeader *hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);
                uint32_t cur_count = hdr->count.load(std::memory_order_relaxed);

                if (cur_count >= dynamic_bucket_capacity_)
                {
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(current_cid);
                        if (nb == 0) return false;
                        hdr->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    continue;
                }

                // Determine how many items fit into this bucket
                size_t start_idx = idx;
                uint32_t remaining = dynamic_bucket_capacity_ - cur_count;
                uint32_t fit = 0;
                while (idx < total && prep[idx].cid == current_cid && fit < remaining)
                {
                    if (prep[idx].it.size == 0 || prep[idx].it.size > MAX_ECHO_BYTES) { ++idx; continue; }
                    ++fit;
                    ++idx;
                }
                if (fit == 0) continue;

                // Reserve slots atomically
                uint32_t slot_base = hdr->count.fetch_add(fit, std::memory_order_acq_rel);

                if (slot_base + fit > dynamic_bucket_capacity_)
                {
                    // reservation overflow (rare): rollback and move to next bucket
                    hdr->count.fetch_sub(fit, std::memory_order_acq_rel);
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(current_cid);
                        if (nb == 0) return false;
                        hdr->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    // reset idx so we retry this segment in next bucket
                    idx = start_idx;
                    continue;
                }

                // Write each reserved slot: memcpy bytes, store id, then publish len (release)
                char *vec_area = bucket_ptr + hdr->off_vectors;
                uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
                uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);

                for (uint32_t j = 0; j < fit; ++j)
                {
                    const auto &it = prep[start_idx + j].it;
                    uint32_t slot = slot_base + j;

                    char *slot_ptr = vec_area + static_cast<size_t>(slot) * MAX_ECHO_BYTES;
                    std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
                    std::memcpy(slot_ptr, it.bytes, it.size);

                    // publish id (atomic) before letting readers consider len
                    pomai::ai::atomic_utils::atomic_store_u64(id_base + slot, it.label);

                    // publish len with release so readers see vector+id first
                    __atomic_store_n(&len_base[slot], static_cast<uint16_t>(it.size), __ATOMIC_RELEASE);

                    // collect label->slot update to apply in batch
                    label_updates.emplace_back(it.label, slot);
                }

                // Apply label_map updates with a single lock acquisition
                if (!label_updates.empty())
                {
                    std::unique_lock<std::shared_mutex> lm(label_map_mu_);
                    for (const auto &lu : label_updates)
                    {
                        label_to_bucket_[lu.first] = current_off;
                        label_to_slot_[lu.first] = lu.second;
                    }
                }

                // continue to fill same bucket (idx already advanced)
            }
        }

        return true;
    }

    // ------------------- Save/Load routing stubs -------------------
    bool PomaiOrbit::save_routing(const std::string & /*path*/) { return false; }
    bool PomaiOrbit::load_routing(const std::string & /*path*/) { return false; }

} // namespace pomai::ai::orbit