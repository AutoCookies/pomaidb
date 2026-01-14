/*
 * src/ai/pomai_orbit.cc
 *
 * PomaiOrbit implementation updated to integrate WhisperGrain budget-aware search.
 *
 * Notes:
 *  - If a WhisperGrain controller is attached to the orbit (whisper_ctrl_),
 *    the public search() and search_filtered() will route through the budget-aware
 *    implementations (search_with_budget / search_filtered_with_budget).
 *  - Budget-aware implementations use a simple ops accounting model (prototype).
 *    Costs are conservative approximations and should be calibrated with measurements.
 *
 * This file keeps the previous behavior when no WhisperGrain is attached.
 */

#include "src/ai/pomai_orbit.h"

#include "src/ai/atomic_utils.h"
#include "src/memory/arena.h"
#include "src/memory/shard_arena.h"
#include "src/ai/eternalecho_quantizer.h"
#include "src/ai/whispergrain.h"

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

namespace pomai::ai::orbit
{

    // Conservative per-entry bytes for serialized EchoCode.
    // Increase if eeq_cfg uses more bits/layers.
    static constexpr size_t MAX_ECHO_BYTES = 48;

    // ----------------- ArenaView dispatch impl -----------------
    char *ArenaView::alloc_blob(uint32_t len) const
    {
        if (pa)
            return pa->alloc_blob(len);
        if (sa)
            return sa->alloc_blob(len);
        return nullptr;
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

    // ---- small helpers (l2sq) ----
    static inline float l2sq(const float *a, const float *b, size_t dim)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    // ------------------- Constructors / Destructors -------------------
    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        if (!arena_.is_pomai_arena())
            throw std::invalid_argument("Arena null");
        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
        schema_file_path_ = (p / "pomai_schema.bin").string();

        // Initialize EternalEchoQuantizer (single shared instance)
        eeq_ = std::make_unique<pomai::ai::EternalEchoQuantizer>(cfg_.dim, cfg_.eeq_cfg);

        if (std::filesystem::exists(schema_file_path_))
        {
            std::clog << "[Orbit] Found existing DB. Loading schema...\n";
            if (load_schema())
                std::clog << "[Orbit] Restored: Dim=" << cfg_.dim << ", Centroids=" << cfg_.num_centroids << "\n";
            else
                std::cerr << "[Orbit] Failed to load schema!\n";
        }
        else
        {
            if (cfg_.dim == 0)
                throw std::runtime_error("[Orbit] New DB initialization requires 'dim' to be set!");
            std::clog << "[Orbit] Initializing NEW Database (Dim=" << cfg_.dim << ")\n";
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
            throw std::invalid_argument("ShardArena null");
        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p))
            std::filesystem::create_directories(p);
        schema_file_path_ = (p / "pomai_schema.bin").string();

        eeq_ = std::make_unique<pomai::ai::EternalEchoQuantizer>(cfg_.dim, cfg_.eeq_cfg);

        if (std::filesystem::exists(schema_file_path_))
        {
            std::clog << "[Orbit] Found existing DB. Loading schema...\n";
            if (load_schema())
                std::clog << "[Orbit] Restored: Dim=" << cfg_.dim << ", Centroids=" << cfg_.num_centroids << "\n";
            else
                std::cerr << "[Orbit] Failed to load schema!\n";
        }
        else
        {
            if (cfg_.dim == 0)
                throw std::runtime_error("[Orbit] New DB initialization requires 'dim' to be set!");
            std::clog << "[Orbit] Initializing NEW Database (Dim=" << cfg_.dim << ")\n";
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
        if (cortex_)
            cortex_->stop();
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
            std::cerr << "[Orbit] ERROR: Could not save schema to " << schema_file_path_ << "\n";
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
            std::cerr << "[Orbit] Invalid schema file signature!\n";
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

        if (cfg_.num_centroids == 0)
        {
            cfg_.num_centroids = static_cast<size_t>(std::sqrt(n));
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
        std::vector<size_t> indices;
        indices.resize(n);
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
        for (size_t i = 0; i < num_c; ++i)
        {
            std::vector<std::pair<float, uint32_t>> dists;
            for (size_t j = 0; j < num_c; ++j)
            {
                if (i == j)
                    continue;
                float d = l2sq(centroids_[i]->vector.data(), centroids_[j]->vector.data(), cfg_.dim);
                dists.push_back({d, static_cast<uint32_t>(j)});
            }
            std::sort(dists.begin(), dists.end());
            for (size_t k = 0; k < std::min(dists.size(), cfg_.m_neighbors); ++k)
                centroids_[i]->neighbors.push_back(dists[k].second);
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

        // zero lengths
        uint16_t *len_base = reinterpret_cast<uint16_t *>(blob_ptr + sizeof(uint32_t) + off_pq);
        for (size_t i = 0; i < cap; ++i)
            len_base[i] = 0;

        return offset;
    }

    uint32_t PomaiOrbit::find_nearest_centroid(const float *vec)
    {
        uint32_t best = 0;
        float min_d = std::numeric_limits<float>::max();
        for (size_t i = 0; i < centroids_.size(); ++i)
        {
            float d = l2sq(vec, centroids_[i]->vector.data(), cfg_.dim);
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
        pq.push({l2sq(vec, centroids_[entry]->vector.data(), cfg_.dim), entry});
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
                    float d = l2sq(vec, centroids_[nb]->vector.data(), cfg_.dim);
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
        if (!eeq_)
            return false;
        if (centroids_.empty())
            return false;

        uint32_t cid = find_nearest_centroid(vec);
        OrbitNode &node = *centroids_[cid];
        std::unique_lock<std::shared_mutex> lock(node.mu);

        uint64_t current_off = node.bucket_offset.load(std::memory_order_acquire);
        char *bucket_ptr = nullptr;
        BucketHeader *hdr = nullptr;

        while (current_off != 0)
        {
            const char *blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
            if (!blob_ptr)
                return false;
            bucket_ptr = const_cast<char *>(blob_ptr) + sizeof(uint32_t);
            hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);
            if (hdr->count.load(std::memory_order_relaxed) < dynamic_bucket_capacity_)
                break;
            uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
            if (nb == 0)
            {
                uint64_t new_off = alloc_new_bucket(cid);
                hdr->next_bucket_offset.store(new_off, std::memory_order_release);
                current_off = new_off;
            }
            else
                current_off = nb;
        }
        if (!hdr)
            return false;

        uint32_t idx = hdr->count.load(std::memory_order_relaxed);

        // EEQ encode
        pomai::ai::EchoCode code = eeq_->encode(vec);

        // serialize: [depth(1)][scales_q x depth][sign_bytes concat]
        std::vector<uint8_t> buf;
        buf.reserve(MAX_ECHO_BYTES);
        buf.push_back(code.depth);
        for (uint8_t q : code.scales_q)
            buf.push_back(q);
        for (const auto &sb : code.sign_bytes)
            buf.insert(buf.end(), sb.begin(), sb.end());

        if (buf.size() > MAX_ECHO_BYTES)
            return false; // too large

        // write into slot
        char *vec_area = bucket_ptr + hdr->off_vectors;
        char *slot_ptr = vec_area + static_cast<size_t>(idx) * MAX_ECHO_BYTES;
        std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
        std::memcpy(slot_ptr, buf.data(), buf.size());

        // write length
        uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
        len_base[idx] = static_cast<uint16_t>(buf.size());

        // write id
        uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);
        pomai::ai::atomic_utils::atomic_store_u64(id_base + idx, label);

        // update maps
        {
            std::unique_lock<std::shared_mutex> lm(label_map_mu_);
            label_to_bucket_[label] = current_off;
        }
        {
            std::unique_lock<std::shared_mutex> dm(del_mu_);
            auto it = deleted_labels_.find(label);
            if (it != deleted_labels_.end())
                deleted_labels_.erase(it);
        }

        hdr->count.fetch_add(1, std::memory_order_release);
        return true;
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

        const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(bucket_off);
        const char *data_base_ptr = nullptr;
        std::vector<char> temp_buffer;
        if (!ram_blob_ptr)
        {
            temp_buffer = arena_.read_remote_blob(bucket_off);
            if (temp_buffer.empty())
                return false;
            data_base_ptr = temp_buffer.data() + sizeof(uint32_t);
        }
        else
            data_base_ptr = ram_blob_ptr + sizeof(uint32_t);

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
        uint16_t len = len_base[found];
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

    // ------------------- Budget helpers -------------------

    // Simple ops payment helper. Returns true if enough ops remain to pay cost.
    static inline bool pay_ops(uint32_t &ops_left, uint32_t cost)
    {
        if (ops_left < cost)
            return false;
        ops_left -= cost;
        return true;
    }

    // ------------------- Budget-aware search implementations -------------------

    // Budget-aware search over centroid routing.
    // This implementation uses a conservative per-item decode cost and
    // calls compute_distance_for_id which performs the actual decode.
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_with_budget(
        const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe)
    {
        if (!query || k == 0)
            return {};
        uint32_t ops_left = budget.ops_budget;

        // Charge tiny startup op
        if (!pay_ops(ops_left, 1))
            return {};

        if (centroids_.empty())
            return {};

        if (nprobe == 0)
        {
            nprobe = std::max(1UL, cfg_.num_centroids / 50);
            if (k > 50)
                nprobe *= 2;
        }

        auto targets = find_routing_centroids(query, nprobe);

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        const uint32_t cost_check = 1;
        const uint32_t cost_decode = 5;
        const uint32_t cost_exact = 100;

        for (uint32_t cid : targets)
        {
            if (ops_left == 0)
                break;
            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (current_off != 0)
            {
                if (ops_left == 0)
                    break;
                const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!ram_blob_ptr)
                    break;
                const char *bucket_base = ram_blob_ptr + sizeof(uint32_t);
                const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(bucket_base);

                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);
                if (count == 0)
                {
                    current_off = next_bucket_offset;
                    continue;
                }

                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(bucket_base + hdr_ptr->off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    if (ops_left == 0)
                        break;
                    if (!pay_ops(ops_left, cost_check))
                    {
                        ops_left = 0;
                        break;
                    }

                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                    {
                        std::shared_lock<std::shared_mutex> dm(del_mu_);
                        if (deleted_labels_.count(id))
                            continue;
                    }

                    // Pay decode cost (approx)
                    if (!pay_ops(ops_left, cost_decode))
                        continue;

                    float dist = 0.0f;
                    // compute_distance_for_id will locate and decode the vector (EEQ)
                    if (!compute_distance_for_id(query, id, dist))
                        continue;

                    if (topk.size() < k)
                        topk.push({dist, id});
                    else if (dist < topk.top().first)
                    {
                        topk.pop();
                        topk.push({dist, id});
                    }

                    // Optional: exact refine (not implemented fully) - pay only if budget requests and allowed
                    if (budget.allow_exact_refine && ops_left >= cost_exact)
                    {
                        // TODO: implement exact refine logic if desired (IO fetch + exact l2)
                        // pay_ops(ops_left, cost_exact);
                        // perform refine...
                    }
                }

                current_off = next_bucket_offset;
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

    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered_with_budget(
        const float *query, size_t k, const std::vector<uint64_t> &candidates, const pomai::ai::Budget &budget)
    {
        if (!query || k == 0 || candidates.empty())
            return {};

        uint32_t ops_left = budget.ops_budget;
        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        const uint32_t cost_check = 1;
        const uint32_t cost_decode = 5;

        // snapshot deleted labels
        std::unordered_set<uint64_t> local_deleted;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (!deleted_labels_.empty())
                local_deleted = deleted_labels_;
        }

        for (uint64_t id : candidates)
        {
            if (ops_left == 0)
                break;
            if (!local_deleted.empty() && local_deleted.count(id))
                continue;
            if (!pay_ops(ops_left, cost_check))
                break;

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

            // pay decode cost
            if (!pay_ops(ops_left, cost_decode))
                continue;

            float dist = 0.0f;
            if (!compute_distance_for_id(query, id, dist))
                continue;

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

    // ------------------- Search (public) -------------------
    // Public search() now routes through WhisperGrain if available; otherwise unchanged behavior.
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        if (whisper_ctrl_)
        {
            // compute budget via controller (non-hot by default)
            auto budget = whisper_ctrl_->compute_budget(false);
            return search_with_budget(query, k, budget, nprobe);
        }

        // legacy path (unchanged)
        if (centroids_.empty())
            return {};

        if (nprobe == 0)
        {
            nprobe = std::max(1UL, cfg_.num_centroids / 50);
            if (k > 50)
                nprobe *= 2;
        }

        std::vector<uint32_t> targets = find_routing_centroids(query, nprobe);

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        for (uint32_t cid : targets)
        {
            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);
            while (current_off != 0)
            {
                const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!ram_blob_ptr)
                    break;
                const char *bucket_base = ram_blob_ptr + sizeof(uint32_t);
                const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(bucket_base);

                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);
                if (count == 0)
                {
                    current_off = next_bucket_offset;
                    continue;
                }

                const uint64_t *id_base = reinterpret_cast<const uint64_t *>(bucket_base + hdr_ptr->off_ids);
                const uint16_t *len_base = reinterpret_cast<const uint16_t *>(bucket_base + hdr_ptr->off_pq_codes);
                const char *vec_area = bucket_base + hdr_ptr->off_vectors;

                for (uint32_t i = 0; i < count; ++i)
                {
                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                    {
                        std::shared_lock<std::shared_mutex> dm(del_mu_);
                        if (deleted_labels_.count(id))
                            continue;
                    }

                    uint16_t len = len_base[i];
                    if (len == 0 || len > MAX_ECHO_BYTES)
                        continue;
                    const char *slot_ptr = vec_area + static_cast<size_t>(i) * MAX_ECHO_BYTES;
                    const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);
                    size_t pos = 0;

                    EchoCode code;
                    code.depth = (pos < len) ? ub[pos++] : 0;
                    size_t depth = code.depth;
                    code.scales_q.resize(depth);
                    for (size_t kk = 0; kk < depth; ++kk)
                    {
                        if (pos < len)
                            code.scales_q[kk] = ub[pos++];
                        else
                            code.scales_q[kk] = 0;
                    }
                    code.bits_per_layer.resize(depth);
                    code.sign_bytes.resize(depth);
                    bool bad = false;
                    for (size_t kk = 0; kk < depth; ++kk)
                    {
                        uint32_t b = cfg_.eeq_cfg.bits_per_layer[kk];
                        code.bits_per_layer[kk] = b;
                        size_t bytes = (b + 7) / 8;
                        if (pos + bytes <= len)
                        {
                            code.sign_bytes[kk].assign(ub + pos, ub + pos + bytes);
                            pos += bytes;
                        }
                        else
                        {
                            bad = true;
                            break;
                        }
                    }
                    if (bad)
                        continue;

                    std::vector<float> recon(cfg_.dim);
                    eeq_->decode(code, recon.data());
                    float dist = l2sq(query, recon.data(), cfg_.dim);

                    if (topk.size() < k)
                        topk.push({dist, id});
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
            auto p = topk.top();
            topk.pop();
            out.emplace_back(p.second, p.first);
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    // search_filtered: if whisper_ctrl_ attached, use budget-aware filtered search
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search_filtered(
        const float *query, size_t k, const std::vector<uint64_t> &candidates)
    {
        if (whisper_ctrl_)
        {
            auto budget = whisper_ctrl_->compute_budget(false);
            return search_filtered_with_budget(query, k, candidates, budget);
        }

        // legacy filtered search (unchanged)
        if (!query || k == 0 || candidates.empty())
            return {};

        // snapshot label->bucket
        std::vector<std::pair<uint64_t, uint64_t>> entries;
        entries.reserve(candidates.size());
        {
            std::shared_lock<std::shared_mutex> lm(label_map_mu_);
            for (uint64_t id : candidates)
            {
                auto it = label_to_bucket_.find(id);
                if (it == label_to_bucket_.end())
                    continue;
                uint64_t boff = it->second;
                if (boff == 0)
                    continue;
                entries.emplace_back(boff, id);
            }
        }
        if (entries.empty())
            return {};

        std::sort(entries.begin(), entries.end(), [](auto &a, auto &b)
                  { return a.first < b.first; });

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> topk;

        // local deleted snapshot
        std::unordered_set<uint64_t> local_deleted;
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (!deleted_labels_.empty())
                local_deleted = deleted_labels_;
        }

        size_t idx = 0;
        while (idx < entries.size())
        {
            uint64_t bucket_off = entries[idx].first;
            const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(bucket_off);
            std::vector<char> temp_buf;
            const char *data_base_ptr = nullptr;
            if (!ram_blob_ptr)
            {
                temp_buf = arena_.read_remote_blob(bucket_off);
                if (temp_buf.empty())
                {
                    while (idx < entries.size() && entries[idx].first == bucket_off)
                        ++idx;
                    continue;
                }
                data_base_ptr = temp_buf.data() + sizeof(uint32_t);
            }
            else
                data_base_ptr = ram_blob_ptr + sizeof(uint32_t);

            const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(data_base_ptr);
            uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
            if (count == 0)
            {
                while (idx < entries.size() && entries[idx].first == bucket_off)
                    ++idx;
                continue;
            }

            const uint64_t *id_base = reinterpret_cast<const uint64_t *>(data_base_ptr + hdr_ptr->off_ids);
            const uint16_t *len_base = reinterpret_cast<const uint16_t *>(data_base_ptr + hdr_ptr->off_pq_codes);
            const char *vec_area = data_base_ptr + hdr_ptr->off_vectors;

            while (idx < entries.size() && entries[idx].first == bucket_off)
            {
                uint64_t id = entries[idx].second;
                ++idx;
                if (!local_deleted.empty() && local_deleted.count(id))
                    continue;

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

                uint16_t len = len_base[found];
                if (len == 0 || len > MAX_ECHO_BYTES)
                    continue;

                const char *slot_ptr = vec_area + static_cast<size_t>(found) * MAX_ECHO_BYTES;
                const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);
                size_t pos = 0;

                EchoCode code;
                code.depth = (pos < len) ? ub[pos++] : 0;
                size_t depth = code.depth;
                code.scales_q.resize(depth);
                for (size_t k2 = 0; k2 < depth; ++k2)
                {
                    if (pos < len)
                        code.scales_q[k2] = ub[pos++];
                    else
                        code.scales_q[k2] = 0;
                }
                code.bits_per_layer.resize(depth);
                code.sign_bytes.resize(depth);
                bool bad = false;
                for (size_t k2 = 0; k2 < depth; ++k2)
                {
                    uint32_t b = cfg_.eeq_cfg.bits_per_layer[k2];
                    code.bits_per_layer[k2] = b;
                    size_t bytes = (b + 7) / 8;
                    if (pos + bytes <= len)
                    {
                        code.sign_bytes[k2].assign(ub + pos, ub + pos + bytes);
                        pos += bytes;
                    }
                    else
                    {
                        bad = true;
                        break;
                    }
                }
                if (bad)
                    continue;

                std::vector<float> recon(cfg_.dim);
                eeq_->decode(code, recon.data());
                float dist = l2sq(query, recon.data(), cfg_.dim);

                if (topk.size() < k)
                    topk.push({dist, id});
                else if (dist < topk.top().first)
                {
                    topk.pop();
                    topk.push({dist, id});
                }
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

    // ------------------- compute_distance_for_id -------------------
    // helper for per-id distance (used by budgeted path)
    bool PomaiOrbit::compute_distance_for_id(const float *query, uint64_t id, float &out_dist)
    {
        // quick deleted check
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.count(id))
                return false;
        }

        uint64_t bucket_off = 0;
        {
            std::shared_lock<std::shared_mutex> lm(label_map_mu_);
            auto it = label_to_bucket_.find(id);
            if (it == label_to_bucket_.end())
                return false;
            bucket_off = it->second;
        }
        if (bucket_off == 0)
            return false;

        const char *ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(bucket_off);
        const char *data_base_ptr = nullptr;
        std::vector<char> temp_buffer;
        if (!ram_blob_ptr)
        {
            temp_buffer = arena_.read_remote_blob(bucket_off);
            if (temp_buffer.empty())
                return false;
            data_base_ptr = temp_buffer.data() + sizeof(uint32_t);
        }
        else
            data_base_ptr = ram_blob_ptr + sizeof(uint32_t);

        const BucketHeader *hdr_ptr = reinterpret_cast<const BucketHeader *>(data_base_ptr);
        uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
        if (count == 0)
            return false;

        const uint64_t *id_base = reinterpret_cast<const uint64_t *>(data_base_ptr + hdr_ptr->off_ids);
        int32_t found = -1;
        for (uint32_t i = 0; i < count; ++i)
        {
            uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
            if (v == id)
            {
                found = static_cast<int32_t>(i);
                break;
            }
        }
        if (found < 0)
            return false;

        const uint16_t *len_base = reinterpret_cast<const uint16_t *>(data_base_ptr + hdr_ptr->off_pq_codes);
        uint16_t len = len_base[found];
        if (len == 0 || len > MAX_ECHO_BYTES)
            return false;

        const char *slot_ptr = data_base_ptr + hdr_ptr->off_vectors + static_cast<size_t>(found) * MAX_ECHO_BYTES;
        const uint8_t *ub = reinterpret_cast<const uint8_t *>(slot_ptr);
        size_t pos = 0;

        EchoCode code;
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

        std::vector<float> recon(cfg_.dim);
        eeq_->decode(code, recon.data());
        out_dist = l2sq(query, recon.data(), cfg_.dim);
        return true;
    }

    bool PomaiOrbit::insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        if (!eeq_ || centroids_.empty() || batch.empty())
            return false;
        struct PendingItem
        {
            uint32_t cid;
            uint64_t label;
            pomai::ai::EchoCode code;
        };
        std::vector<PendingItem> items;
        items.reserve(batch.size());
        for (const auto &p : batch)
        {
            uint32_t cid = find_nearest_centroid(p.second.data());
            auto code = eeq_->encode(p.second.data());
            items.push_back({cid, p.first, std::move(code)});
        }
        std::sort(items.begin(), items.end(), [](const auto &a, const auto &b)
                  { return a.cid < b.cid; });
        size_t idx = 0, total = items.size();
        while (idx < total)
        {
            uint32_t current_cid = items[idx].cid;
            OrbitNode &node = *centroids_[current_cid];
            std::unique_lock<std::shared_mutex> lock(node.mu);
            uint64_t current_off = node.bucket_offset.load(std::memory_order_acquire);
            BucketHeader *hdr = nullptr;
            char *bucket_ptr = nullptr;
            while (idx < total && items[idx].cid == current_cid)
            {
                if (current_off == 0)
                {
                    current_off = alloc_new_bucket(current_cid);
                    node.bucket_offset.store(current_off, std::memory_order_release);
                }
                const char *blob = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!blob)
                    break;
                bucket_ptr = const_cast<char *>(blob) + sizeof(uint32_t);
                hdr = reinterpret_cast<BucketHeader *>(bucket_ptr);
                if (hdr->count.load(std::memory_order_relaxed) >= dynamic_bucket_capacity_)
                {
                    uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
                    if (nb == 0)
                    {
                        nb = alloc_new_bucket(current_cid);
                        hdr->next_bucket_offset.store(nb, std::memory_order_release);
                    }
                    current_off = nb;
                    continue;
                }
                uint32_t slot = hdr->count.load(std::memory_order_relaxed);
                auto &it = items[idx];
                std::vector<uint8_t> buf;
                buf.reserve(MAX_ECHO_BYTES);
                buf.push_back(it.code.depth);
                for (uint8_t q : it.code.scales_q)
                    buf.push_back(q);
                for (const auto &sb : it.code.sign_bytes)
                    buf.insert(buf.end(), sb.begin(), sb.end());
                if (buf.size() <= MAX_ECHO_BYTES)
                {
                    char *slot_ptr = bucket_ptr + hdr->off_vectors + static_cast<size_t>(slot) * MAX_ECHO_BYTES;
                    std::memset(slot_ptr, 0, MAX_ECHO_BYTES);
                    std::memcpy(slot_ptr, buf.data(), buf.size());
                    uint16_t *len_base = reinterpret_cast<uint16_t *>(bucket_ptr + hdr->off_pq_codes);
                    len_base[slot] = static_cast<uint16_t>(buf.size());
                    uint64_t *id_base = reinterpret_cast<uint64_t *>(bucket_ptr + hdr->off_ids);
                    pomai::ai::atomic_utils::atomic_store_u64(id_base + slot, it.label);
                    {
                        std::unique_lock<std::shared_mutex> lm(label_map_mu_);
                        label_to_bucket_[it.label] = current_off;
                    }
                    hdr->count.fetch_add(1, std::memory_order_release);
                }
                idx++;
            }
        }
        return true;
    }

    // ------------------- Save/Load routing stubs -------------------
    bool PomaiOrbit::save_routing(const std::string & /*path*/) { return false; }
    bool PomaiOrbit::load_routing(const std::string & /*path*/) { return false; }

} // namespace pomai::ai::orbit