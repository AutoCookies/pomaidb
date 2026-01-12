/*
 * src/ai/pomai_orbit.cc
 *
 * PomaiOrbit implementation with on-demand defrost support and lock-free readers.
 *
 * Changes:
 *  - Uses ArenaView adapter to talk to either PomaiArena or ShardArena.
 *  - Adds in-memory label->bucket index and soft-delete set.
 *  - Implements get(label) and remove(label).
 */

#include "src/ai/pomai_orbit.h"

#include "src/ai/atomic_utils.h"
#include "src/ai/ids_block.h"
#include "src/ai/simhash.h"
#include "src/core/synapse_codec.h"
#include "src/ai/pq_eval.h"
#include "src/memory/arena.h"        // PomaiArena declaration
#include "src/memory/shard_arena.h"  // ShardArena declaration

#include <new> // placement new
#include <algorithm>
#include <random>
#include <limits>
#include <cstring>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <numeric>
#include <stdexcept>
#include <cstddef> // offsetof
#include <chrono>

namespace pomai::ai::orbit
{

    // ArenaView small method implementations (dispatch)
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

    // Hàm tính khoảng cách L2 Squared cho Raw Float (Fallback & Training)
    static inline float l2sq(const float *a, const float *b, size_t dim)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    // --- Constructors: accept both arena flavors ---
    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        if (!arena_.is_pomai_arena()) throw std::invalid_argument("Arena null");
        // rest of initialization identical
        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p)) {
            std::filesystem::create_directories(p);
        }
        schema_file_path_ = (p / "pomai_schema.bin").string();

        if (std::filesystem::exists(schema_file_path_)) {
            std::clog << "[Orbit] Found existing DB. Loading schema...\n";
            if (load_schema()) {
                std::clog << "[Orbit] Restored: Dim=" << cfg_.dim << ", Centroids=" << cfg_.num_centroids << "\n";
            } else {
                std::cerr << "[Orbit] Failed to load schema!\n";
            }
        } else {
            if (cfg_.dim == 0) {
                throw std::runtime_error("[Orbit] New DB initialization requires 'dim' to be set!");
            }
            std::clog << "[Orbit] Initializing NEW Database (Dim=" << cfg_.dim << ")\n";
            save_schema();
        }

        if (std::getenv("POMAI_DISABLE_SYNAPSE")) {
            cfg_.use_synapse_4bit = false;
            std::clog << "[Orbit] !!! SYNAPSE DISABLED BY ENV VAR !!!\n";
        }

        if (cfg_.use_cortex) {
            try {
                cortex_ = std::make_unique<NetworkCortex>(7777); // Default UDP Port 7777
                cortex_->start();
                std::clog << "[Orbit] Cortex Online: Pulse active.\n";
            } catch (const std::exception& e) {
                std::cerr << "[Orbit] Failed to start Cortex: " << e.what() << "\n";
            }
        }
    }

    PomaiOrbit::PomaiOrbit(const Config &cfg, pomai::memory::ShardArena *arena)
        : cfg_(cfg), arena_(ArenaView(arena))
    {
        if (!arena_.is_shard_arena()) throw std::invalid_argument("ShardArena null");
        std::filesystem::path p(cfg_.data_path);
        if (!std::filesystem::exists(p)) {
            std::filesystem::create_directories(p);
        }
        schema_file_path_ = (p / "pomai_schema.bin").string();

        if (std::filesystem::exists(schema_file_path_)) {
            std::clog << "[Orbit] Found existing DB. Loading schema...\n";
            if (load_schema()) {
                std::clog << "[Orbit] Restored: Dim=" << cfg_.dim << ", Centroids=" << cfg_.num_centroids << "\n";
            } else {
                std::cerr << "[Orbit] Failed to load schema!\n";
            }
        } else {
            if (cfg_.dim == 0) {
                throw std::runtime_error("[Orbit] New DB initialization requires 'dim' to be set!");
            }
            std::clog << "[Orbit] Initializing NEW Database (Dim=" << cfg_.dim << ")\n";
            save_schema();
        }

        if (std::getenv("POMAI_DISABLE_SYNAPSE")) {
            cfg_.use_synapse_4bit = false;
            std::clog << "[Orbit] !!! SYNAPSE DISABLED BY ENV VAR !!!\n";
        }

        if (cfg_.use_cortex) {
            try {
                cortex_ = std::make_unique<NetworkCortex>(7777); // Default UDP Port 7777
                cortex_->start();
                std::clog << "[Orbit] Cortex Online: Pulse active.\n";
            } catch (const std::exception& e) {
                std::cerr << "[Orbit] Failed to start Cortex: " << e.what() << "\n";
            }
        }
    }

    PomaiOrbit::~PomaiOrbit() {
        if (cortex_) cortex_->stop();
    }

    // --- Persistence: Quản lý Schema ---
    void PomaiOrbit::save_schema() {
        SchemaHeader header;
        header.dim = cfg_.dim;
        header.num_centroids = cfg_.num_centroids;

        std::ofstream out(schema_file_path_, std::ios::binary);
        if (out.is_open()) {
            out.write(reinterpret_cast<const char*>(&header), sizeof(header));
            out.close();
        } else {
            std::cerr << "[Orbit] ERROR: Could not save schema to " << schema_file_path_ << "\n";
        }
    }

    bool PomaiOrbit::load_schema() {
        std::ifstream in(schema_file_path_, std::ios::binary);
        if (!in.is_open()) return false;

        SchemaHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        in.close();

        if (header.magic_number != 0x504F4D41) {
            std::cerr << "[Orbit] Invalid schema file signature!\n";
            return false;
        }

        cfg_.dim = header.dim;
        if (header.num_centroids > 0) {
            cfg_.num_centroids = header.num_centroids;
        }

        return true;
    }

    // --- Training: Học tập & Tinh chỉnh cấu trúc ---
    bool PomaiOrbit::train(const float *data, size_t n)
    {
        if (!data || n == 0) return false;

        if (cfg_.num_centroids == 0) {
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
        for(size_t i=0; i<num_c; ++i) centroids_[i] = std::make_unique<OrbitNode>();

        std::mt19937 rng(42);
        std::vector<size_t> indices;

        if (cfg_.use_kmeans_pp && n > num_c) {
            indices.clear(); indices.reserve(num_c);
            std::uniform_int_distribution<size_t> dist(0, n - 1);
            indices.push_back(dist(rng));

            std::vector<float> min_dists(n, std::numeric_limits<float>::max());
            for (size_t c = 1; c < num_c; ++c) {
                size_t last_idx = indices.back();
                const float* last_vec = data + last_idx * cfg_.dim;
                double sum_sq_dist = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    float d = l2sq(data + i * cfg_.dim, last_vec, cfg_.dim);
                    if (d < min_dists[i]) min_dists[i] = d;
                    sum_sq_dist += min_dists[i];
                }
                std::uniform_real_distribution<double> rand_dist(0.0, sum_sq_dist);
                double target = rand_dist(rng);
                double cum_sum = 0.0;
                size_t selected = 0;
                for (size_t i = 0; i < n; ++i) {
                    cum_sum += min_dists[i];
                    if (cum_sum >= target) { selected = i; break; }
                }
                indices.push_back(selected);
            }
        } else {
            indices.resize(n); std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            indices.resize(num_c);
        }

        for (size_t i = 0; i < num_c; ++i) {
            centroids_[i]->vector.resize(cfg_.dim);
            std::memcpy(centroids_[i]->vector.data(), data + indices[i] * cfg_.dim, cfg_.dim * sizeof(float));
            centroids_[i]->neighbors.reserve(cfg_.m_neighbors * 2);
        }

        // Training Iterations
        std::vector<int> counts(num_c, 0);
        std::vector<float> accum(num_c * cfg_.dim, 0.0f);
        for (int iter = 0; iter < 5; ++iter) {
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(accum.begin(), accum.end(), 0.0f);
            for (size_t i = 0; i < n; ++i) {
                const float *vec = data + i * cfg_.dim;
                uint32_t best_c = find_nearest_centroid(vec);
                counts[best_c]++;
                float *acc_ptr = &accum[best_c * cfg_.dim];
                for (size_t d = 0; d < cfg_.dim; ++d) acc_ptr[d] += vec[d];
            }
            for (size_t c = 0; c < num_c; ++c) {
                if (counts[c] > 0) {
                    float inv = 1.0f / counts[c];
                    for (size_t d = 0; d < cfg_.dim; ++d) centroids_[c]->vector[d] = accum[c * cfg_.dim + d] * inv;
                }
            }
        }

        // Build HNSW-like Graph
        for (size_t i = 0; i < num_c; ++i) {
            std::vector<std::pair<float, uint32_t>> dists;
            for (size_t j = 0; j < num_c; ++j) {
                if (i == j) continue;
                float d = l2sq(centroids_[i]->vector.data(), centroids_[j]->vector.data(), cfg_.dim);
                dists.push_back({d, static_cast<uint32_t>(j)});
            }
            std::sort(dists.begin(), dists.end());
            for (size_t k = 0; k < std::min(dists.size(), cfg_.m_neighbors); ++k) {
                centroids_[i]->neighbors.push_back(dists[k].second);
            }
        }

        // Init Components
        if (cfg_.use_fingerprint) fp_ = FingerprintEncoder::createSimHash(cfg_.dim, 512);

        if (cfg_.use_pq) {
            size_t pq_m = 16;
            while (pq_m > 1 && (cfg_.dim % pq_m != 0)) pq_m /= 2;
            pq_ = std::make_unique<ProductQuantizer>(cfg_.dim, pq_m, 256);
            pq_->train(data, n, 10);
        }

        for (size_t i = 0; i < num_c; ++i) {
            uint64_t off = alloc_new_bucket(static_cast<uint32_t>(i));
            centroids_[i]->bucket_offset.store(off, std::memory_order_release);
        }

        save_schema();
        return true;
    }

    // --- Memory Management: Cấp phát Bucket ---
    uint64_t PomaiOrbit::alloc_new_bucket(uint32_t centroid_id)
    {
        size_t cap = dynamic_bucket_capacity_;

        size_t head_sz = sizeof(BucketHeader);
        size_t fp_sz = (cfg_.use_fingerprint && fp_) ? fp_->bytes() * cap : 0;
        size_t pq_sz = (cfg_.use_pq && pq_) ? pq_->m() * cap : 0;

        size_t vec_sz = 0;
        if (cfg_.use_synapse_4bit) {
            vec_sz = pomai::core::SynapseCodec::packed_byte_size(cfg_.dim) * cap;
        } else {
            vec_sz = cfg_.dim * sizeof(float) * cap;
        }

        size_t ids_sz = sizeof(uint64_t) * cap;
        auto align64 = [](size_t s) { return (s + 63) & ~63; };

        uint32_t off_fp = static_cast<uint32_t>(align64(head_sz));
        uint32_t off_pq = static_cast<uint32_t>(align64(off_fp + fp_sz));
        uint32_t off_vec = static_cast<uint32_t>(align64(off_pq + pq_sz));
        uint32_t off_ids = static_cast<uint32_t>(align64(off_vec + vec_sz));
        size_t total_bytes = off_ids + ids_sz;

        char* blob_ptr = arena_.alloc_blob(static_cast<uint32_t>(total_bytes));
        if (!blob_ptr) return 0;

        uint64_t offset = arena_.offset_from_blob_ptr(blob_ptr);
        BucketHeader* hdr = reinterpret_cast<BucketHeader*>(blob_ptr + sizeof(uint32_t));
        // Construct in-place to properly initialize atomics / non-trivial members.
        new (hdr) BucketHeader();

        hdr->centroid_id = centroid_id;
        hdr->count.store(0, std::memory_order_relaxed);
        hdr->next_bucket_offset.store(0, std::memory_order_relaxed);

        hdr->off_fingerprints = off_fp;
        hdr->off_pq_codes = off_pq;
        hdr->off_vectors = off_vec;
        hdr->off_ids = off_ids;
        hdr->synapse_scale = 10.0f;

        hdr->is_frozen = false;
        hdr->disk_offset = 0;
        hdr->last_access_ms = 0;

        return offset;
    }

    uint32_t PomaiOrbit::find_nearest_centroid(const float *vec)
    {
        uint32_t best = 0;
        float min_d = std::numeric_limits<float>::max();
        for (size_t i = 0; i < centroids_.size(); ++i) {
            float d = l2sq(vec, centroids_[i]->vector.data(), cfg_.dim);
            if (d < min_d) { min_d = d; best = static_cast<uint32_t>(i); }
        }
        return best;
    }

    // --- Routing: find a small set of centroids to probe for a query ---
    std::vector<uint32_t> PomaiOrbit::find_routing_centroids(const float *vec, size_t n)
    {
        if (centroids_.empty()) return {};
        using NodeDist = std::pair<float, uint32_t>;
        std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> candidates;
        std::unordered_set<uint32_t> visited;

        uint32_t entry = 0;
        float d = l2sq(vec, centroids_[entry]->vector.data(), cfg_.dim);
        candidates.push({d, entry});
        visited.insert(entry);

        std::vector<uint32_t> result;
        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();
            result.push_back(current.second);
            if (result.size() >= n * 2) break;

            const auto& node = *centroids_[current.second];
            for (uint32_t neighbor_id : node.neighbors) {
                if (visited.find(neighbor_id) == visited.end()) {
                    visited.insert(neighbor_id);
                    float d_n = l2sq(vec, centroids_[neighbor_id]->vector.data(), cfg_.dim);
                    candidates.push({d_n, neighbor_id});
                }
            }
        }
        if (result.size() > n) result.resize(n);
        return result;
    }

    void PomaiOrbit::init_centroids_kmeans_pp(const float* /*data*/, size_t /*n*/, size_t /*k*/, std::vector<size_t>& /*indices*/, std::mt19937& /*rng*/) {}

    // --- Insert: Nạp dữ liệu vào "Phổi" (và cập nhật in-memory index) ---
    bool PomaiOrbit::insert(const float *vec, uint64_t label)
    {
        if (centroids_.empty()) return false;

        uint32_t cid = find_nearest_centroid(vec);
        OrbitNode& node = *centroids_[cid];
        std::unique_lock<std::shared_mutex> lock(node.mu);

        uint64_t current_off = node.bucket_offset.load(std::memory_order_acquire);
        char* bucket_ptr = nullptr;
        BucketHeader* hdr = nullptr;

        while (current_off != 0) {
            const char* blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
            if (!blob_ptr) return false;
            bucket_ptr = const_cast<char*>(blob_ptr) + sizeof(uint32_t);
            hdr = reinterpret_cast<BucketHeader*>(bucket_ptr);

            if (hdr->count.load(std::memory_order_relaxed) < dynamic_bucket_capacity_) break;

            uint64_t nb = hdr->next_bucket_offset.load(std::memory_order_acquire);
            if (nb == 0) {
                uint64_t new_off = alloc_new_bucket(cid);
                hdr->next_bucket_offset.store(new_off, std::memory_order_release);
                current_off = new_off;
            } else {
                current_off = nb;
            }
        }

        if (!hdr) return false;

        uint32_t idx = hdr->count.load(std::memory_order_relaxed);

        if (cfg_.use_fingerprint && fp_) {
            uint8_t* fp_base = reinterpret_cast<uint8_t*>(bucket_ptr + hdr->off_fingerprints);
            fp_->compute(vec, fp_base + idx * fp_->bytes());
        }
        if (cfg_.use_pq && pq_) {
            uint8_t* pq_base = reinterpret_cast<uint8_t*>(bucket_ptr + hdr->off_pq_codes);
            pq_->encode(vec, pq_base + idx * pq_->m());
        }

        if (cfg_.use_synapse_4bit)
        {
            uint8_t* vec_base = reinterpret_cast<uint8_t*>(bucket_ptr + hdr->off_vectors);
            size_t vec_byte_size = pomai::core::SynapseCodec::packed_byte_size(cfg_.dim);
            uint8_t* target = vec_base + idx * vec_byte_size;

            pomai::core::SynapseCodec::pack_4bit_delta(
                cfg_.dim,
                vec,
                node.vector.data(),
                hdr->synapse_scale,
                target
            );
        }
        else
        {
            float* vec_base = reinterpret_cast<float*>(bucket_ptr + hdr->off_vectors);
            std::memcpy(vec_base + idx * cfg_.dim, vec, cfg_.dim * sizeof(float));
        }

        uint64_t* id_base = reinterpret_cast<uint64_t*>(bucket_ptr + hdr->off_ids);
        pomai::ai::atomic_utils::atomic_store_u64(id_base + idx, label);

        // Update in-memory label -> bucket offset map
        {
            std::unique_lock<std::shared_mutex> lm(label_map_mu_);
            // current_off is the blob offset that contains this bucket
            label_to_bucket_[label] = current_off;
        }
        // If previously soft-deleted, clear
        {
            std::unique_lock<std::shared_mutex> dm(del_mu_);
            auto it = deleted_labels_.find(label);
            if (it != deleted_labels_.end()) deleted_labels_.erase(it);
        }

        hdr->count.fetch_add(1, std::memory_order_release);
        return true;
    }

    // --- Get by label (used for training / VGET) ---
    bool PomaiOrbit::get(uint64_t label, std::vector<float>& out_vec)
    {
        // check soft-deleted
        {
            std::shared_lock<std::shared_mutex> dm(del_mu_);
            if (deleted_labels_.count(label)) return false;
        }

        uint64_t bucket_off = 0;
        {
            std::shared_lock<std::shared_mutex> lm(label_map_mu_);
            auto it = label_to_bucket_.find(label);
            if (it == label_to_bucket_.end()) return false;
            bucket_off = it->second;
        }
        if (bucket_off == 0) return false;

        const char* ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(bucket_off);
        const char* data_base_ptr = nullptr;
        std::vector<char> temp_buffer;

        bool from_remote = false;
        if (!ram_blob_ptr) {
            temp_buffer = arena_.read_remote_blob(bucket_off);
            if (temp_buffer.empty()) return false;
            data_base_ptr = temp_buffer.data() + sizeof(uint32_t);
            from_remote = true;
        } else {
            data_base_ptr = ram_blob_ptr + sizeof(uint32_t);
        }

        // Read header fields safely: if from_remote, use memcpy to POD locals.
        uint32_t off_fingerprints = 0, off_vectors = 0, off_ids = 0, centroid_id = 0;
        uint32_t count = 0;
        float synapse_scale = 0.0f;

        if (from_remote) {
            // safe POD reads from temp buffer (don't construct atomic)
            std::memcpy(&centroid_id, data_base_ptr + offsetof(BucketHeader, centroid_id), sizeof(centroid_id));
            std::memcpy(&count, data_base_ptr + offsetof(BucketHeader, count), sizeof(count));
            std::memcpy(&off_fingerprints, data_base_ptr + offsetof(BucketHeader, off_fingerprints), sizeof(off_fingerprints));
            std::memcpy(&off_vectors, data_base_ptr + offsetof(BucketHeader, off_vectors), sizeof(off_vectors));
            std::memcpy(&off_ids, data_base_ptr + offsetof(BucketHeader, off_ids), sizeof(off_ids));
            std::memcpy(&synapse_scale, data_base_ptr + offsetof(BucketHeader, synapse_scale), sizeof(synapse_scale));
        } else {
            const BucketHeader* hdr_ptr = reinterpret_cast<const BucketHeader*>(data_base_ptr);
            centroid_id = hdr_ptr->centroid_id;
            count = hdr_ptr->count.load(std::memory_order_acquire);
            off_fingerprints = hdr_ptr->off_fingerprints;
            off_vectors = hdr_ptr->off_vectors;
            off_ids = hdr_ptr->off_ids;
            synapse_scale = hdr_ptr->synapse_scale;
        }

        if (count == 0) return false;

        const uint64_t* id_base = reinterpret_cast<const uint64_t*>(data_base_ptr + off_ids);

        int32_t found_idx = -1;
        for (uint32_t i = 0; i < count; ++i) {
            uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
            if (id == label) { found_idx = static_cast<int32_t>(i); break; }
        }
        if (found_idx < 0) return false;

        out_vec.assign(cfg_.dim, 0.0f);
        const char* vec_data_base = data_base_ptr + off_vectors;

        if (cfg_.use_synapse_4bit) {
            size_t packed_sz = pomai::core::SynapseCodec::packed_byte_size(cfg_.dim);
            const uint8_t* packed_ptr = reinterpret_cast<const uint8_t*>(vec_data_base) + found_idx * packed_sz;
            const float* centroid = centroids_[centroid_id]->vector.data();
            float inv_scale = 1.0f / synapse_scale;
            for (size_t d = 0; d < cfg_.dim; ++d) {
                size_t byte_idx = d / 2;
                uint8_t byte = packed_ptr[byte_idx];
                uint8_t nibble = (d % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                int delta = static_cast<int>(nibble) - 8; // map 0..15 -> -8..7
                out_vec[d] = centroid[d] + (static_cast<float>(delta) * inv_scale);
            }
        } else {
            const float* fptr = reinterpret_cast<const float*>(vec_data_base) + found_idx * cfg_.dim;
            std::memcpy(out_vec.data(), fptr, cfg_.dim * sizeof(float));
        }

        return true;
    }

    // --- Soft remove label ---
    bool PomaiOrbit::remove(uint64_t label)
    {
        std::unique_lock<std::shared_mutex> dm(del_mu_);
        deleted_labels_.insert(label);
        return true;
    }

    // --- Search: same as before but skip soft-deleted labels ---
    std::vector<std::pair<uint64_t, float>> PomaiOrbit::search(const float *query, size_t k, size_t nprobe)
    {
        if (centroids_.empty()) return {};

        if (nprobe == 0) {
            nprobe = std::max(1UL, cfg_.num_centroids / 50);
            if (k > 50) nprobe *= 2;
        }

        std::vector<uint32_t> targets = find_routing_centroids(query, nprobe);

        std::vector<uint8_t> qfp;
        if (cfg_.use_fingerprint && fp_) {
            qfp.resize(fp_->bytes());
            fp_->compute(query, qfp.data());
        }

        std::vector<float> lut_buffer;
        if (cfg_.use_synapse_4bit) {
            lut_buffer.resize(cfg_.dim * 16);
        }

        constexpr size_t BATCH_SIZE = 128;
        size_t packed_vec_size = cfg_.use_synapse_4bit ? pomai::core::SynapseCodec::packed_byte_size(cfg_.dim) : 0;

        std::vector<uint8_t> batch_packed_data;
        if (cfg_.use_synapse_4bit) batch_packed_data.resize(BATCH_SIZE * packed_vec_size);

        std::vector<uint64_t> batch_ids(BATCH_SIZE);
        std::vector<float> batch_dists(BATCH_SIZE);
        size_t batch_count = 0;

        using ResPair = std::pair<float, uint64_t>;
        std::priority_queue<ResPair> candidates;

        auto flush_batch = [&](size_t n) {
            if (n == 0) return;
            pomai::ai::pq_approx_dist_batch_packed4(
                lut_buffer.data(), cfg_.dim, 16,
                batch_packed_data.data(), n, batch_dists.data()
            );

            for (size_t i = 0; i < n; ++i) {
                float d = batch_dists[i];
                uint64_t id = batch_ids[i];
                {
                    std::shared_lock<std::shared_mutex> dm(del_mu_);
                    if (deleted_labels_.count(id)) continue;
                }
                if (candidates.size() < k) {
                    candidates.push({d, id});
                } else if (d < candidates.top().first) {
                    candidates.pop();
                    candidates.push({d, id});
                }
            }
        };

        for (uint32_t cid : targets)
        {
            uint64_t current_off = centroids_[cid]->bucket_offset.load(std::memory_order_acquire);

            while (current_off != 0)
            {
                const char* ram_blob_ptr = arena_.blob_ptr_from_offset_for_map(current_off);
                if (!ram_blob_ptr) break;
                const char* bucket_base = ram_blob_ptr + sizeof(uint32_t);

                const BucketHeader* hdr_ptr = reinterpret_cast<const BucketHeader*>(bucket_base);

                uint32_t count = hdr_ptr->count.load(std::memory_order_acquire);
                uint64_t next_bucket_offset = hdr_ptr->next_bucket_offset.load(std::memory_order_acquire);

                uint32_t off_fingerprints = hdr_ptr->off_fingerprints;
                uint32_t off_vectors = hdr_ptr->off_vectors;
                uint32_t off_ids = hdr_ptr->off_ids;
                float synapse_scale = hdr_ptr->synapse_scale;
                bool is_frozen = hdr_ptr->is_frozen;
                uint64_t disk_offset = hdr_ptr->disk_offset;

                if (count == 0) { current_off = next_bucket_offset; continue; }

                const char* data_base_ptr = nullptr;
                std::vector<char> temp_buffer;

                if (is_frozen) {
                    temp_buffer = arena_.read_remote_blob(disk_offset);
                    if (temp_buffer.empty()) {
                        current_off = next_bucket_offset;
                        continue;
                    }
                    data_base_ptr = temp_buffer.data() + sizeof(uint32_t);
                    // read header fields from temp buffer safely
                    uint32_t tmp_count = 0;
                    std::memcpy(&tmp_count, data_base_ptr + offsetof(BucketHeader, count), sizeof(tmp_count));
                    count = tmp_count;
                    std::memcpy(&off_fingerprints, data_base_ptr + offsetof(BucketHeader, off_fingerprints), sizeof(off_fingerprints));
                    std::memcpy(&off_vectors, data_base_ptr + offsetof(BucketHeader, off_vectors), sizeof(off_vectors));
                    std::memcpy(&off_ids, data_base_ptr + offsetof(BucketHeader, off_ids), sizeof(off_ids));
                    std::memcpy(&synapse_scale, data_base_ptr + offsetof(BucketHeader, synapse_scale), sizeof(synapse_scale));
                } else {
                    data_base_ptr = bucket_base;
                }

                if (cfg_.use_synapse_4bit) {
                    if (batch_count > 0) { flush_batch(batch_count); batch_count = 0; }
                    pomai::core::SynapseCodec::precompute_search_lut(
                        cfg_.dim, query, centroids_[cid]->vector.data(),
                        synapse_scale, lut_buffer.data()
                    );
                }

                const uint8_t* fp_base = reinterpret_cast<const uint8_t*>(data_base_ptr + off_fingerprints);
                const void* vec_data_base = reinterpret_cast<const void*>(data_base_ptr + off_vectors);
                const uint64_t* id_base = reinterpret_cast<const uint64_t*>(data_base_ptr + off_ids);

                for (uint32_t i = 0; i < count; ++i)
                {
                    uint64_t id = pomai::ai::atomic_utils::atomic_load_u64(id_base + i);
                    {
                        std::shared_lock<std::shared_mutex> dm(del_mu_);
                        if (deleted_labels_.count(id)) continue;
                    }

                    if (cfg_.use_fingerprint && fp_) {
                        int dist_hamming = pomai::ai::simhash::hamming_dist(qfp.data(), fp_base + i * fp_->bytes(), fp_->bytes());
                        if (dist_hamming > 140) continue;
                    }

                    if (cfg_.use_synapse_4bit) {
                        const uint8_t* src = static_cast<const uint8_t*>(vec_data_base) + i * packed_vec_size;
                        std::memcpy(&batch_packed_data[batch_count * packed_vec_size], src, packed_vec_size);
                        batch_ids[batch_count] = id;
                        batch_count++;
                        if (batch_count == BATCH_SIZE) { flush_batch(BATCH_SIZE); batch_count = 0; }
                    } else {
                        const float* raw_vec = static_cast<const float*>(vec_data_base) + i * cfg_.dim;
                        float d = l2sq(query, raw_vec, cfg_.dim);
                        if (candidates.size() < k) candidates.push({d, id});
                        else if (d < candidates.top().first) { candidates.pop(); candidates.push({d, id}); }
                    }
                }

                current_off = next_bucket_offset;
            }
        }

        if (cfg_.use_synapse_4bit && batch_count > 0) flush_batch(batch_count);

        std::vector<std::pair<uint64_t, float>> final_out;
        final_out.reserve(candidates.size());
        while (!candidates.empty()) {
            auto p = candidates.top(); candidates.pop();
            final_out.emplace_back(p.second, p.first);
        }
        std::reverse(final_out.begin(), final_out.end());
        return final_out;
    }

    bool PomaiOrbit::save_routing(const std::string& /*path*/) { return false; }
    bool PomaiOrbit::load_routing(const std::string& /*path*/) { return false; }

} // namespace pomai::ai::orbit