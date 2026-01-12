#pragma once

#include <vector>
#include <atomic>
#include <shared_mutex>
#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "src/ai/fingerprint.h"
#include "src/ai/pq.h"
#include "src/core/config.h"
#include "src/core/synapse_codec.h"
#include "src/ai/network_cortex.h" // [NEW] Hệ thần kinh

// Forward-declare both arena flavors to avoid heavy includes here.
// PomaiArena is the original mmap-backed arena; ShardArena is the newer per-shard bump allocator.
namespace pomai::memory { class PomaiArena; class ShardArena; }

namespace pomai::ai::orbit
{
    // [FIXED] Magic Number hợp lệ (ASCII 'POMA' = 0x504F4D41)
    struct SchemaHeader {
        uint32_t magic_number = 0x504F4D41;
        uint32_t version = 1;
        uint64_t dim;
        uint64_t num_centroids;
        uint64_t total_vectors;
    };

    struct BucketHeader
    {
        uint32_t centroid_id;
        std::atomic<uint32_t> count;

        // next bucket offset (atomic so readers can traverse lock-free)
        std::atomic<uint64_t> next_bucket_offset;

        uint32_t off_fingerprints;
        uint32_t off_pq_codes;
        uint32_t off_vectors;
        uint32_t off_ids;
        float synapse_scale;

        // Freeze/defrost bookkeeping (added fields)
        // If is_frozen == true then payload was demoted; disk_offset holds remote id (encoded with MSB set)
        bool is_frozen;
        uint64_t disk_offset;
        uint64_t last_access_ms;
    };

    struct OrbitNode
    {
        std::vector<float> vector;
        std::vector<uint32_t> neighbors;
        std::atomic<uint64_t> bucket_offset{0}; // atomic so readers can snapshot lock-free
        std::shared_mutex mu;            // writers keep mutex; readers do not take it
    };

    // Small adapter that lets PomaiOrbit call into either PomaiArena or ShardArena
    // without forcing a single concrete type across the entire codebase.
    struct ArenaView
    {
        ArenaView() : pa(nullptr), sa(nullptr) {}
        explicit ArenaView(pomai::memory::PomaiArena *a) : pa(a), sa(nullptr) {}
        explicit ArenaView(pomai::memory::ShardArena *s) : pa(nullptr), sa(s) {}

        char *alloc_blob(uint32_t len) const;
        uint64_t offset_from_blob_ptr(const char *p) const noexcept;
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const;
        std::vector<char> read_remote_blob(uint64_t remote_id) const;

        // helpers to test which is set
        bool is_pomai_arena() const { return pa != nullptr; }
        bool is_shard_arena() const { return sa != nullptr; }

    private:
        pomai::memory::PomaiArena *pa;
        pomai::memory::ShardArena *sa;
    };

    class PomaiOrbit
    {
    public:
        struct Config
        {
            size_t dim = 0;
            std::string data_path = "./data";

            size_t num_centroids = 0;
            size_t m_neighbors = 16;
            bool use_synapse_4bit = true;

            bool use_pq = true;
            bool use_fingerprint = true;
            bool use_kmeans_pp = true;
            bool adaptive_probe = true;

            // [NETWORK] Có bật tính năng sinh học không?
            bool use_cortex = true;
        };

        // Constructors: accept either arena flavor
        PomaiOrbit(const Config& cfg, pomai::memory::PomaiArena* arena);
        PomaiOrbit(const Config& cfg, pomai::memory::ShardArena* arena);
        ~PomaiOrbit();

        bool train(const float* data, size_t n);
        bool insert(const float* vec, uint64_t label);
        std::vector<std::pair<uint64_t, float>> search(const float* query, size_t k, size_t nprobe = 0);

        // New: random-access get/remove by label for training & lifecycle ops
        bool get(uint64_t label, std::vector<float>& out_vec);
        bool remove(uint64_t label);

        void save_schema();
        bool load_schema();
        bool save_routing(const std::string& path);
        bool load_routing(const std::string& path);

    private:
        Config cfg_;
        ArenaView arena_;                      // unified view to underlying arena
        std::unique_ptr<ProductQuantizer> pq_;
        std::unique_ptr<FingerprintEncoder> fp_;
        std::vector<std::unique_ptr<OrbitNode>> centroids_;

        std::unique_ptr<NetworkCortex> cortex_;

        uint32_t dynamic_bucket_capacity_ = 128;
        std::string schema_file_path_;

        // In-memory index: Label -> blob offset (fast VGET)
        std::unordered_map<uint64_t, uint64_t> label_to_bucket_;
        mutable std::shared_mutex label_map_mu_;

        // Soft-deleted labels set
        std::unordered_set<uint64_t> deleted_labels_;
        mutable std::shared_mutex del_mu_;

        uint32_t find_nearest_centroid(const float* vec);
        std::vector<uint32_t> find_routing_centroids(const float* vec, size_t n);
        uint64_t alloc_new_bucket(uint32_t centroid_id);
        void init_centroids_kmeans_pp(const float* data, size_t n, size_t k, std::vector<size_t>& indices, std::mt19937& rng);
    };
}