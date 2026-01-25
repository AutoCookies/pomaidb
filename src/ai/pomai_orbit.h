#pragma once

#include <vector>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <array>

#include "src/ai/fingerprint.h"
#include "src/ai/ids_block.h"
#include "src/core/config.h"
#include "src/ai/network_cortex.h"
#include "src/core/metadata_index.h"
#include "src/ai/zeroharmony_pack.h"
#include "src/ai/whispergrain.h"
#include "src/memory/wal_manager.h"
#include "src/core/algo/echo_graph.h"

namespace pomai::memory
{
    class PomaiArena;
    class ShardArena;
}

namespace pomai::ai::orbit
{
    struct SchemaHeader
    {
        uint32_t magic_number = 0x504F4D41;
        uint32_t version = 2;
        uint64_t dim;
        uint64_t num_centroids;
        uint64_t total_vectors;
    };

    struct BucketHeader
    {
        uint32_t centroid_id;
        uint32_t count;
        uint64_t next_bucket_offset;
        uint32_t off_mean;
        uint32_t off_fingerprints;
        uint32_t off_pq_codes;
        uint32_t off_vectors;
        uint32_t off_ids;
        float synapse_scale;
        bool is_frozen;
        uint64_t disk_offset;
        uint64_t last_access_ms;
    };
    static_assert(std::is_trivially_copyable_v<BucketHeader>, "BucketHeader must be POD for disk IO");

    struct OrbitNode
    {
        std::vector<float> vector;
        std::vector<uint32_t> neighbors;
        std::atomic<uint64_t> bucket_offset{0};
        std::shared_mutex mu;
    };

    struct ArenaView
    {
        ArenaView() : pa(nullptr), sa(nullptr) {}
        explicit ArenaView(pomai::memory::PomaiArena *a) : pa(a), sa(nullptr) {}
        explicit ArenaView(pomai::memory::ShardArena *s) : pa(nullptr), sa(s) {}

        char *alloc_blob(uint32_t len) const;
        uint64_t offset_from_blob_ptr(const char *p) const noexcept;
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const;
        std::vector<char> read_remote_blob(uint64_t remote_id) const;
        void demote_range(uint64_t offset, size_t len) const;

        bool is_pomai_arena() const { return pa != nullptr; }
        bool is_shard_arena() const { return sa != nullptr; }

    private:
        pomai::memory::PomaiArena *pa;
        pomai::memory::ShardArena *sa;
    };

    struct MembranceInfo
    {
        size_t dim = 0;
        size_t num_vectors = 0;
        uint64_t disk_bytes = 0;
        double disk_gb() const noexcept { return static_cast<double>(disk_bytes) / (1024.0 * 1024.0 * 1024.0); }
    };

    class PomaiOrbit
    {
    public:
        struct Config
        {
            size_t dim = 0;
            std::string data_path = "./data";
            pomai::config::OrbitConfig algo;
            pomai::config::ZeroHarmonyConfig zero_harmony_cfg;
            pomai::config::NetworkCortexConfig cortex_cfg;
            bool use_cortex = true;
        };

        PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena);
        PomaiOrbit(const Config &cfg, pomai::memory::ShardArena *arena);
        ~PomaiOrbit();

        bool train(const float *data, size_t n);
        bool insert(const float *vec, uint64_t label);
        bool insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch);

        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k, size_t nprobe = 0);
        std::vector<std::pair<uint64_t, float>> search_with_budget(const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe = 0);

        bool get(uint64_t label, std::vector<float> &out_vec);
        bool remove(uint64_t label);

        void set_whisper_grain(std::shared_ptr<pomai::ai::WhisperGrain> wg) { whisper_ctrl_ = std::move(wg); }
        std::shared_ptr<pomai::ai::WhisperGrain> whisper_grain() const { return whisper_ctrl_; }

        void save_schema();
        bool load_schema();

        void set_metadata_index(std::shared_ptr<pomai::core::MetadataIndex> idx) { metadata_index_ = std::move(idx); }
        std::shared_ptr<pomai::core::MetadataIndex> metadata_index() const { return metadata_index_; }

        MembranceInfo get_info() const;
        void apply_thermal_policy();
        std::vector<uint64_t> get_centroid_ids(uint32_t cid) const;
        size_t num_centroids() const { return centroids_.size(); }
        std::vector<uint64_t> get_all_labels() const;
        bool get_vectors_raw(const std::vector<uint64_t> &ids, std::vector<std::string> &outs) const;
        bool checkpoint();
        size_t packed_slot_size_ = 512;

        void build_echo_graph(float beta = 1.0f, float threshold = 0.5f);

        static constexpr uint32_t K_SPLIT_THRESHOLD = 2048;

    private:
        bool insert_batch_memory_only(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch);
        void recover_from_wal();
        void rebuild_index();
        void check_and_split_bucket(uint32_t cid);

        std::vector<uint32_t> bucket_sizes_;

        Config cfg_;
        ArenaView arena_;

        std::mutex train_mu_;
        std::unique_ptr<pomai::memory::WalManager> wal_;
        std::unique_ptr<NetworkCortex> cortex_;
        std::vector<std::unique_ptr<OrbitNode>> centroids_;

        uint32_t dynamic_bucket_capacity_ = 128;
        std::string schema_file_path_;

        static constexpr size_t kLabelShardBits = 6;
        static constexpr size_t kLabelShardCount = 1u << kLabelShardBits;

        struct LabelShard
        {
            mutable std::shared_mutex mu;
            std::unordered_map<uint64_t, uint64_t> bucket;
            std::unordered_map<uint64_t, uint32_t> slot;
        };
        std::array<LabelShard, kLabelShardCount> label_shards_;

        static inline size_t label_shard_index(uint64_t label) noexcept
        {
            const uint64_t kMul = 11400714819323198485ull;
            return static_cast<size_t>((label * kMul) >> (64 - kLabelShardBits));
        }

        void set_label_map(uint64_t label, uint64_t bucket_off, uint32_t slot);
        bool get_label_bucket(uint64_t label, uint64_t &out_bucket) const;
        bool get_label_slot(uint64_t label, uint32_t &out_slot) const;

        std::unordered_set<uint64_t> deleted_labels_;
        mutable std::shared_mutex del_mu_;

        std::unique_ptr<pomai::ai::ZeroHarmonyPacker> zeroharmony_;
        std::shared_ptr<pomai::core::MetadataIndex> metadata_index_;
        std::shared_ptr<pomai::ai::WhisperGrain> whisper_ctrl_;

        std::vector<std::atomic<uint8_t>> thermal_map_;
        std::vector<std::atomic<uint32_t>> last_access_epoch_;

        void init_thermal_map(size_t num_centroids);
        void touch_centroid(uint32_t cid);
        uint8_t get_temperature(uint32_t cid) const;

        uint32_t find_nearest_centroid(const float *vec);
        std::vector<uint32_t> find_routing_centroids(const float *vec, size_t n);
        uint64_t alloc_new_bucket(uint32_t centroid_id);

        bool compute_distance_for_id_with_proj(const std::vector<std::vector<float>> &qproj, float qnorm2, uint64_t id, float &out_dist);
        mutable std::shared_mutex checkpoint_mu_;
        std::vector<uint64_t> split_last_ts_;
        std::mutex split_ts_mu_;

        pomai::core::algo::EchoGraph echo_graph_;
        void scan_bucket_blitz_avx2(const float *query, uint32_t cid,
                                    std::priority_queue<std::pair<float, uint64_t>> &heap,
                                    size_t &scanned, size_t limit, size_t keep_k) const;

        void auto_robust_build();
        float compute_harmony_weight(uint32_t i, uint32_t j) const;
        uint32_t hamming_threshold_ = 32;
        std::vector<std::vector<pomai::core::algo::EchoEdge>> adj_snapshot_;
        size_t next_graph_snapshot_cid_ = 0;
        std::mutex echo_graph_bg_mu_;
        uint32_t last_centroid_dirty_ = 0;

        // Helper: apply a persisted delete (used by runtime remove and WAL replay)
        void apply_persisted_delete(uint64_t label);
    };
}