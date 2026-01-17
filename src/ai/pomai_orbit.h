#pragma once
/*
 * src/ai/pomai_orbit.h
 *
 * PomaiOrbit public header (updated to integrate WhisperGrain controller + Thermal Thermodynamics).
 *
 * Notes:
 * - Added Thermal Gating logic: Centroids now have temperature.
 * - Added apply_thermal_policy() for background memory management (madvise).
 * - This enables the "Active Working Set" architecture where cold data is implicitly demoted.
 * - [NEW] Added train_mu_ mutex to fix Race Condition in Auto-Train.
 */

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
#include "src/core/config.h"
#include "src/ai/network_cortex.h"
#include "src/core/metadata_index.h"
#include "src/ai/eternalecho_quantizer.h"
#include "src/ai/whispergrain.h"

// Forward-declare both arena flavors to avoid heavy includes here.
namespace pomai::memory
{
    class PomaiArena;
    class ShardArena;
}

namespace pomai::ai::orbit
{
    // Magic number (ASCII 'POMA' = 0x504F4D41)
    struct SchemaHeader
    {
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

        uint32_t off_fingerprints; // unused now but kept for compatibility
        uint32_t off_pq_codes;     // repurposed: per-entry code length array (uint16_t per slot)
        uint32_t off_vectors;      // raw bytes area to store serialized EchoCode per entry
        uint32_t off_ids;
        float synapse_scale; // kept for compatibility; not used for EEQ

        bool is_frozen;
        uint64_t disk_offset;
        uint64_t last_access_ms;
    };

    struct OrbitNode
    {
        std::vector<float> vector;
        std::vector<uint32_t> neighbors;
        std::atomic<uint64_t> bucket_offset{0};
        std::shared_mutex mu;
    };

    // Small adapter that lets PomaiOrbit call into either PomaiArena or ShardArena
    struct ArenaView
    {
        ArenaView() : pa(nullptr), sa(nullptr) {}
        explicit ArenaView(pomai::memory::PomaiArena *a) : pa(a), sa(nullptr) {}
        explicit ArenaView(pomai::memory::ShardArena *s) : pa(nullptr), sa(s) {}

        char *alloc_blob(uint32_t len) const;
        uint64_t offset_from_blob_ptr(const char *p) const noexcept;
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const;
        std::vector<char> read_remote_blob(uint64_t remote_id) const;

        // [New] Support for demoting pages (madvise DONTNEED)
        void demote_range(uint64_t offset, size_t len) const;

        bool is_pomai_arena() const { return pa != nullptr; }
        bool is_shard_arena() const { return sa != nullptr; }

    private:
        pomai::memory::PomaiArena *pa;
        pomai::memory::ShardArena *sa;
    };

    // Lightweight introspection structure describing a membrance.
    // Best-effort snapshot only (may be approximate for disk bytes).
    struct MembranceInfo
    {
        size_t dim = 0;
        size_t num_vectors = 0;
        uint64_t disk_bytes = 0;

        double disk_gb() const noexcept
        {
            return static_cast<double>(disk_bytes) / (1024.0 * 1024.0 * 1024.0);
        }
    };

    class PomaiOrbit
    {
    public:
        struct Config
        {
            size_t dim = 0;
            std::string data_path = "./data";

            pomai::config::OrbitConfig algo;

            // EternalEcho quantizer config will be used exclusively
            pomai::ai::EternalEchoConfig eeq_cfg;

            // Network Config
            pomai::config::NetworkCortexConfig cortex_cfg;
            bool use_cortex = true;
        };

        struct DeletedBloom
        {
            static constexpr uint32_t kBits = 1 << 16; // 65536 bits = 8KB
            static constexpr uint32_t kMask = kBits - 1;

            alignas(64) uint8_t bits[kBits / 8]{};

            static inline uint32_t h1(uint64_t x)
            {
                x ^= x >> 33;
                x *= 0xff51afd7ed558ccdULL;
                return uint32_t(x);
            }
            static inline uint32_t h2(uint64_t x)
            {
                x ^= x >> 29;
                x *= 0xc4ceb9fe1a85ec53ULL;
                return uint32_t(x);
            }

            inline void add(uint64_t v)
            {
                uint32_t a = h1(v) & kMask;
                uint32_t b = h2(v) & kMask;
                uint32_t c = (a ^ b) & kMask;
                bits[a >> 3] |= 1u << (a & 7);
                bits[b >> 3] |= 1u << (b & 7);
                bits[c >> 3] |= 1u << (c & 7);
            }

            inline bool maybe_contains(uint64_t v) const
            {
                uint32_t a = h1(v) & kMask;
                uint32_t b = h2(v) & kMask;
                uint32_t c = (a ^ b) & kMask;
                return (bits[a >> 3] & (1u << (a & 7))) &&
                       (bits[b >> 3] & (1u << (b & 7))) &&
                       (bits[c >> 3] & (1u << (c & 7)));
            }
        };

        // Constructors: accept either arena flavor
        PomaiOrbit(const Config &cfg, pomai::memory::PomaiArena *arena);
        PomaiOrbit(const Config &cfg, pomai::memory::ShardArena *arena);
        ~PomaiOrbit();

        bool train(const float *data, size_t n);
        bool insert(const float *vec, uint64_t label);
        bool insert_batch(const std::vector<std::pair<uint64_t, std::vector<float>>> &batch);
        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k, size_t nprobe = 0);

        // Random-access get/remove by label
        bool get(uint64_t label, std::vector<float> &out_vec);
        bool remove(uint64_t label);

        // Search restricted to candidate set (pre-filtered by metadata)
        std::vector<std::pair<uint64_t, float>> search_filtered(const float *query, size_t k, const std::vector<uint64_t> &candidates);

        // ------------------ WhisperGrain / Budget-aware APIs ------------------
        // Attach an external WhisperGrain controller (optional). If not attached,
        // callers may still call the budget-aware search APIs directly.
        void set_whisper_grain(std::shared_ptr<pomai::ai::WhisperGrain> wg) { whisper_ctrl_ = std::move(wg); }
        std::shared_ptr<pomai::ai::WhisperGrain> whisper_grain() const { return whisper_ctrl_; }

        // Budget-aware search: caller supplies a pomai::ai::Budget (computed by WhisperGrain)
        std::vector<std::pair<uint64_t, float>> search_with_budget(
            const float *query, size_t k, const pomai::ai::Budget &budget, size_t nprobe = 0);

        std::vector<std::pair<uint64_t, float>> search_filtered_with_budget(
            const float *query, size_t k, const std::vector<uint64_t> &candidates, const pomai::ai::Budget &budget);

        // --------------------------------------------------------------------

        void save_schema();
        bool load_schema();
        bool save_routing(const std::string &path);
        bool load_routing(const std::string &path);

        void set_metadata_index(std::shared_ptr<pomai::core::MetadataIndex> idx) { metadata_index_ = std::move(idx); }
        std::shared_ptr<pomai::core::MetadataIndex> metadata_index() const { return metadata_index_; }

        // Introspection: best-effort snapshot info about this membrance
        MembranceInfo get_info() const;

        // ------------------ [NEW] Thermal Thermodynamics API ------------------
        // Called by background worker to scan thermal map and evict cold pages (madvise DONTNEED).
        void apply_thermal_policy();
        std::vector<uint64_t> get_centroid_ids(uint32_t cid) const;
        size_t num_centroids() const { return centroids_.size(); }

    private:
        Config cfg_;
        ArenaView arena_;

        // [FIXED] Mutex for Thread-Safe Auto-Train (Double-Checked Locking)
        std::mutex train_mu_;

        std::unique_ptr<NetworkCortex> cortex_;

        std::vector<std::unique_ptr<OrbitNode>> centroids_;

        uint32_t dynamic_bucket_capacity_ = 128;
        std::string schema_file_path_;

        // Label map: sharded maps to reduce global lock contention on label->bucket/slot updates.
        static constexpr size_t kLabelShardBits = 6; // 2^6 = 64 shards
        static constexpr size_t kLabelShardCount = 1u << kLabelShardBits;

        struct LabelShard
        {
            mutable std::shared_mutex mu;
            std::unordered_map<uint64_t, uint64_t> bucket;
            std::unordered_map<uint64_t, uint32_t> slot;
        };
        std::array<LabelShard, kLabelShardCount> label_shards_;

        // Helper: compute shard index for a label
        static inline size_t label_shard_index(uint64_t label) noexcept
        {
            // 64-bit multiplicative hash, then take top bits
            const uint64_t kMul = 11400714819323198485ull;
            return static_cast<size_t>((label * kMul) >> (64 - kLabelShardBits));
        }

        // Helper accessors for the sharded label map
        void set_label_map(uint64_t label, uint64_t bucket_off, uint32_t slot);
        bool get_label_bucket(uint64_t label, uint64_t &out_bucket) const;
        bool get_label_slot(uint64_t label, uint32_t &out_slot) const;

        std::unordered_set<uint64_t> deleted_labels_;
        mutable std::shared_mutex del_mu_;

        // EternalEcho quantizer (single shared instance per orbit)
        std::unique_ptr<pomai::ai::EternalEchoQuantizer> eeq_;

        // Optional shared metadata index (inverted index)
        std::shared_ptr<pomai::core::MetadataIndex> metadata_index_;

        // Optional WhisperGrain controller (shared across orbits/DB)
        std::shared_ptr<pomai::ai::WhisperGrain> whisper_ctrl_;

        // ------------------ [NEW] Thermal Thermodynamics Engine ------------------

        // Thermal Map: Structure of Arrays (SoA).
        // Index i corresponds to centroids_[i].
        // Value: 0 (Frozen/Cold) -> 255 (Red Hot).
        // Fits in L2 Cache for instant checks.
        std::vector<std::atomic<uint8_t>> thermal_map_;

        // Epoch (Seconds) of last access. Used for lazy decay calculation.
        std::vector<std::atomic<uint32_t>> last_access_epoch_;

        // Helper: Initialize thermal maps when centroids are created/loaded
        void init_thermal_map(size_t num_centroids);

        // Helper: Touch a centroid -> Increases temperature, updates timestamp
        void touch_centroid(uint32_t cid);

        // Helper: Get current temperature (calculates lazy decay based on time elapsed)
        uint8_t get_temperature(uint32_t cid) const;

        // --------------------------------------------------------------------------
        uint32_t find_nearest_centroid(const float *vec);
        std::vector<uint32_t> find_routing_centroids(const float *vec, size_t n);
        uint64_t alloc_new_bucket(uint32_t centroid_id);
        // Helper: compute distance by decoding EEQ bytes for given id
        bool compute_distance_for_id(const float *query, uint64_t id, float &out_dist);

        // NEW: helper that accepts precomputed projections to avoid recomputing per-id.
        bool compute_distance_for_id_with_proj(const std::vector<std::vector<float>> &qproj, float qnorm2, uint64_t id, float &out_dist);
    };
} // namespace pomai::ai::orbit