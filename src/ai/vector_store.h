// Updated: add ProductQuantizer member and packed4 bookkeeping for PQ integration
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <chrono>

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/pomai_hnsw.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/ai/pp_ivf.h"

// shard support: include the manager + router + pps manager
#include "src/core/shard_manager.h"
#include "src/core/shard_router.h"
#include "src/core/pps_manager.h"
#include "src/ai/candidate_collector.h"
#include "src/ai/codebooks.h"
#include "src/ai/pq_eval.h"

namespace pomai::ai
{
    // Forward-declare FingerprintEncoder to avoid including fingerprint.h in this header.
    class FingerprintEncoder;

    // Forward-declare ProductQuantizer for PQ support (complete type only needed in .cc)
    class ProductQuantizer;
}

namespace pomai::ai::soa
{
    // Forward-declare SoA helper (defined in src/ai/vector_store_soa.h)
    class VectorStoreSoA;
}

namespace pomai::ai
{

    class VectorStore
    {
    public:
        VectorStore();
        ~VectorStore();

        // Initialize underlying PPHNSW parameters. In sharded mode we'll split max_elements
        // across shards (evenly). The 'arena' parameter is only used in single-map mode.
        bool init(size_t dim, size_t max_elements, size_t M, size_t ef_construction, pomai::memory::PomaiArena *arena = nullptr);

        // Attach persistent PomaiMap (legacy single-map mode).
        void attach_map(PomaiMap *map);

        // Attach ShardManager to enable sharded operation.
        // Must be called after init() and before any upsert/remove/search if you want sharded mode.
        // shard_count is the number of shards the manager exposes (should match manager internal count).
        void attach_shard_manager(ShardManager *mgr, uint32_t shard_count);

        // Attach SoA storage helper (optional). Ownership transferred.
        void attach_soa(std::unique_ptr<pomai::ai::soa::VectorStoreSoA> soa);

        // Enable IVF-PQ filter (optional). Must be called after init() (and before heavy upserts).
        bool enable_ivf(size_t num_clusters = 1024, size_t m_sub = 16, size_t nbits = 8, uint64_t seed = 12345);

        // Upsert / remove / search
        bool upsert(const char *key, size_t klen, const float *vec);
        bool remove(const char *key, size_t klen);
        // Search returns vector of (key, score). Scores are squared L2 distances.
        std::vector<std::pair<std::string, float>> search(const float *query, size_t dim, size_t topk);

        // Number of indexed items known in-memory (sum across shards or local cache)
        size_t size() const;

        // ---------------- Memory usage reporting ----------------
        // Approximate memory usage breakdown aggregated across mode.
        struct MemoryUsage
        {
            uint64_t payload_bytes{0};        // bytes used by stored payloads (PPEHeader + payload)
            uint64_t index_overhead_bytes{0}; // neighbor lists, internal tables, misc
            uint64_t total_bytes{0};          // payload + overhead
        };

        // Return best-effort estimate of memory usage (bytes). Fast, non-blocking.
        MemoryUsage memoryUsage() const noexcept;

        // PQ / Codebooks (single declaration)
        Codebooks codebooks_;
        std::unique_ptr<ProductQuantizer> pq_;
        size_t pq_packed_bytes_{0};

    private:
        // Legacy single-map implementation (kept for compatibility)
        bool init_single(size_t dim, size_t max_elements, size_t M, size_t ef_construction, pomai::memory::PomaiArena *arena);
        // Sharded initialization: create per-shard VectorStore and initialize each.
        bool init_sharded(size_t dim, size_t max_elements_total, size_t M, size_t ef_construction, ShardManager *mgr);

        // helpers
        std::unique_ptr<char[]> build_seed_buffer(const float *vec) const;
        bool store_label_in_map(uint64_t label, const char *key, size_t klen);
        uint64_t read_label_from_map(const char *key, size_t klen) const;

        // merge helper: merge sorted K lists into final topk (nearest)
        static std::vector<std::pair<std::string, float>> merge_topk(const std::vector<std::vector<std::pair<std::string, float>>> &lists, size_t topk);

        // configuration & state
        size_t dim_{0};
        size_t max_elements_total_{0};
        size_t M_{0};
        size_t ef_construction_{0};

        // single-map mode members
        PomaiMap *map_{nullptr};
        pomai::memory::PomaiArena *arena_{nullptr};
        std::unique_ptr<hnswlib::L2Space> l2space_;
        std::unique_ptr<PPHNSW<float>> pphnsw_;
        std::unique_ptr<PPIVF> ivf_;
        bool ivf_enabled_{false};

        // SoA + fingerprint helpers (phase 2)
        std::unique_ptr<pomai::ai::soa::VectorStoreSoA> soa_;
        std::unique_ptr<FingerprintEncoder> fingerprint_;
        size_t fingerprint_bytes_{0};

        // label <-> key mappings (in-memory cache) for single-mode
        mutable std::mutex label_map_mu_;
        std::unordered_map<uint64_t, std::string> label_to_key_;
        std::unordered_map<std::string, uint64_t> key_to_label_;
        std::atomic<uint64_t> next_label_{1};

        // NEW: map id_entry (what's stored in SoA ids block) -> label, used when SoA stores local offsets
        // protected by label_map_mu_
        std::unordered_map<uint64_t, uint64_t> identry_to_label_;

        // Reader-writer lock (single-mode). Per-shard VectorStore has its own locks.
        mutable std::shared_mutex rw_mu_;

        // Sharded mode members
        ShardManager *shard_mgr_{nullptr};
        uint32_t shard_count_{0};
        // per-shard VectorStore instances (owned)
        std::vector<std::unique_ptr<VectorStore>> per_shard_stores_; // note: recursive use — per-shard single-mode VectorStore
        bool sharded_mode_{false};

        // PPSM instance used in sharded mode to enqueue writes / run searches
        std::unique_ptr<pomai::core::PPSM> ppsm_;

        // PPPQ demoter thread state (single-mode)
        std::thread ppq_demote_thread_;
        std::atomic<bool> ppq_demote_running_{false};
        uint64_t ppq_demote_interval_ms_{5000};
        uint64_t ppq_demote_cold_thresh_ms_{5000};

        // disable copy
        VectorStore(const VectorStore &) = delete;
        VectorStore &operator=(const VectorStore &) = delete;
    };

    // ---------------------------
    // Inline helper implementations ...
    // ---------------------------

    inline std::unique_ptr<char[]> VectorStore::build_seed_buffer(const float *vec) const
    {
        if (!vec || dim_ == 0)
            return nullptr;
        size_t payload_size = dim_ * sizeof(float);
        auto buf = std::make_unique<char[]>(payload_size);
        std::memcpy(buf.get(), reinterpret_cast<const char *>(vec), payload_size);
        return buf;
    }

    inline bool VectorStore::store_label_in_map(uint64_t label, const char *key, size_t klen)
    {
        if (!map_ || !key || klen == 0)
            return false;
        uint64_t le = label;
        const char *vptr = reinterpret_cast<const char *>(&le);
        bool ok = map_->put(key, static_cast<uint32_t>(klen), vptr, static_cast<uint32_t>(sizeof(le)));
        if (!ok)
            return false;
        // mark seed as vector if present
        Seed *s = map_->find_seed(key, static_cast<uint32_t>(klen));
        if (s)
            s->type = Seed::OBJ_VECTOR;
        return true;
    }

    inline uint64_t VectorStore::read_label_from_map(const char *key, size_t klen) const
    {
        if (!map_ || !key || klen == 0)
            return 0;
        uint32_t outlen = 0;
        const char *val = map_->get(key, static_cast<uint32_t>(klen), &outlen);
        if (!val || outlen != sizeof(uint64_t))
            return 0;
        uint64_t label = 0;
        std::memcpy(&label, val, sizeof(label));
        return label;
    }

} // namespace pomai::ai