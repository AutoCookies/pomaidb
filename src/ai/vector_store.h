#pragma once
// src/ai/vector_store.h
//
// VectorStore: a small wrapper combining an HNSW-based PPHNSW index with an
// optional PP-IVF candidate filter (IVF + tiny PQ). The IVFPQ path reduces the
// candidate set for search and avoids building a very large monolithic index
// in memory on resource-constrained machines.
//

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/pomai_hnsw.h"
#include "src/memory/arena.h"
#include "src/core/map.h"
#include "src/ai/pp_ivf.h"

namespace pomai::ai
{

    class VectorStore
    {
    public:
        VectorStore() = default;
        ~VectorStore();

        // Initialize underlying PPHNSW index.
        bool init(size_t dim, size_t max_elements, size_t M, size_t ef_construction, pomai::memory::PomaiArena *arena = nullptr);

        // Attach persistent PomaiMap (key->label storage)
        void attach_map(PomaiMap *map);

        // Enable IVF-PQ filter (optional). Must be called after init().
        // Returns true on success.
        bool enable_ivf(size_t num_clusters = 1024, size_t m_sub = 16, size_t nbits = 8, uint64_t seed = 12345);

        // Upsert / remove / search
        bool upsert(const char *key, size_t klen, const float *vec);
        bool remove(const char *key, size_t klen);
        std::vector<std::pair<std::string, float>> search(const float *query, size_t dim, size_t topk);

        // Number of indexed items known in-memory
        size_t size() const;

    private:
        size_t dim_{0};
        pomai::memory::PomaiArena *arena_{nullptr};
        PomaiMap *map_{nullptr};

        // HNSW components
        std::unique_ptr<hnswlib::L2Space> l2space_;
        std::unique_ptr<PPHNSW<float>> pphnsw_;

        // Optional IVFPQ filter
        std::unique_ptr<PPIVF> ivf_;
        bool ivf_enabled_{false};

        // label <-> key mappings (in-memory cache)
        mutable std::mutex label_map_mu_;
        std::unordered_map<uint64_t, std::string> label_to_key_;
        std::unordered_map<std::string, uint64_t> key_to_label_;

        // monotonic label allocator
        std::atomic<uint64_t> next_label_{1};

        // PPPQ demoter thread state
        std::thread ppq_demote_thread_;
        std::atomic<bool> ppq_demote_running_{false};
        uint64_t ppq_demote_interval_ms_{5000};
        uint64_t ppq_demote_cold_thresh_ms_{5000};

        // helpers
        std::unique_ptr<char[]> build_seed_buffer(const float *vec) const;
        bool store_label_in_map(uint64_t label, const char *key, size_t klen);
        uint64_t read_label_from_map(const char *key, size_t klen) const;

        // internal helpers for PPPQ demoter
        void start_pppq_demoter();
        void stop_pppq_demoter();
    };

} // namespace pomai::ai