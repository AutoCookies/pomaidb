#pragma once
// src/core/pps_manager.h
//
// Pomai Pomegranate Shard Manager (PPSM)
// Extended: per-shard label->key in-memory map to resolve keys on search results.
// Added MemoryUsage reporting API to estimate memory used by per-shard HNSW indices.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <future> // for std::promise/std::future
#include <cstdint>

#include "src/core/shard.h"         // Shard type (now in core)
#include "src/core/shard_manager.h" // ShardManager
#include "src/ai/pomai_hnsw.h"
#include "src/ai/ppe_predictor.h"
#include "src/ai/vector_store_soa.h"
#include "src/ai/fingerprint.h"
#include "src/ai/pq.h"

namespace pomai::core
{

    using labeltype = hnswlib::labeltype;

    class PPSM
    {
    public:
        PPSM(ShardManager *shard_mgr,
             size_t dim,
             size_t max_elements_total,
             size_t M = 8,
             size_t ef_construction = 50,
             bool async_insert_ack = true);

        ~PPSM();

        PPSM(const PPSM &) = delete;
        PPSM &operator=(const PPSM &) = delete;

        bool addVec(const char *key, size_t klen, const float *vec);
        bool removeKey(const char *key, size_t klen);
        std::vector<std::pair<std::string, float>> search(const float *query, size_t dim, size_t topk);

        size_t size() const noexcept;
        uint32_t shardCount() const noexcept { return static_cast<uint32_t>(shards_.size()); }

        // Memory usage reporting ------------------------------------------------
        // Approximate memory usage breakdown aggregated across all shards.
        // Values are estimates (conservative) and reported in bytes.
        struct MemoryUsage
        {
            uint64_t payload_bytes{0};        // space holding PPEHeader + payload (per-element * count)
            uint64_t index_overhead_bytes{0}; // graph/link lists + misc indexing overhead
            uint64_t total_bytes{0};          // sum of payload + index_overhead
        };

        // Return approximate MemoryUsage aggregated over all shards.
        MemoryUsage memoryUsage() const noexcept;
        std::vector<std::pair<std::string, float>> searchHolographic(const float *query, size_t dim, size_t topk);

    private:
        struct Task
        {
            std::string key;
            std::vector<float> vec;
            labeltype label;
            bool replace;
            std::promise<bool> done;
        };

        struct ShardState
        {
            uint32_t id{0};

            std::unique_ptr<hnswlib::L2Space> l2space;
            std::unique_ptr<ai::PPHNSW<float>> pphnsw;

            PomaiArena *arena{nullptr};
            PomaiMap *map{nullptr};

            std::deque<std::unique_ptr<Task>> q;
            mutable std::mutex q_mu;
            std::condition_variable q_cv;
            std::thread worker;
            bool running{false};

            mutable std::shared_mutex index_mu;

            // PPE predictor for shard
            pomai::ai::PPEPredictor ppe;

            // label -> key map for this shard (protected by label_map_mu)
            std::unordered_map<uint64_t, std::string> label_to_key;
            mutable std::mutex label_map_mu;

            std::unique_ptr<ai::soa::VectorStoreSoA> soa;
            std::unique_ptr<ai::FingerprintEncoder> fp_enc;
            std::unique_ptr<ai::ProductQuantizer> pq;

            size_t pq_packed_bytes{0};
        };

        uint32_t computeShard(const char *key, size_t klen) const noexcept;
        void workerLoop(ShardState &sh);
        void startWorkers();
        void stopWorkers();

        bool initPerShard(ShardState &s, size_t dim, size_t per_shard_max, size_t M, size_t ef_construction);
        bool initSoAPerShard(ShardState &s, size_t dim);

        static std::vector<std::pair<std::string, float>> merge_results(const std::vector<std::vector<std::pair<std::string, float>>> &parts, size_t topk);

        ShardManager *shard_mgr_; // non-owning
        size_t dim_;
        size_t max_elements_total_;
        size_t per_shard_max_;
        size_t M_;
        size_t ef_construction_;
        bool async_insert_ack_;

        std::vector<std::unique_ptr<ShardState>> shards_;

        std::atomic<uint64_t> next_label_global_{1};
    };

} // namespace pomai::core