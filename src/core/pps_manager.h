/*
 * src/core/pps_manager.h
 *
 * Pomai Pomegranate Shard Manager (PPSM) - ORBIT EDITION.
 *
 * Updates:
 * - Replaced PPHNSW/SoA/FP/PQ with a single Unified Engine: PomaiOrbit.
 * - simplified ShardState to hold just the Map and the Orbit instance.
 */

#pragma once

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
#include <future>
#include <cstdint>

#include "src/core/shard.h"
#include "src/core/shard_manager.h"
#include "src/ai/pomai_orbit.h" // The new Unified Engine

namespace pomai::core
{

    class PPSM
    {
    public:
        PPSM(ShardManager *shard_mgr,
             size_t dim,
             size_t max_elements_total,
             bool async_insert_ack = true);

        ~PPSM();

        PPSM(const PPSM &) = delete;
        PPSM &operator=(const PPSM &) = delete;

        // Core API
        bool addVec(const char *key, size_t klen, const float *vec);
        bool removeKey(const char *key, size_t klen);
        std::vector<std::pair<std::string, float>> search(const float *query, size_t dim, size_t topk);

        // Stats
        size_t size() const noexcept;

        struct MemoryUsage
        {
            uint64_t payload_bytes;
            uint64_t index_overhead_bytes;
            uint64_t total_bytes;
        };
        MemoryUsage memoryUsage() const noexcept;

    private:
        struct Task
        {
            std::string key;
            std::vector<float> vec;
            uint64_t label;
            bool replace;
            std::promise<bool> done;
        };

        struct ShardState
        {
            uint32_t id;

            // Storage Components
            PomaiArena *arena{nullptr};
            PomaiMap *map{nullptr}; // Key -> Label mapping

            // The Unified Vector Engine
            std::unique_ptr<ai::orbit::PomaiOrbit> orbit;

            // Worker Thread handling writes
            std::deque<std::unique_ptr<Task>> q;
            mutable std::mutex q_mu;
            std::condition_variable q_cv;
            std::thread worker;
            bool running{false};

            // Label -> Key mapping (In-memory reverse lookup for search results)
            std::unordered_map<uint64_t, std::string> label_to_key;
            mutable std::mutex label_map_mu;
        };

        uint32_t computeShard(const char *key, size_t klen) const noexcept;
        void workerLoop(ShardState &sh);
        void startWorkers();
        void stopWorkers();

        // Initialize Orbit for a shard (including auto-training if fresh)
        bool initPerShard(ShardState &s, size_t dim, size_t per_shard_max);

        static std::vector<std::pair<std::string, float>> merge_results(const std::vector<std::vector<std::pair<std::string, float>>> &parts, size_t topk);

        ShardManager *shard_mgr_; // non-owning
        size_t dim_;
        size_t max_elements_total_;
        bool async_insert_ack_;

        std::vector<std::unique_ptr<ShardState>> shards_;
        std::atomic<uint64_t> next_label_global_{1};
    };

} // namespace pomai::core