#pragma once
// sharding/shard_manager.h
/*
 * [FIXED] Updated to use centralized PomaiConfig and Shard injection.
 */

#include <vector>
#include <thread>
#include <sched.h>
#include <iostream>
#include <cmath>
#include "src/core/shard.h"
#include "src/core/config.h"

namespace pomai::core
{
    static inline uint64_t next_power_of_two_u64(uint64_t v)
    {
        if (v == 0)
            return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return ++v;
    }

    class ShardManager
    {
    private:
        std::vector<std::unique_ptr<Shard>> shards_;
        const pomai::config::PomaiConfig &cfg_;
        uint32_t thread_count_;

    public:
        // [FIXED] Constructor nhận config tập trung
        ShardManager(uint32_t count, const pomai::config::PomaiConfig &cfg)
            : cfg_(cfg), thread_count_(count)
        {
            std::cout << "[ShardManager] Initializing " << count << " shards...\n";

            for (uint32_t i = 0; i < count; ++i)
            {
                // [FIXED] Khởi tạo Shard theo constructor mới nhận PomaiConfig
                shards_.push_back(std::make_unique<Shard>(cfg_));
            }
        }

        ~ShardManager() = default;

        Shard *get_shard_by_id(uint32_t id)
        {
            if (id >= shards_.size())
                return nullptr;
            return shards_[id].get();
        }

        void pin_thread(uint32_t core_id)
        {
            if (!cfg_.shard_manager.enable_cpu_pinning)
                return;

#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core_id, &cpuset);
            pthread_t current_thread = pthread_self();
            if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0)
            {
                std::cerr << "ShardManager: failed to pin thread to core " << core_id << "\n";
            }
#endif
        }

        size_t shard_count() const { return shards_.size(); }
    };

} // namespace pomai::core