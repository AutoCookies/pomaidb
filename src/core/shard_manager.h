// sharding/shard_manager.h
#pragma once

#include <vector>
#include <thread>
#include <sched.h>
#include <iostream>
#include <cmath>
#include "src/core/shard.h"
#include "src/core/config.h"
#include "src/core/seed.h"

namespace pomai::core
{
    // Helper: compute next power of two >= v
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
        std::vector<Shard *> shards;
        uint32_t thread_count;

    public:
        ShardManager(uint32_t count) : thread_count(count)
        {
            using pomai::config::runtime;

            // Arena size per shard in MB from config (default 512)
            uint64_t arena_mb = runtime.arena_mb_per_shard;
            if (arena_mb == 0)
                arena_mb = 512; // fallback

            // Convert MB to GB for Shard constructor (double)
            double arena_gb = static_cast<double>(arena_mb) / 1024.0;

            // Pre-compute map_slots based on estimated average object size
            // Assume average object ~128 bytes? This is heuristic.
            // Better: use a small probe arena to check capacity.
            auto probe = pomai::memory::PomaiArena::FromMB(1);
            if (!probe.is_valid())
            {
                throw std::runtime_error("ShardManager: probe arena allocation failed for determining map_slots");
            }
            uint64_t max_seeds = probe.get_capacity_bytes() / sizeof(Seed);
            if (max_seeds == 0)
                max_seeds = 1;

            uint64_t map_slots = next_power_of_two_u64(max_seeds);

            std::cout << "[ShardManager] Creating " << count << " shards; arena_mb_per_shard=" << arena_mb
                      << " MB, arena_gb=" << arena_gb << ", estimated max_seeds=" << max_seeds
                      << ", map_slots=" << map_slots << "\n";

            for (uint32_t i = 0; i < count; ++i)
            {
                shards.push_back(new Shard(arena_gb, map_slots));
            }
        }

        ~ShardManager()
        {
            for (auto *s : shards)
                delete s;
            shards.clear();
        }

        Shard *get_shard_by_id(uint32_t id)
        {
            if (id >= shards.size())
                return nullptr;
            return shards[id];
        }

        // Ép luồng vào Core CPU (Affinity) - Tuyệt kỹ 10/10 để tránh Context Switch
        void pin_thread(uint32_t core_id)
        {
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
    };

} // namespace pomai::core