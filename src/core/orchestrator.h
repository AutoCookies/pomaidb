/* src/core/orchestrator.h */
#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <iostream>
#include <future>
#include <algorithm>

#include "src/memory/shard_arena.h"
#include "src/ai/pomai_orbit.h"
#include "src/core/config.h"

namespace pomai::core
{
    class GlobalOrchestrator
    {
    public:
        explicit GlobalOrchestrator(const pomai::config::PomaiConfig &config)
        {
            size_t num_shards = config.orchestrator.shard_count > 0
                                    ? config.orchestrator.shard_count
                                    : std::max<size_t>(1, std::thread::hardware_concurrency());

            size_t ram_bytes = config.res.arena_mb_per_shard * 1024 * 1024;

            for (size_t i = 0; i < num_shards; ++i)
            {
                auto arena = std::make_unique<pomai::memory::ShardArena>(static_cast<uint32_t>(i), ram_bytes);

                pomai::ai::orbit::PomaiOrbit::Config orbit_cfg;
                orbit_cfg.dim = 128; // Có thể lấy từ config chung

                // Sử dụng tiền tố từ config
                orbit_cfg.data_path = config.res.data_root + "/" + config.orchestrator.shard_path_prefix + std::to_string(i);
                orbit_cfg.use_cortex = (i == 0);

                auto orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(orbit_cfg, arena.get());
                arenas_.push_back(std::move(arena));
                orbits_.push_back(std::move(orbit));
            }
        }

        // Router: hash label -> shard
        bool insert(const float *vec, uint64_t label)
        {
            if (orbits_.empty())
                return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->insert(vec, label);
        }

        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k)
        {
            if (orbits_.empty())
                return {};
            std::vector<std::future<std::vector<std::pair<uint64_t, float>>>> futures;
            futures.reserve(orbits_.size());

            for (auto &orbit : orbits_)
            {
                futures.push_back(std::async(std::launch::async, [&orbit, query, k]()
                                             { return orbit->search(query, k); }));
            }

            // gather
            std::vector<std::pair<uint64_t, float>> merged;
            for (auto &f : futures)
            {
                try
                {
                    auto res = f.get();
                    merged.insert(merged.end(), res.begin(), res.end());
                }
                catch (...)
                {
                }
            }

            // keep top-k by ascending distance
            if (merged.size() > k)
            {
                std::partial_sort(merged.begin(), merged.begin() + k, merged.end(), [](const auto &a, const auto &b)
                                  { return a.second < b.second; });
                merged.resize(k);
            }
            else
            {
                std::sort(merged.begin(), merged.end(), [](const auto &a, const auto &b)
                          { return a.second < b.second; });
            }
            return merged;
        }

        // New: get vector by label (routes to shard)
        bool get(uint64_t label, std::vector<float> &out_vec)
        {
            if (orbits_.empty())
                return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->get(label, out_vec);
        }

        // New: remove (soft-delete) by label
        bool remove(uint64_t label)
        {
            if (orbits_.empty())
                return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->remove(label);
        }

        size_t num_shards() const noexcept { return orbits_.size(); }

    private:
        // Sở hữu tài nguyên
        std::vector<std::unique_ptr<pomai::memory::ShardArena>> arenas_;
        std::vector<std::unique_ptr<pomai::ai::orbit::PomaiOrbit>> orbits_;
    };
}