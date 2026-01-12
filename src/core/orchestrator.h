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
    // GlobalOrchestrator: Bộ điều phối trung tâm thay thế PPSM.
    // Quản lý N Shard (mỗi Shard gồm 1 Arena + 1 Orbit Engine).
    class GlobalOrchestrator
    {
    public:
        GlobalOrchestrator()
        {
            auto& cfg = pomai::config::runtime;
            size_t num_shards = cfg.shard_count > 0 ? cfg.shard_count : std::max<size_t>(1, std::thread::hardware_concurrency());
            size_t ram_per_shard = cfg.arena_mb_per_shard;
            if (ram_per_shard == 0) ram_per_shard = 16; // default MB
            size_t ram_bytes = ram_per_shard * 1024 * 1024;

            std::cout << "[Orchestrator] Initializing " << num_shards << " shards (Hardcore Mode)...\n";

            for (size_t i = 0; i < num_shards; ++i)
            {
                // 1. Init Arena (Lock-free)
                auto arena = std::make_unique<pomai::memory::ShardArena>(static_cast<uint32_t>(i), ram_bytes);
                
                // 2. Init Engine
                pomai::ai::orbit::PomaiOrbit::Config orbit_cfg;
                orbit_cfg.dim = cfg.dim > 0 ? cfg.dim : 128; // prefer runtime dim if set
                orbit_cfg.data_path = "./data/shard_" + std::to_string(i);
                orbit_cfg.use_cortex = (i == 0); // Chỉ Shard 0 chạy Cortex để tiết kiệm thread

                // Orbit sở hữu Arena thông qua ArenaView, nhưng ở đây ta giữ ownership trong Orchestrator
                // và truyền pointer raw vào Orbit.
                auto orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(orbit_cfg, arena.get());

                arenas_.push_back(std::move(arena));
                orbits_.push_back(std::move(orbit));
            }
        }

        // Router: hash label -> shard
        bool insert(const float* vec, uint64_t label)
        {
            if (orbits_.empty()) return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->insert(vec, label);
        }

        std::vector<std::pair<uint64_t, float>> search(const float* query, size_t k)
        {
            if (orbits_.empty()) return {};
            std::vector<std::future<std::vector<std::pair<uint64_t, float>>>> futures;
            futures.reserve(orbits_.size());

            for (auto& orbit : orbits_) {
                futures.push_back(std::async(std::launch::async, [&orbit, query, k]() {
                    return orbit->search(query, k);
                }));
            }

            // gather
            std::vector<std::pair<uint64_t, float>> merged;
            for (auto &f : futures) {
                try {
                    auto res = f.get();
                    merged.insert(merged.end(), res.begin(), res.end());
                } catch (...) {}
            }

            // keep top-k by ascending distance
            if (merged.size() > k) {
                std::partial_sort(merged.begin(), merged.begin() + k, merged.end(), [](const auto &a, const auto &b){
                    return a.second < b.second;
                });
                merged.resize(k);
            } else {
                std::sort(merged.begin(), merged.end(), [](const auto &a, const auto &b){
                    return a.second < b.second;
                });
            }
            return merged;
        }

        // New: get vector by label (routes to shard)
        bool get(uint64_t label, std::vector<float> &out_vec)
        {
            if (orbits_.empty()) return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->get(label, out_vec);
        }

        // New: remove (soft-delete) by label
        bool remove(uint64_t label)
        {
            if (orbits_.empty()) return false;
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