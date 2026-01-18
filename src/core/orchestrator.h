#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <iostream>
#include <future>
#include <algorithm>
#include <string>

#include "src/memory/shard_arena.h"
#include "src/ai/pomai_orbit.h"
#include "src/core/config.h"

namespace pomai::core
{
    class GlobalOrchestrator
    {
    public:
        // Constructor: Khởi tạo toàn bộ Shards và Orbits dựa trên cấu hình Global
        explicit GlobalOrchestrator(const pomai::config::PomaiConfig &config)
        {
            // 1. Xác định số lượng Shard (phân mảnh)
            size_t num_shards = config.orchestrator.shard_count > 0
                                    ? config.orchestrator.shard_count
                                    : std::max<size_t>(1, std::thread::hardware_concurrency());

            // 2. Tính toán Memory cho mỗi Shard
            size_t ram_bytes = config.res.arena_mb_per_shard * 1024 * 1024;

            arenas_.reserve(num_shards);
            orbits_.reserve(num_shards);

            for (size_t i = 0; i < num_shards; ++i)
            {
                // [FIX 1] Construct ShardArena với đúng chữ ký (ID, Capacity, RemoteDir)
                std::string remote_dir = config.res.data_root + "/remote_shard_" + std::to_string(i);

                auto arena = std::make_unique<pomai::memory::ShardArena>(
                    static_cast<uint32_t>(i),
                    ram_bytes,
                    remote_dir);

                // [FIX 2] Map cấu hình từ Global Config sang Local Orbit Config
                pomai::ai::orbit::PomaiOrbit::Config orbit_cfg;

                // Lưu ý: Dimension hiện chưa có trong Config Global, tạm thời hardcode hoặc lấy default
                // TODO: Đưa `dim` vào pomai::config::PomaiConfig
                orbit_cfg.dim = 128; 

                // Path riêng cho metadata/schema của orbit này
                orbit_cfg.data_path = config.res.data_root + "/" + config.orchestrator.shard_path_prefix + std::to_string(i);

                // Map các tham số thuật toán
                orbit_cfg.num_centroids = config.orbit.num_centroids;
                orbit_cfg.m_neighbors = config.orbit.m_neighbors;
                
                // Map cấu hình mạng (Cortex)
                // orbit_cfg.use_cortex = config.network.enable_broadcast; // Nếu cần map dynamic

                // Khởi tạo Orbit với Arena pointer (Dependency Injection)
                auto orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(orbit_cfg, arena.get());

                arenas_.push_back(std::move(arena));
                orbits_.push_back(std::move(orbit));
            }
            
            std::clog << "[Orchestrator] Initialized " << num_shards << " shards.\n";
        }

        // Router: Hash Label -> Shard ID -> Insert
        bool insert(const float *vec, uint64_t label)
        {
            if (orbits_.empty())
                return false;
            
            // Simple Modulo Hashing (Cần Consistent Hashing cho Production scaling)
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->insert(vec, label);
        }

        // Search: Scatter-Gather (Gửi query tới tất cả shards -> Gom kết quả)
        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k)
        {
            if (orbits_.empty())
                return {};
                
            // [PERF WARNING] std::async tạo thread mới cho mỗi request.
            // TODO: Thay thế bằng ThreadPool cố định để tránh Context Switch Overhead.
            std::vector<std::future<std::vector<std::pair<uint64_t, float>>>> futures;
            futures.reserve(orbits_.size());

            for (auto &orbit : orbits_)
            {
                futures.push_back(std::async(std::launch::async, [&orbit, query, k]()
                                             { return orbit->search(query, k); }));
            }

            // Gather Phase: Thu thập kết quả
            std::vector<std::pair<uint64_t, float>> merged;
            for (auto &f : futures)
            {
                try
                {
                    auto res = f.get();
                    merged.insert(merged.end(), res.begin(), res.end());
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[Orchestrator] Search shard error: " << e.what() << "\n";
                }
            }

            // Merge Phase: Sort và lấy Top-K toàn cục
            if (merged.size() > k)
            {
                // Partial Sort hiệu quả hơn Full Sort cho Top-K
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

        // Get Vector by Label
        bool get(uint64_t label, std::vector<float> &out_vec)
        {
            if (orbits_.empty())
                return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->get(label, out_vec);
        }

        // Remove Vector by Label
        bool remove(uint64_t label)
        {
            if (orbits_.empty())
                return false;
            size_t shard_id = static_cast<size_t>(label % orbits_.size());
            return orbits_[shard_id]->remove(label);
        }

        size_t num_shards() const noexcept { return orbits_.size(); }

    private:
        // Sở hữu tài nguyên (Resource Ownership)
        std::vector<std::unique_ptr<pomai::memory::ShardArena>> arenas_;
        std::vector<std::unique_ptr<pomai::ai::orbit::PomaiOrbit>> orbits_;
    };
}