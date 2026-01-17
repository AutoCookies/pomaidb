#pragma once
// src/core/shard.h
//
// Lightweight Shard wrapper owning a PomaiArena and a PomaiMap.
// Keeps construction simple and throws on allocation failure.

#include <memory>
#include <stdexcept>
#include <iostream>

#include "src/memory/arena.h"
#include "src/core/map.h"

namespace pomai::core
{
    /*
     * Shard
     * - Owns a PomaiArena and PomaiMap (per-shard).
     * - Validates arena allocation at construction and throws if allocation fails.
     * - Policy: single-thread-per-shard. If you change to multi-writer, add synchronization.
     */
    struct Shard
    {
        std::unique_ptr<pomai::memory::PomaiArena> arena;
        std::unique_ptr<PomaiMap> map;

        // Constructor expects arena size in GB and the number of map slots (power of two).
        Shard(const pomai::config::PomaiConfig &cfg)
        {
            // Sử dụng MB để chính xác hơn GB và khớp với config.res
            auto a = pomai::memory::PomaiArena::FromMB(cfg.res.arena_mb_per_shard);
            if (!a.is_valid())
            {
                throw std::runtime_error("Shard: PomaiArena allocation failed");
            }
            arena = std::make_unique<pomai::memory::PomaiArena>(std::move(a));

            // [FIXED] Truyền đúng 3 tham số theo định nghĩa mới của PomaiMap
            map = std::make_unique<pomai::core::PomaiMap>(
                arena.get(),
                cfg.map_tuning.default_slots,
                cfg);
        }

        ~Shard() = default;

        // No copy
        Shard(const Shard &) = delete;
        Shard &operator=(const Shard &) = delete;

        // Move
        Shard(Shard &&) noexcept = default;
        Shard &operator=(Shard &&) noexcept = default;

        // Helper accessor to get raw PomaiMap*
        PomaiMap *get_map() const { return map.get(); }
        pomai::memory::PomaiArena *get_arena() const { return arena.get(); }
    };

} // namespace pomai::core