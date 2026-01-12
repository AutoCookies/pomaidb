// sharding/shard.h
#pragma once

#include <memory>
#include <stdexcept>
#include <iostream>
#include "src/core/map.h"
#include "src/memory/arena.h"

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
        Shard(double size_gb, uint64_t map_slots)
        {
            // PomaiArena is in pomai::memory namespace
            auto a = pomai::memory::PomaiArena::FromGB(size_gb);
            if (!a.is_valid())
            {
                throw std::runtime_error("Shard: PomaiArena allocation failed (is_valid == false)");
            }
            arena = std::make_unique<pomai::memory::PomaiArena>(std::move(a));
            map = std::make_unique<PomaiMap>(arena.get(), map_slots);
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