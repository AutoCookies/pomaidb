#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <optional>

namespace pomai::config
{
    // --- Compile-time Constants (Immutable) ---

    constexpr size_t SEED_PAYLOAD_BYTES = 48;
    constexpr size_t SEED_ALIGNMENT = 64;

    // Server Limits
    constexpr size_t SERVER_MAX_EVENTS = 1024;
    constexpr size_t SERVER_READ_BUFFER = 4096;
    constexpr size_t SERVER_MAX_COMMAND_BYTES = 4 << 20; // 4 MiB max

    // PWP Protocol
    constexpr uint8_t PWP_MAGIC = 0x50; // 'P'

    // Map Tuning
    constexpr size_t MAP_MAX_INLINE_KEY = 40;
    constexpr size_t MAP_PTR_BYTES = sizeof(uint64_t);

    // Restore missing Seed constants required by src/core/seed.h
    constexpr uint8_t SEED_OBJ_STRING = 0;
    constexpr uint8_t SEED_FLAG_INLINE = 0;
    constexpr uint8_t SEED_FLAG_INDIRECT = 0x1;

    // --- Runtime Configuration (Tunable via CLI flags or ENV) ---
    struct Runtime
    {
        // 1. Server Basics
        std::uint16_t default_port = 7777;
        std::optional<uint64_t> rng_seed{};

        // 2. Memory & Storage
        std::uint64_t arena_mb_per_shard = 2048; // 2 GiB

        // Async IO Settings
        std::uint64_t demote_async_max_pending = 1000;
        bool demote_sync_fallback = true;

        // 3. Sharding & Scaling
        std::uint32_t shard_count = 0;
        std::uint64_t max_elements_total = 0;

        // 4. Algorithm Tuning (Orbit/SimHash)
        std::uint32_t fingerprint_bits = 512;
        std::uint32_t prefilter_hamming_threshold = 128;

        // NEW: vector dimensionality
        std::uint32_t dim = 0;

        // NEW: disable synapse codec via flag (no env needed)
        bool disable_synapse = false;

        // [FIX] Restore Map tuning parameters required by src/core/map.h
        std::uint32_t harvest_sample = 5;
        std::uint32_t harvest_max_attempts = 20;
        std::uint32_t initial_entropy = 8;
        std::uint32_t max_entropy = 1024;
    };

    extern Runtime runtime;

    void init_from_env();
    void init_from_args(int argc, char **argv);

} // namespace pomai::config