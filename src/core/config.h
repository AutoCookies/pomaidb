// core/config.h
#pragma once
#include <cstdint>
#include <string>
#include <atomic>
#include <optional>

namespace pomai::config
{

    // Immutable compile-time constants (kept as constexpr)
    constexpr size_t SEED_PAYLOAD_BYTES = 48;
    constexpr size_t SEED_ALIGNMENT = 64;

    // Server compile-time configuration constants (moved here for centralised tuning)
    // These are constexpr so they can be used in compile-time contexts (array sizes, std::array, etc).
    constexpr size_t SERVER_MAX_EVENTS = 1024;           // epoll events array size
    constexpr size_t SERVER_READ_BUFFER = 4096;          // default per-read temporary buffer / initial connection reserve
    constexpr size_t SERVER_MAX_RESP_PARTS = 16;         // maximum number of RESP array elements we accept
    constexpr size_t SERVER_MAX_PART_BYTES = 1 << 20;    // 1 MiB max size per RESP bulk-string part (safety)
    constexpr size_t SERVER_MAX_COMMAND_BYTES = 4 << 20; // 4 MiB max total command frame (safety)
    constexpr bool SERVER_REQUIRE_RESP_ARRAYS = true;    // require RESP arrays (disable inline commands)

    // PWP (Pomai Wire Protocol) related constants
    constexpr uint8_t PWP_MAGIC = 0x50;                      // 'P'
    constexpr size_t PWP_MAX_PACKET_SIZE = 16 * 1024 * 1024; // 16 MiB cap for a single frame
    constexpr size_t PWP_INBUF_RESERVE = SERVER_READ_BUFFER; // initial inbuf reservation
    constexpr size_t PWP_OUTBUF_RESERVE = 4096;              // initial outbuf reservation
    constexpr int PWP_LISTEN_BACKLOG = 1024;                 // listen backlog

    // Seed-related constants (centralised)
    constexpr uint8_t SEED_OBJ_STRING = 0;
    constexpr uint8_t SEED_FLAG_INLINE = 0;
    constexpr uint8_t SEED_FLAG_INDIRECT = 0x1;

    // Map-related compile-time tuning
    // - MAP_MAX_INLINE_KEY: maximum key bytes stored inline inside Seed.payload (leave room for pointer)
    // - MAP_PTR_BYTES: size used for storing blob offset (uint64_t)
    constexpr size_t MAP_MAX_INLINE_KEY = 40;          // keys must be reasonably small to fit inline
    constexpr size_t MAP_PTR_BYTES = sizeof(uint64_t); // we store blob offsets (uint64_t)

    // Runtime-tunable defaults (can be changed at startup)
    struct Runtime
    {
        // harvest tuning
        std::uint32_t harvest_sample = 5;
        std::uint32_t harvest_max_attempts = 20;
        std::uint32_t initial_entropy = 8;
        std::uint32_t max_entropy = 1024;

        // arena defaults (MB)
        std::uint64_t arena_mb_per_shard = 512;

        // server default port
        std::uint16_t default_port = 6379;

        // deterministic RNG for tests; empty = random_device
        std::optional<uint64_t> rng_seed{};

        // Number of shards to create when using ShardManager / PPSM.
        // 0 means "auto" (server may pick hardware_concurrency()). Can be set via env POMAI_SHARD_COUNT.
        std::uint32_t shard_count = 0;

        // Optional: total maximum elements for vector index (across all shards).
        // Set via env POMAI_MAX_ELEMENTS_TOTAL (optional). 0 => use code default/fallback.
        std::uint64_t max_elements_total = 0;

        // --- Phase 4: parameterizable thresholds (ms / counts) ----------------
        // Maximum number of elements considered "hot" before we become aggressive about promotions.
        // This can be used by demoter/promoter heuristics if desired.
        std::uint64_t hot_size_limit = 65536;

        // Lookahead window (milliseconds) used when considering promotions:
        // if predicted next-access > now + promote_lookahead_ms then attempt promote.
        std::uint64_t promote_lookahead_ms = 1000; // default 1 second

        // Threshold used by PPPQ/predictor for deciding demotion/pack4 (milliseconds).
        std::uint64_t demote_threshold_ms = 5000; // default 5 seconds

        // Async demote configuration:
        // - demote_async_max_pending: number of outstanding async demote tasks allowed (0 = disabled -> synchronous writes)
        std::uint64_t demote_async_max_pending = 1000;

        // If queue is full and demote_async_max_pending > 0, demote_sync_fallback controls behavior:
        //  - true: perform synchronous demote write in caller (blocking)
        //  - false: skip demotion when queue full (keep RAM representation)
        bool demote_sync_fallback = true;

        // --- Phase 2: fingerprint / prefilter tuning --------------------------
        // Default number of fingerprint bits (SimHash). Typical values: 256, 512.
        // Can be overridden via env POMAI_FINGERPRINT_BITS
        std::uint32_t fingerprint_bits = 512;

        // Hamming distance threshold (max allowed bits differing) used by prefilter
        // when collecting candidate indices. Typical default for 512-bit SimHash: 64..128.
        // Can be overridden via env POMAI_PREFILTER_HAMMING_THRESHOLD
        std::uint32_t prefilter_hamming_threshold = 128;
    };

    // Global runtime config (initialize early in main)
    extern Runtime runtime;

    // Helpers: init from env / string
    void init_from_env();                       // read env vars like POMAI_HARVEST_SAMPLE
    void init_from_args(int argc, char **argv); // optional lightweight arg parser

} // namespace pomai::config