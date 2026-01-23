#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <optional>
#include <vector>

namespace pomai::config
{
    constexpr size_t MAP_MAX_INLINE_KEY = 40;
    constexpr size_t MAP_PTR_BYTES = sizeof(uint64_t);

    // --- Sub-module Configurations ---

    struct MapTuning
    {
        uint32_t initial_entropy = 8;
        uint32_t max_entropy = 1024;
        uint32_t harvest_sample = 5;
        uint32_t harvest_max_attempts = 20;
        uint64_t default_slots = 1048576;
        size_t max_key_inline = 40;
    };

    // Replaced EternalEchoConfig with a new lossless packer config: ZeroHarmonyConfig.
    // Keep a backwards-compatibility alias `EternalEchoConfig = ZeroHarmonyConfig`.
    struct ZeroHarmonyConfig
    {
        // Enable the zero-harmony packing pipeline (per-bucket mean + delta packing).
        bool enable = true;

        // RLE zero-run encoding of exact-zero deltas (lossless).
        bool enable_rle_zero = true;

        // Attempt to store small non-zero deltas as IEEE754 float16 when the roundtrip
        // is exact; otherwise store full float32. This preserves lossless reconstruction
        // because the packer will use float32 whenever fp16 would lose information.
        bool use_half_nonzero = true;

        // Maximum absolute magnitude for which fp16 is considered (safety).
        // 65504 is the maximum finite fp16 value; conservative default.
        float half_max_exact_abs = 65504.0f;

        // Advisory: target maximum per-vector packed bytes for short in-place storage.
        // If a packed vector exceeds this size, callers may store it out-of-line (blob arena).
        size_t max_packed_bytes = 64;

        float zero_threshold = 1e-7f;
    };

    // Backwards alias: where code still refers to EternalEchoConfig, it will resolve to ZeroHarmonyConfig.
    using EternalEchoConfig = ZeroHarmonyConfig;

    struct FingerprintConfig
    {
        std::uint32_t fingerprint_bits = 512;
    };

    struct IdsBlockLayout
    {
        static constexpr uint64_t TOTAL_BITS = 64;
        static constexpr uint64_t TAG_BITS = 2;
        static constexpr uint64_t PAYLOAD_BITS = 62;
        static constexpr uint64_t TAG_SHIFT = PAYLOAD_BITS;
        static constexpr uint64_t TAG_MASK = (uint64_t)0x3ULL << TAG_SHIFT;
        static constexpr uint64_t PAYLOAD_MASK = (1ULL << PAYLOAD_BITS) - 1ULL;
    };

    struct StorageLayout
    {
        static constexpr size_t BLOB_HEADER_BYTES = sizeof(uint32_t);
    };

    struct NetworkCortexConstants
    {
        static constexpr size_t RECV_BUF_SZ = 4096;
        static constexpr size_t SAFE_UDP_PAYLOAD = 1200;
    };

    struct NetworkCortexConfig
    {
        uint16_t udp_port = 7777;
        uint64_t neighbor_ttl_ms = 5000;
        uint64_t pulse_interval_ms = 1000;
        bool enable_broadcast = true;
    };

    struct OrbitConfig
    {
        uint32_t auto_scale_factor = 1500;
        uint32_t min_centroids = 64;
        uint32_t max_centroids = 65536;
        uint32_t num_centroids = 64;
        uint32_t initial_bucket_cap = 128;
    };

    struct QuantizedSpaceConfig
    {
        uint32_t precision_bits = 8;
    };

    struct WhisperConfig
    {
        uint32_t cost_check = 1;
        uint32_t cost_echo_decode = 5;
        uint32_t cost_exact = 100;
        uint32_t base_budget_ops = 5000;
        uint32_t min_budget_ops = 250;
        uint32_t hot_query_floor = 2000;
        float budget_headroom = 1.2f;
        float latency_target_ms = 50.0f;
        float latency_ema_alpha = 0.15f;
        float cpu_soft_threshold = 75.0f;
        float cpu_hard_threshold = 90.0f;
        uint32_t refine_enable_margin_ms = 20;
    };

    struct HotTierConfig
    {
        size_t initial_capacity = 4096;
        uint32_t flush_interval_ms = 20;
    };

    struct IngestorConfig
    {
        size_t batch_size = 4096;
        size_t max_free_batches = 100;
        uint32_t num_workers = 1;
    };

    struct MetadataConfig
    {
        size_t initial_capacity = 65536;
        std::string delimiter = "=";
        bool enable_persistence = true;
    };

    struct MetricsConfig
    {
        bool enabled = true;                // Có thu thập metrics hay không
        uint32_t report_interval_ms = 5000; // Tần suất in báo cáo tóm tắt ra console/log
        bool verbose_arena_metrics = false; // Có in chi tiết các chỉ số của Arena không
    };

    struct OrchestratorConfig
    {
        uint32_t shard_count = 4;                 // 0 = auto (hardware concurrency)
        std::string shard_path_prefix = "shard_"; // Tiền tố tên thư mục shard
        bool use_parallel_merging = true;         // Có sử dụng async futures không
    };

    struct DBConfig
    {
        uint32_t bg_worker_interval_ms = 20;   //
        size_t default_membrance_ram_mb = 256; //
        std::string engine_type = "orbit";     //
        std::string manifest_file = "manifest.json";
    };

    struct SeedLayout
    {
        static constexpr uint64_t EXPIRY_MASK = 0xFFFFFFFFULL;
        static constexpr int KLEN_SHIFT = 32;
        static constexpr int VLEN_SHIFT = 48;
    };

    struct ShardConfig
    {
        uint64_t arena_size_mb = 2048;
        uint64_t map_slots = 1048576; // 1M slots mặc định
    };

    struct ShardManagerConfig
    {
        bool enable_cpu_pinning = true;
        uint64_t fallback_arena_mb = 512;
        size_t estimated_object_size = 64;
    };

    struct StorageConfig
    {
        size_t initial_arena_size_mb = 64;
        float growth_factor = 2.0f;
        size_t alignment = 8;
        size_t fallback_page_size = 4096;
        int default_file_permissions = 0644;
        bool prefer_fallocate = true;
    };

    struct ArenaConfig
    {
        float seed_region_ratio = 0.25f;             // Tỷ lệ vùng Seed trong Arena
        size_t max_remote_mmaps = 256;               // Số lượng file mmap tối đa được cache
        size_t demote_batch_bytes = 4 * 1024 * 1024; // 4MB per batch cho worker
        uint64_t min_blob_block = 64;                // Kích thước block tối thiểu
        size_t max_freelist_per_bucket = 4096;       // Tránh freelist phình quá to
        std::string remote_dir = "/tmp";             // Thư mục lưu blob demote
    };

    struct WalConfig
    {
        bool sync_on_append = true;
        size_t batch_commit_size = 0;
    };

    // [NEW] Cấu hình cho Data Split (Train/Val/Test)
    struct SplitConfig
    {
        std::string file_name = "splits.bin";
        float default_train_ratio = 0.8f;
        float default_val_ratio = 0.1f;
        float default_test_ratio = 0.1f;
    };

    // [NEW] Cấu hình cho Server (TCP/Epoll tuning)
    struct ServerConfig
    {
        int backlog = 1024;               // Kích thước hàng đợi kết nối (cho hàm listen)
        int max_events = 1024;            // Số lượng sự kiện tối đa xử lý trong một vòng lặp epoll
        int epoll_timeout_ms = 100;       // Thời gian chờ của epoll (ms)
        int cpu_sample_interval_ms = 200; // Chu kỳ lấy mẫu CPU cho WhisperGrain (ms)
    };

    // --- Compile-time Constants ---
    constexpr size_t SERVER_READ_BUFFER = 4096;

    // [RESTORED] Essential constants for Seed/Arena
    constexpr size_t SEED_PAYLOAD_BYTES = 48;
    constexpr size_t SEED_ALIGNMENT = 64;
    constexpr uint8_t SEED_OBJ_STRING = 0;
    constexpr uint8_t SEED_FLAG_INLINE = 0;
    constexpr uint8_t SEED_FLAG_INDIRECT = 0x1;

    // --- Main Configuration Struct (Nested) ---
    struct PomaiConfig
    {
        // Network Settings
        struct Net
        {
            uint16_t port = 7777;
        } net;

        // Resources & Storage
        struct Res
        {
            uint64_t arena_mb_per_shard = 2048;
            std::string data_root = "./data/pomai_db";
            uint64_t demote_async_max_pending = 1000; // Added for compatibility
        } res;

        // Core settings
        std::optional<uint64_t> rng_seed;

        // Sub-modules
        MapTuning map_tuning;
        WhisperConfig whisper;
        QuantizedSpaceConfig quantized_space;
        OrbitConfig orbit;
        FingerprintConfig fingerprint;
        NetworkCortexConfig network;
        NetworkCortexConfig cortex_cfg;
        HotTierConfig hot_tier;
        IngestorConfig ingestor;
        MetadataConfig metadata;
        MetricsConfig metrics;
        OrchestratorConfig orchestrator;
        DBConfig db;
        ShardConfig shard;
        ShardManagerConfig shard_manager;
        StorageConfig storage;
        ArenaConfig arena;
        WalConfig wal;
        ServerConfig server;
        SplitConfig split;

        PomaiConfig()
        {
            cortex_cfg = network;
        }
    };

    // Factory Function
    PomaiConfig load_from_args(int argc, char **argv);

} // namespace pomai::config