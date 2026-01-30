#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <pomai/core/types.h>

namespace pomai
{

    struct WhisperConfig
    {
        float latency_target_ms = 10.0f;
        float latency_ema_alpha = 0.1f;
        float cpu_soft_threshold = 70.0f;
        float cpu_hard_threshold = 90.0f;
        std::uint32_t base_budget_ops = 10000;
        std::uint32_t min_budget_ops = 1000;
        std::uint32_t hot_query_floor = 5000;
        float budget_headroom = 5.0f;
        std::uint32_t refine_enable_margin_ms = 2;
    };

    struct DbOptions
    {
        std::size_t dim{384};
        Metric metric{Metric::Cosine};
        std::size_t shards{4};
        std::size_t shard_queue_capacity{1024};
        std::string wal_dir{"./data"};
        WhisperConfig whisper;

        // New: index build pool threads (0 = auto)
        std::size_t index_build_threads{0};
        bool allow_sync_on_append{true};

        // Centroid persistence
        std::string centroids_path{};
        CentroidsLoadMode centroids_load_mode{CentroidsLoadMode::Auto};

        // NEW: search pool worker count (0 = auto)
        std::size_t search_pool_workers{0};
        std::size_t search_timeout_ms{500};

        // Metadata filter knobs
        std::size_t filtered_candidate_k{5000};
        std::uint32_t filter_expand_factor{4};
        std::uint32_t filter_max_visits{20000};
        std::uint64_t filter_time_budget_us{5000};
        std::size_t max_filtered_candidate_k{20000};
        std::uint32_t max_filter_graph_ef{2048};
        std::size_t max_filter_visits{100000};
        std::uint64_t max_filter_time_budget_us{50000};
        std::size_t filter_max_retries{3};
        std::size_t tag_dictionary_max_size{100000};
        std::size_t max_tags_per_vector{32};
        std::size_t max_filter_tags{64};

        // Compaction knobs
        std::size_t level_fanout{4};
        std::size_t max_concurrent_compactions{1};
        std::size_t compaction_trigger_threshold{4};

        // Scan knobs
        std::size_t scan_batch_cap{4096};
        std::size_t scan_id_order_max_rows{1000000};
    };

} // namespace pomai
