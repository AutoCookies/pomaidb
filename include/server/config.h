#pragma once
#include <cstdint>
#include <string>

namespace pomai::server
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

    struct ServerConfig
    {
        std::string data_dir{"/var/lib/pomai"};
        std::string listen_host{"127.0.0.1"};
        std::uint16_t listen_port{7733};
        std::string unix_socket{"/tmp/pomai.sock"};

        std::size_t shards{4};
        std::size_t shard_queue_capacity{4096};

        std::size_t default_dim{128};
        std::string default_metric{"cosine"};

        std::string log_path{"/var/log/pomai/pomai.log"};
        std::string log_level{"info"};

        WhisperConfig whisper;
    };

    ServerConfig LoadConfigFile(const std::string &path);

}