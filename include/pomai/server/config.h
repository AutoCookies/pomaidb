#pragma once
#include <cstdint>
#include <string>

#include <pomai/api/options.h>

namespace pomai::server
{

    using pomai::WhisperConfig;

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
        bool dev_mode{false};

        WhisperConfig whisper;
        bool allow_sync_on_append{true};
    };

    ServerConfig LoadConfigFile(const std::string &path);

}
