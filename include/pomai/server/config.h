#pragma once
#include <cstdint>
#include <string>

namespace pomai::server
{

    struct ServerConfig
    {
        std::string data_dir{"/var/lib/pomai"};
        std::string listen_host{"127.0.0.1"};
        std::uint16_t listen_port{7733};
        std::string unix_socket{"/tmp/pomai.sock"};

        std::size_t shards{4};
        std::size_t shard_queue_capacity{4096};

        std::size_t default_dim{128};
        std::string default_metric{"cosine"}; // "cosine" or "l2"

        std::string log_path{"/var/log/pomai/pomai.log"};
        std::string log_level{"info"};
    };

    ServerConfig LoadConfigFile(const std::string &path);

} // namespace pomai::server
