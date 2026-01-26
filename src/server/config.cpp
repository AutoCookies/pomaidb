#include "pomai/server/config.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pomai::server
{

    static std::string Trim(const std::string &s)
    {
        std::size_t a = 0;
        while (a < s.size() && (s[a] == ' ' || s[a] == '\t'))
            ++a;
        std::size_t b = s.size();
        while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r'))
            --b;
        return s.substr(a, b - a);
    }

    ServerConfig LoadConfigFile(const std::string &path)
    {
        ServerConfig cfg;
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("cannot open config: " + path);

        std::string line;
        while (std::getline(in, line))
        {
            line = Trim(line);
            if (line.empty() || line[0] == '#')
                continue;

            auto pos = line.find(':');
            if (pos == std::string::npos)
                continue;

            std::string key = Trim(line.substr(0, pos));
            std::string val = Trim(line.substr(pos + 1));

            // Remove optional quotes
            if (!val.empty() && val.front() == '"' && val.back() == '"' && val.size() >= 2)
            {
                val = val.substr(1, val.size() - 2);
            }

            if (key == "data_dir")
                cfg.data_dir = val;
            else if (key == "listen_host")
                cfg.listen_host = val;
            else if (key == "listen_port")
                cfg.listen_port = static_cast<std::uint16_t>(std::stoul(val));
            else if (key == "unix_socket")
                cfg.unix_socket = val;

            else if (key == "shards")
                cfg.shards = static_cast<std::size_t>(std::stoull(val));
            else if (key == "shard_queue_capacity")
                cfg.shard_queue_capacity = static_cast<std::size_t>(std::stoull(val));

            else if (key == "default_dim")
                cfg.default_dim = static_cast<std::size_t>(std::stoull(val));
            else if (key == "default_metric")
                cfg.default_metric = val;

            else if (key == "log_path")
                cfg.log_path = val;
            else if (key == "log_level")
                cfg.log_level = val;
        }

        return cfg;
    }

} // namespace pomai::server
