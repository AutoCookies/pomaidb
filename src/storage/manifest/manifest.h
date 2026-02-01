#pragma once
#include <cstdint>
#include <string>
#include "pomai/status.h"

namespace pomai::storage
{

    class Manifest
    {
    public:
        static pomai::Status EnsureInitialized(const std::string &db_path,
                                               std::uint32_t shard_count,
                                               std::uint32_t dim);
    };

} // namespace pomai::storage
