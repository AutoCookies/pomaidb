#pragma once
#include <cstdint>
#include <string>

namespace pomai
{

    enum class FsyncPolicy : uint8_t
    {
        kNever = 0,
        kAlways = 1,
    };

    struct DBOptions
    {
        std::string path;
        uint32_t shard_count = 4;
        uint32_t dim = 512;
        FsyncPolicy fsync = FsyncPolicy::kNever;
    };

    // One membrane = one logical collection.
    struct MembraneSpec
    {
        std::string name;
        uint32_t shard_count = 0; // 0 => inherit DBOptions.shard_count
        uint32_t dim = 0;         // 0 => inherit DBOptions.dim
    };

} // namespace pomai
