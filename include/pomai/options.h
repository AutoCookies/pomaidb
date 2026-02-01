#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace pomai
{

    enum class FsyncPolicy : std::uint8_t
    {
        kNever = 0,
        kOnFlush = 1,
        kAlways = 2,
    };

    struct DBOptions
    {
        std::string path;
        std::uint32_t shard_count = 0;
        std::uint32_t dim = 0;
        std::size_t arena_block_bytes = 1ULL << 20;
        std::size_t wal_segment_bytes = 64ULL << 20;
        FsyncPolicy fsync = FsyncPolicy::kOnFlush;
    };

} // namespace pomai
