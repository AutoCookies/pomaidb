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

    enum class MetricType : uint8_t
    {
        kL2 = 0,
        kInnerProduct = 1,
        kCosine = 2,
    };

    struct IndexParams
    {
        uint32_t wsbr_block_size = 512;
        uint32_t wsbr_top_blocks = 4096;
        uint32_t wsbr_widen_blocks = 32;
    };

    struct DBOptions
    {
        std::string path;
        uint32_t shard_count = 4;
        uint32_t dim = 512;
        FsyncPolicy fsync = FsyncPolicy::kNever;
        IndexParams index_params;
    };

    // One membrane = one logical collection.
    struct MembraneSpec
    {
        std::string name;
        uint32_t shard_count = 0; // 0 => inherit DBOptions.shard_count
        uint32_t dim = 0;         // 0 => inherit DBOptions.dim
        MetricType metric = MetricType::kL2;
        IndexParams index_params;
    };

} // namespace pomai
