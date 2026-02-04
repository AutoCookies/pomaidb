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
        // Example parameters for IVF or HNSW
        uint32_t num_lists = 100;
        uint32_t ef_construction = 200;
        uint32_t ef_search = 50;
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
        MetricType metric = MetricType::kL2;
        IndexParams index_params;
    };

} // namespace pomai
