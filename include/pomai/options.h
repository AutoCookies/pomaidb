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

    enum class MembraneKind : uint8_t
    {
        kVector = 0,
        kRag = 1,
    };

    struct IndexParams
    {
        uint32_t nlist = 64;
        uint32_t nprobe = 16;
    };

    struct DBOptions
    {
        std::string path;
        uint32_t shard_count = 4;
        uint32_t dim = 512;
        uint32_t search_threads = 0; // 0 => auto
        FsyncPolicy fsync = FsyncPolicy::kNever;
        IndexParams index_params;
        bool routing_enabled = false;
        uint32_t routing_k = 0;
        uint32_t routing_probe = 0;
        uint32_t routing_warmup_mult = 20;
        uint32_t routing_keep_prev = 1;
    };

    // One membrane = one logical collection.
    struct MembraneSpec
    {
        std::string name;
        uint32_t shard_count = 0; // 0 => inherit DBOptions.shard_count
        uint32_t dim = 0;         // 0 => inherit DBOptions.dim
        MetricType metric = MetricType::kL2;
        IndexParams index_params;
        MembraneKind kind = MembraneKind::kVector;
    };

} // namespace pomai
