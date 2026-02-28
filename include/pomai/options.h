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

    enum class IndexType : uint8_t
    {
        kIvfFlat = 0,
        kHnsw = 1,
    };

    enum class QuantizationType : uint8_t
    {
        kNone = 0,
        kSq8 = 1,
        kFp16 = 2,
    };

    struct IndexParams
    {
        IndexType type = IndexType::kIvfFlat;
        // IVF Params
        uint32_t nlist = 64;
        uint32_t nprobe = 16;
        // HNSW Params
        uint32_t hnsw_m = 32;
        uint32_t hnsw_ef_construction = 200;
        uint32_t hnsw_ef_search = 64;
        // Adaptive dispatcher: segments with fewer vectors use brute-force SIMD
        // (guaranteeing 100% recall). Larger segments use HNSW graph traversal.
        // Default: 0 = always use HNSW when available (rely on ef_search for recall).
        uint32_t adaptive_threshold = 5000;
        QuantizationType quant_type = QuantizationType::kNone;
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
        MetricType metric = MetricType::kL2;
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
