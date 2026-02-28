#pragma once
#include <cstdint>
#include <vector>

#include "types.h"

namespace pomai
{

    struct SearchHit
    {
        VectorId id = 0;
        float score = 0.0f; // higher is better
    };

    struct ShardError
    {
        uint32_t shard_id;
        std::string message;
    };

    struct SemanticPointer {
        const void* raw_data_ptr = nullptr;
        uint32_t dim = 0;
        float quant_min = 0.0f;
        float quant_inv_scale = 0.0f;
        int quant_type = 0; // 0=None, 1=SQ8, 2=FP16
        uint64_t session_id = 0;
    };

    struct SearchResult
    {
        std::vector<SearchHit> hits;
        std::vector<ShardError> errors; // Partial failures
        uint32_t routed_shards_count = 0;
        uint32_t routing_probe_centroids = 0;
        uint64_t routed_buckets_count = 0; // Candidate/bucket count when routing enabled.

        std::vector<SemanticPointer> zero_copy_pointers;
        uint64_t zero_copy_session_id = 0;

        void Clear() {
            hits.clear();
            errors.clear();
            routed_shards_count = 0;
            routing_probe_centroids = 0;
            routed_buckets_count = 0;
            zero_copy_pointers.clear();
            zero_copy_session_id = 0;
        }
    };

} // namespace pomai
