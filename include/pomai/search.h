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

    struct SearchResult
    {
        std::vector<SearchHit> hits;
        std::vector<ShardError> errors; // Partial failures
        uint32_t routed_shards_count = 0;
        uint32_t routing_probe_centroids = 0;
        uint64_t routed_buckets_count = 0; // Candidate/bucket count when routing enabled.

        void Clear() {
            hits.clear();
            errors.clear();
            routed_shards_count = 0;
            routing_probe_centroids = 0;
            routed_buckets_count = 0;
        }
    };

} // namespace pomai
