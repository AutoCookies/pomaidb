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

        void Clear() { 
            hits.clear(); 
            errors.clear(); 
        }
    };

} // namespace pomai
