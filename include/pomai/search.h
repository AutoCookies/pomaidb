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

    struct SearchResult
    {
        std::vector<SearchHit> hits;

        void Clear() { hits.clear(); }
    };

} // namespace pomai
