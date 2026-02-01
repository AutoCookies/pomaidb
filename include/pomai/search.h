#pragma once
#include <cstdint>
#include "pomai/types.h"

namespace pomai
{

    struct SearchHit
    {
        VectorId id = 0;
        float score = 0.0f;
    };

} // namespace pomai
