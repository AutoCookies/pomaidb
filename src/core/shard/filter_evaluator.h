#pragma once
#include "pomai/metadata.h"

namespace pomai::core
{
    class FilterEvaluator
    {
    public:
        static bool Matches(const Metadata& meta, const SearchOptions& opts)
        {
            return opts.Matches(meta);
        }
    };
}
