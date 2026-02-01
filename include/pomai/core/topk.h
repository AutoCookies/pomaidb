#pragma once
#include <algorithm>
#include <vector>

#include "pomai/types.h"

namespace pomai::core
{

    inline void MergeTopK(std::vector<pomai::SearchHit> &dst,
                          std::vector<pomai::SearchHit> &&src,
                          std::size_t k)
    {
        dst.insert(dst.end(),
                   std::make_move_iterator(src.begin()),
                   std::make_move_iterator(src.end()));

        if (k == 0)
            return;
        if (dst.size() > k)
        {
            using Diff = typename std::vector<pomai::SearchHit>::difference_type;
            std::nth_element(dst.begin(), dst.begin() + static_cast<Diff>(k), dst.end(),
                             [](auto &a, auto &b)
                             { return a.score > b.score; });
            dst.resize(k);
        }
        std::sort(dst.begin(), dst.end(),
                  [](auto &a, auto &b)
                  { return a.score > b.score; });
    }

} // namespace pomai::core
