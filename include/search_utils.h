#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "types.h"

namespace pomai
{
    inline void SortAndDedupeResults(std::vector<SearchResultItem> &items, std::size_t k)
    {
        if (items.empty())
            return;
        std::sort(items.begin(), items.end(), [](const SearchResultItem &a, const SearchResultItem &b)
                  {
                      if (a.score == b.score)
                          return a.id < b.id;
                      return a.score > b.score;
                  });
        std::unordered_set<Id> seen;
        seen.reserve(items.size());
        std::vector<SearchResultItem> deduped;
        deduped.reserve(items.size());
        for (const auto &item : items)
        {
            if (seen.insert(item.id).second)
            {
                deduped.push_back(item);
                if (deduped.size() >= k)
                    break;
            }
        }
        items.swap(deduped);
    }
} // namespace pomai
