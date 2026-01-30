#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include <pomai/core/status.h>
#include <pomai/core/types.h>

namespace pomai
{
    inline void SortAndDedupeTags(std::vector<TagId> &tags)
    {
        std::sort(tags.begin(), tags.end());
        tags.erase(std::unique(tags.begin(), tags.end()), tags.end());
    }

    inline Result<Filter> NormalizeFilter(const Filter &filter, std::size_t max_filter_tags)
    {
        Filter out = filter;
        SortAndDedupeTags(out.require_all_tags);
        SortAndDedupeTags(out.require_any_tags);
        SortAndDedupeTags(out.exclude_tags);
        const std::size_t total_tags = out.require_all_tags.size() + out.require_any_tags.size() + out.exclude_tags.size();
        if (total_tags > max_filter_tags)
            return Result<Filter>(Status::Invalid("filter tags exceeded configured limit"));
        return Result<Filter>(std::move(out));
    }
} // namespace pomai
