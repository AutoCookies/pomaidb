#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pomai
{

    using Id = std::uint64_t;
    using Lsn = std::uint64_t;

    enum class Metric : std::uint8_t
    {
        L2 = 0,
        Cosine = 1
    };

    struct Vector
    {
        std::vector<float> data;
    };

    struct UpsertRequest
    {
        Id id{};
        Vector vec;
    };

    struct SearchRequest
    {
        Vector query;
        std::size_t topk{100};
    };

    struct SearchResultItem
    {
        Id id{};
        float score{};
    };

    struct SearchResponse
    {
        std::vector<SearchResultItem> items;
    };
} // namespace pomai
