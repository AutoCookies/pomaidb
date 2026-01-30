#pragma once
#include <algorithm>
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


    enum class CentroidsLoadMode : std::uint8_t
    {
        Auto,
        Sync,
        Async,
        None
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
        std::size_t candidate_k{0};
        std::size_t max_rerank_k{2000};
        std::uint32_t graph_ef{0};
        Metric metric{Metric::L2};
    };

    struct SearchResultItem
    {
        Id id{};
        float score{};
    };

    struct SearchResponse
    {
        std::vector<SearchResultItem> items;
        bool partial{false};
    };

    inline std::size_t NormalizeMaxRerankK(const SearchRequest &req)
    {
        constexpr std::size_t kHardMax = 2000;
        std::size_t max_k = req.max_rerank_k == 0 ? kHardMax : req.max_rerank_k;
        max_k = std::min(max_k, kHardMax);
        if (req.topk > max_k)
            max_k = req.topk;
        return max_k;
    }

    inline std::size_t NormalizeCandidateK(const SearchRequest &req)
    {
        std::size_t max_k = NormalizeMaxRerankK(req);
        std::size_t candidate = req.candidate_k;
        if (candidate == 0)
            candidate = std::max<std::size_t>(req.topk * 20, 200);
        candidate = std::max(candidate, req.topk);
        candidate = std::min(candidate, max_k);
        return candidate;
    }

    inline std::uint32_t NormalizeGraphEf(const SearchRequest &req, std::size_t candidate_k)
    {
        constexpr std::uint32_t kDefaultEf = 256;
        constexpr std::uint32_t kMinEf = 64;
        constexpr std::uint32_t kMaxEf = 2048;
        std::uint32_t ef = req.graph_ef == 0 ? kDefaultEf : req.graph_ef;
        ef = std::min(ef, kMaxEf);
        ef = std::max(ef, kMinEf);
        if (candidate_k > ef)
            ef = static_cast<std::uint32_t>(std::min<std::size_t>(candidate_k, kMaxEf));
        ef = std::max(ef, kMinEf);
        return ef;
    }
} // namespace pomai
