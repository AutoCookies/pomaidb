#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pomai
{

    using Id = std::uint64_t;
    using Lsn = std::uint64_t;
    using TagId = std::uint32_t;

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

    struct Metadata
    {
        std::uint32_t namespace_id{0};
        std::string namespace_name{};
        std::vector<TagId> tag_ids;
        std::vector<std::string> tags;
    };

    struct Filter
    {
        std::optional<std::uint32_t> namespace_id;
        std::string namespace_name{};
        std::vector<TagId> require_all_tags;
        std::vector<TagId> require_any_tags;
        std::vector<TagId> exclude_tags;
        std::vector<std::string> require_all_tag_names;
        std::vector<std::string> require_any_tag_names;
        std::vector<std::string> exclude_tag_names;
        std::vector<std::pair<std::string, std::string>> kv_equals;
        bool match_none{false};

        bool empty() const
        {
            return !match_none && !namespace_id &&
                   namespace_name.empty() &&
                   require_all_tags.empty() && require_any_tags.empty() && exclude_tags.empty() &&
                   require_all_tag_names.empty() && require_any_tag_names.empty() && exclude_tag_names.empty() &&
                   kv_equals.empty();
        }
    };

    struct UpsertRequest
    {
        Id id{};
        Vector vec;
        Metadata metadata;
    };

    struct SearchRequest
    {
        Vector query;
        std::size_t topk{100};
        std::size_t candidate_k{0};
        std::size_t max_rerank_k{2000};
        std::uint32_t graph_ef{0};
        Metric metric{Metric::L2};
        std::size_t filtered_candidate_k{0};
        std::uint32_t filter_expand_factor{0};
        std::uint32_t filter_max_visits{0};
        std::shared_ptr<const Filter> filter;
    };

    struct SearchResultItem
    {
        Id id{};
        float score{};
    };

    struct SearchStats
    {
        bool partial{false};
        bool filtered_partial{false};
        std::size_t filtered_candidates{0};
    };

    struct SearchResponse
    {
        std::vector<SearchResultItem> items;
        bool partial{false};
        SearchStats stats;
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
