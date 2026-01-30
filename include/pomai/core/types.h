#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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

    enum class SearchMode : std::uint8_t
    {
        Latency = 0,
        Quality = 1
    };

    enum class SearchStatus : std::uint8_t
    {
        Ok = 0,
        InsufficientResults = 1
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
        std::vector<TagId> tag_ids;
    };

    struct Filter
    {
        std::optional<std::uint32_t> namespace_id;
        std::vector<TagId> require_all_tags;
        std::vector<TagId> require_any_tags;
        std::vector<TagId> exclude_tags;
        bool match_none{false};

        bool empty() const
        {
            return !match_none && !namespace_id &&
                   require_all_tags.empty() && require_any_tags.empty() && exclude_tags.empty();
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
        std::uint64_t filter_time_budget_us{0};
        SearchMode search_mode{SearchMode::Latency};
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
        bool filtered_time_budget_hit{false};
        bool filtered_visit_budget_hit{false};
        bool filtered_budget_exhausted{false};
        std::size_t filtered_candidates_generated{0};
        std::size_t filtered_candidates_passed_filter{0};
        std::size_t filtered_reranked{0};
        std::size_t filtered_candidates{0};
        std::size_t filtered_visits{0};
        std::size_t filtered_missing_hits{0};
        std::size_t filtered_retries{0};
        std::size_t filtered_candidate_k{0};
        std::uint32_t filtered_graph_ef{0};
        std::size_t filtered_max_visits{0};
        std::uint64_t filtered_time_budget_us{0};
        bool filtered_candidate_cap_hit{false};
        bool filtered_graph_ef_cap_hit{false};
        bool filtered_visit_cap_hit{false};
        bool filtered_time_cap_hit{false};
        double filtered_selectivity{0.0};
        bool quality_failure{false};
    };

    struct SearchResponse
    {
        std::vector<SearchResultItem> items;
        bool partial{false};
        SearchStats stats;
        SearchStatus status{SearchStatus::Ok};
        std::string error;
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
