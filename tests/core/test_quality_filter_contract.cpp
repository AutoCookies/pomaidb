#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Quality namespace filter returns full results without cap hits", "[core][filter][quality]")
{
    TempDir dir;
    const std::size_t dim = 8;
    auto opts = DefaultDbOptions(dir.str(), dim, 1);
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.reserve(100);
    for (std::size_t i = 0; i < 100; ++i)
    {
        std::uint32_t ns = static_cast<std::uint32_t>(i % 10);
        batch.push_back(MakeUpsert(static_cast<Id>(i + 1), dim, 0.1f + static_cast<float>(i) * 0.01f, ns));
    }
    db.UpsertBatch(batch, true).get();

    Filter filter;
    filter.namespace_id = 3;
    SearchRequest req = MakeSearchRequest(batch[3].vec, 5);
    req.search_mode = SearchMode::Quality;
    req.filter = std::make_shared<Filter>(filter);

    auto resp = db.Search(req);
    db.Stop();

    REQUIRE(resp.status == SearchStatus::Ok);
    REQUIRE(resp.items.size() >= req.topk);
    REQUIRE_FALSE(resp.partial);
    REQUIRE_FALSE(resp.stats.filtered_partial);
    REQUIRE(resp.stats.filtered_missing_hits == 0);
    REQUIRE_FALSE(resp.stats.filtered_candidate_cap_hit);
    REQUIRE_FALSE(resp.stats.filtered_graph_ef_cap_hit);
    REQUIRE_FALSE(resp.stats.filtered_visit_cap_hit);
    REQUIRE_FALSE(resp.stats.filtered_time_cap_hit);
}

TEST_CASE("Quality filtered search expands to satisfy low selectivity", "[core][filter][quality]")
{
    TempDir dir;
    const std::size_t dim = 8;
    auto opts = DefaultDbOptions(dir.str(), dim, 1);
    opts.filtered_candidate_k = 5;
    opts.filter_max_visits = 5;
    opts.max_filter_visits = 160;
    opts.max_filtered_candidate_k = 10;
    opts.filter_max_retries = 5;
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.reserve(200);
    for (std::size_t i = 0; i < 200; ++i)
    {
        std::vector<TagId> tags;
        if (i % 16 == 0)
            tags = {1, 2};
        else
            tags = {static_cast<TagId>(i % 5)};
        batch.push_back(MakeUpsert(static_cast<Id>(i + 1), dim, 0.2f + static_cast<float>(i) * 0.01f, 0, tags));
    }
    db.UpsertBatch(batch, true).get();

    Filter filter;
    filter.require_all_tags = {1, 2};
    SearchRequest req = MakeSearchRequest(batch[0].vec, 5);
    req.search_mode = SearchMode::Quality;
    req.filter = std::make_shared<Filter>(filter);

    auto resp = db.Search(req);
    db.Stop();

    REQUIRE(resp.status == SearchStatus::Ok);
    REQUIRE(resp.items.size() >= req.topk);
    REQUIRE_FALSE(resp.partial);
    REQUIRE_FALSE(resp.stats.filtered_partial);
    REQUIRE(resp.stats.filtered_retries > 0);
    REQUIRE(resp.stats.filtered_missing_hits == 0);
}

TEST_CASE("Latency mode reports partials and budget hits", "[core][filter][latency]")
{
    TempDir dir;
    const std::size_t dim = 8;
    auto opts = DefaultDbOptions(dir.str(), dim, 1);
    opts.filtered_candidate_k = 5;
    opts.filter_max_visits = 5;
    opts.max_filter_visits = 5;
    opts.max_filtered_candidate_k = 5;
    opts.filter_max_retries = 2;
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.reserve(200);
    for (std::size_t i = 0; i < 200; ++i)
    {
        std::vector<TagId> tags;
        if (i % 20 == 0)
            tags = {1, 2};
        else
            tags = {static_cast<TagId>(i % 5)};
        batch.push_back(MakeUpsert(static_cast<Id>(i + 1), dim, 0.3f + static_cast<float>(i) * 0.01f, 0, tags));
    }
    db.UpsertBatch(batch, true).get();

    Filter filter;
    filter.require_all_tags = {1, 2};
    SearchRequest req = MakeSearchRequest(batch[0].vec, 5);
    req.search_mode = SearchMode::Latency;
    req.filter = std::make_shared<Filter>(filter);

    auto resp = db.Search(req);
    db.Stop();

    REQUIRE(resp.status == SearchStatus::Ok);
    REQUIRE(resp.partial);
    REQUIRE(resp.stats.filtered_missing_hits > 0);
    REQUIRE(resp.stats.filtered_budget_exhausted);
    REQUIRE(resp.stats.filtered_visit_cap_hit);
}

TEST_CASE("Quality mode fails loudly on insufficient filtered results", "[core][filter][quality]")
{
    TempDir dir;
    const std::size_t dim = 8;
    auto opts = DefaultDbOptions(dir.str(), dim, 1);
    opts.filtered_candidate_k = 5;
    opts.filter_max_visits = 5;
    opts.max_filter_visits = 5;
    opts.max_filtered_candidate_k = 5;
    opts.filter_max_retries = 1;
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.reserve(200);
    for (std::size_t i = 0; i < 200; ++i)
    {
        std::vector<TagId> tags;
        if (i % 20 == 0)
            tags = {1, 2};
        else
            tags = {static_cast<TagId>(i % 5)};
        batch.push_back(MakeUpsert(static_cast<Id>(i + 1), dim, 0.4f + static_cast<float>(i) * 0.01f, 0, tags));
    }
    db.UpsertBatch(batch, true).get();

    Filter filter;
    filter.require_all_tags = {1, 2};
    SearchRequest req = MakeSearchRequest(batch[0].vec, 5);
    req.search_mode = SearchMode::Quality;
    req.filter = std::make_shared<Filter>(filter);

    auto resp = db.Search(req);
    db.Stop();

    REQUIRE(resp.status == SearchStatus::InsufficientResults);
    REQUIRE(resp.stats.quality_failure);
    REQUIRE(resp.items.empty());
    REQUIRE_FALSE(resp.partial);
    REQUIRE_FALSE(resp.error.empty());
}
