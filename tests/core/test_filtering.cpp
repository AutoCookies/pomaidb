#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "tests/common/test_utils.h"

#include <unordered_set>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Search filter namespaces only", "[core][filter]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 6, 2);
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.push_back(MakeUpsert(1, 6, 0.1f, 10));
    batch.push_back(MakeUpsert(2, 6, 0.2f, 20));
    batch.push_back(MakeUpsert(3, 6, 0.3f, 10));

    db.UpsertBatch(batch, true).get();

    Filter filter;
    filter.namespace_id = 10;
    SearchRequest req = MakeSearchRequest(batch[0].vec, 10);
    req.filter = std::make_shared<Filter>(filter);

    auto resp = db.Search(req);
    db.Stop();

    REQUIRE_FALSE(resp.items.empty());
    for (const auto &item : resp.items)
        REQUIRE(item.id == 1 || item.id == 3);
}

TEST_CASE("Search filter tags any/all", "[core][filter]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 6, 1);
    PomaiDB db(opts);
    db.Start();

    std::vector<UpsertRequest> batch;
    batch.push_back(MakeUpsert(1, 6, 0.1f, 0, {1, 2}));
    batch.push_back(MakeUpsert(2, 6, 0.2f, 0, {2, 3}));
    batch.push_back(MakeUpsert(3, 6, 0.3f, 0, {3, 4}));
    db.UpsertBatch(batch, true).get();

    Filter any_filter;
    any_filter.require_any_tags = {1, 4};
    SearchRequest any_req = MakeSearchRequest(batch[0].vec, 10);
    any_req.filter = std::make_shared<Filter>(any_filter);
    auto any_resp = db.Search(any_req);

    Filter all_filter;
    all_filter.require_all_tags = {2, 3};
    SearchRequest all_req = MakeSearchRequest(batch[1].vec, 10);
    all_req.filter = std::make_shared<Filter>(all_filter);
    auto all_resp = db.Search(all_req);

    db.Stop();

    std::unordered_set<Id> any_ids;
    for (const auto &item : any_resp.items)
        any_ids.insert(item.id);
    REQUIRE(any_ids.count(1) == 1);
    REQUIRE(any_ids.count(3) == 1);

    REQUIRE_FALSE(all_resp.items.empty());
    for (const auto &item : all_resp.items)
        REQUIRE(item.id == 2);
}
