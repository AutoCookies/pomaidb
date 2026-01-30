#include <catch2/catch.hpp>

#include <pomai/index/orbit_index.h>

#include "common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("OrbitIndex finds exact vector", "[core][orbit]")
{
    const std::size_t dim = 4;
    std::vector<float> flat;
    std::vector<Id> ids;
    for (std::size_t i = 0; i < 16; ++i)
    {
        auto vec = MakeVector(dim, static_cast<float>(i));
        ids.push_back(static_cast<Id>(i + 1));
        flat.insert(flat.end(), vec.data.begin(), vec.data.end());
    }

    pomai::core::OrbitIndex index(dim, 8, 32);
    index.Build(flat, ids);

    SearchRequest req;
    req.query = MakeVector(dim, 3.0f);
    req.topk = 5;
    req.metric = Metric::L2;

    pomai::ai::Budget budget{};
    budget.ops_budget = 5000;
    budget.bucket_budget = 10;
    budget.allow_exact_refine = true;

    auto resp = index.Search(req, budget);
    REQUIRE_FALSE(resp.items.empty());
    REQUIRE(resp.items.front().id == 4);
}

TEST_CASE("OrbitIndex filtered search respects namespace", "[core][orbit]")
{
    const std::size_t dim = 4;
    Seed seed(dim);
    std::vector<UpsertRequest> batch;
    batch.push_back(MakeUpsert(1, dim, 0.1f, 10, {1, 2}));
    batch.push_back(MakeUpsert(2, dim, 0.2f, 20, {2, 3}));
    batch.push_back(MakeUpsert(3, dim, 0.3f, 10, {3, 4}));
    seed.ApplyUpserts(batch);
    auto snap = seed.MakeSnapshot();

    std::vector<float> flat;
    std::vector<Id> ids;
    for (const auto &row : batch)
    {
        ids.push_back(row.id);
        flat.insert(flat.end(), row.vec.data.begin(), row.vec.data.end());
    }

    pomai::core::OrbitIndex index(dim, 8, 32);
    index.Build(flat, ids);

    SearchRequest req;
    req.query = batch[0].vec;
    req.topk = 5;
    req.metric = Metric::L2;

    Filter filter;
    filter.namespace_id = 10;

    pomai::ai::Budget budget{};
    budget.ops_budget = 5000;
    budget.bucket_budget = 10;
    budget.allow_exact_refine = true;

    auto resp = index.SearchFiltered(req, budget, filter, *snap);
    REQUIRE_FALSE(resp.items.empty());
    for (const auto &item : resp.items)
        REQUIRE(item.id == 1 || item.id == 3);
}
