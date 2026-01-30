#include <catch2/catch.hpp>

#include <pomai/core/seed.h>

#include "common/test_utils.h"

#include <unordered_set>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Seed search returns exact self", "[core][seed]")
{
    const std::size_t dim = 8;
    Seed seed(dim);
    auto batch = MakeBatch(20, dim, 0.2f);
    seed.ApplyUpserts(batch);

    auto snap = seed.MakeSnapshot();
    auto query = batch[7].vec;
    auto req = MakeSearchRequest(query, 5);
    auto resp = Seed::SearchSnapshot(snap, req);

    REQUIRE_FALSE(resp.items.empty());
    REQUIRE(resp.items.front().id == batch[7].id);
}

TEST_CASE("Seed search has no duplicate ids", "[core][seed]")
{
    const std::size_t dim = 8;
    Seed seed(dim);
    auto batch = MakeBatch(50, dim, 1.0f);
    seed.ApplyUpserts(batch);

    auto snap = seed.MakeSnapshot();
    auto req = MakeSearchRequest(batch[0].vec, 20);
    auto resp = Seed::SearchSnapshot(snap, req);

    std::unordered_set<Id> seen;
    for (const auto &item : resp.items)
    {
        REQUIRE(seen.insert(item.id).second);
    }
}

TEST_CASE("Seed search tie-breaks by id", "[core][seed]")
{
    const std::size_t dim = 4;
    Seed seed(dim);
    std::vector<UpsertRequest> batch;
    batch.push_back(MakeUpsert(42, dim, 0.1f));
    batch.push_back(MakeUpsert(7, dim, 0.1f));
    seed.ApplyUpserts(batch);

    auto snap = seed.MakeSnapshot();
    auto req = MakeSearchRequest(batch[0].vec, 2);
    auto resp = Seed::SearchSnapshot(snap, req);

    REQUIRE(resp.items.size() >= 2);
    REQUIRE(resp.items[0].id == 7);
    REQUIRE(resp.items[1].id == 42);
}

TEST_CASE("Seed rerank matches brute force", "[core][seed]")
{
    const std::size_t dim = 6;
    Seed seed(dim);
    auto batch = MakeBatch(15, dim, 0.25f);
    seed.ApplyUpserts(batch);

    auto snap = seed.MakeSnapshot();
    Vector query = MakeVector(dim, 0.55f);
    auto req = MakeSearchRequest(query, 5);
    auto resp = Seed::SearchSnapshot(snap, req);
    auto brute = BruteForceL2(batch, query, 5);

    REQUIRE(resp.items.size() == brute.size());
    for (std::size_t i = 0; i < brute.size(); ++i)
        REQUIRE(resp.items[i].id == brute[i].id);
}
