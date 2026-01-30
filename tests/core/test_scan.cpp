#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "common/test_utils.h"

#include <unordered_set>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Scan returns all ids once", "[core][scan]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 5, 2);
    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    auto batch = MakeBatch(40, 5, 0.1f, 1);
    auto upsert_res = db.UpsertBatch(batch, true).get();
    REQUIRE(upsert_res.ok());

    ScanRequest req;
    req.batch_size = 7;
    req.include_vectors = true;
    req.include_metadata = true;
    req.order = ScanOrder::IdAsc;

    std::unordered_set<Id> seen;
    std::string cursor;
    while (true)
    {
        req.cursor = cursor;
        auto resp_res = db.Scan(req);
        REQUIRE(resp_res.ok());
        auto resp = resp_res.move_value();
        REQUIRE(resp.status == ScanStatus::Ok);
        for (const auto &item : resp.items)
        {
            REQUIRE(seen.insert(item.id).second);
            REQUIRE(item.vector_offset + opts.dim <= resp.vectors.size());
            REQUIRE(item.tag_offset + item.tag_count <= resp.tags.size());
        }
        if (resp.next_cursor.empty())
            break;
        cursor = resp.next_cursor;
    }

    REQUIRE(db.Stop().ok());

    REQUIRE(seen.size() == batch.size());
}

TEST_CASE("Scan cursor snapshot consistency", "[core][scan]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 5, 1);
    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    auto batch = MakeBatch(20, 5, 0.1f, 1);
    auto upsert_res = db.UpsertBatch(batch, true).get();
    REQUIRE(upsert_res.ok());

    ScanRequest req;
    req.batch_size = 5;
    req.consistency = ScanConsistency::ConsistentSnapshot;

    auto resp_res = db.Scan(req);
    REQUIRE(resp_res.ok());
    auto resp = resp_res.move_value();
    REQUIRE(resp.status == ScanStatus::Ok);
    REQUIRE_FALSE(resp.items.empty());

    auto extra = MakeUpsert(999, 5, 9.9f, 1);
    auto upsert_extra = db.UpsertBatch({extra}, true).get();
    REQUIRE(upsert_extra.ok());

    std::unordered_set<Id> seen;
    for (const auto &item : resp.items)
        seen.insert(item.id);

    std::string cursor = resp.next_cursor;
    while (!cursor.empty())
    {
        req.cursor = cursor;
        auto next_res = db.Scan(req);
        REQUIRE(next_res.ok());
        auto next = next_res.move_value();
        REQUIRE(next.status == ScanStatus::Ok);
        for (const auto &item : next.items)
            seen.insert(item.id);
        cursor = next.next_cursor;
    }

    REQUIRE(db.Stop().ok());

    REQUIRE(seen.count(999) == 0);
    REQUIRE(seen.size() == batch.size());
}

TEST_CASE("Scan respects filters", "[core][scan]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 5, 1);
    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    std::vector<UpsertRequest> batch;
    batch.push_back(MakeUpsert(1, 5, 0.1f, 1, {1}));
    batch.push_back(MakeUpsert(2, 5, 0.2f, 2, {2}));
    batch.push_back(MakeUpsert(3, 5, 0.3f, 1, {2}));
    auto upsert_res = db.UpsertBatch(batch, true).get();
    REQUIRE(upsert_res.ok());

    ScanRequest req;
    req.batch_size = 10;
    req.filter.namespace_id = 1;

    auto resp_res = db.Scan(req);
    REQUIRE(resp_res.ok());
    auto resp = resp_res.move_value();
    REQUIRE(resp.status == ScanStatus::Ok);

    REQUIRE(db.Stop().ok());

    std::unordered_set<Id> ids;
    for (const auto &item : resp.items)
        ids.insert(item.id);
    REQUIRE(ids.count(1) == 1);
    REQUIRE(ids.count(3) == 1);
    REQUIRE(ids.count(2) == 0);
}
