#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Search returns self in top-k", "[core][search]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 6, 2);
    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    auto batch = MakeBatch(20, 6, 0.2f, 1);
    auto upsert_res = db.UpsertBatch(batch, true).get();
    REQUIRE(upsert_res.ok());

    for (const auto &row : batch)
    {
        SearchRequest req = MakeSearchRequest(row.vec, 5);
        auto resp_res = db.Search(req);
        REQUIRE(resp_res.ok());
        auto resp = resp_res.move_value();
        REQUIRE(ContainsId(resp.items, row.id));
    }

    REQUIRE(db.Stop().ok());
}
