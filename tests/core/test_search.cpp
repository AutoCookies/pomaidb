#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "tests/common/test_utils.h"

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
    db.Start();

    auto batch = MakeBatch(20, 6, 0.2f, 1);
    db.UpsertBatch(batch, true).get();

    for (const auto &row : batch)
    {
        SearchRequest req = MakeSearchRequest(row.vec, 5);
        auto resp = db.Search(req);
        REQUIRE(ContainsId(resp.items, row.id));
    }

    db.Stop();
}
