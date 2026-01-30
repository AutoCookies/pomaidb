#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Checkpoint recovery restores data", "[storage][checkpoint]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 6, 2);

    {
        PomaiDB db(opts);
        REQUIRE(db.Start().ok());
        auto batch = MakeBatch(30, 6, 0.1f, 2);
        auto upsert_res = db.UpsertBatch(batch, true).get();
        REQUIRE(upsert_res.ok());
        auto checkpoint = db.RequestCheckpoint();
        auto checkpoint_res = checkpoint.get();
        REQUIRE(checkpoint_res.ok());
        REQUIRE(checkpoint_res.value());
        WARN("Simulating crash by skipping explicit Stop(); destructor will still clean up threads.");
    }

    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    SearchRequest req = MakeSearchRequest(MakeVector(6, 0.1f), 5);
    auto resp_res = db.Search(req);
    REQUIRE(resp_res.ok());
    auto resp = resp_res.move_value();
    auto count = db.TotalApproxCountUnsafe();

    REQUIRE(db.Stop().ok());

    REQUIRE_FALSE(resp.items.empty());
    REQUIRE(count >= 30);
}
