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
        db.Start();
        auto batch = MakeBatch(30, 6, 0.1f, 2);
        db.UpsertBatch(batch, true).get();
        auto checkpoint = db.RequestCheckpoint();
        REQUIRE(checkpoint.get());
        WARN("Simulating crash by skipping explicit Stop(); destructor will still clean up threads.");
    }

    PomaiDB db(opts);
    db.Start();

    SearchRequest req = MakeSearchRequest(MakeVector(6, 0.1f), 5);
    auto resp = db.Search(req);
    auto count = db.TotalApproxCountUnsafe();

    db.Stop();

    REQUIRE_FALSE(resp.items.empty());
    REQUIRE(count >= 30);
}
