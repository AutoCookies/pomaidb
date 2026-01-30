#include <catch2/catch.hpp>

#include <pomai/core/shard.h>
#include <pomai/index/whispergrain.h>

#include "tests/common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Shard upsert/search round-trip", "[core][shard]")
{
    TempDir dir;
    CompactionConfig compaction{};
    Shard shard("shard-0", 8, 128, dir.str(), compaction);
    shard.Start();

    auto batch = MakeBatch(25, 8, 0.1f);
    auto fut = shard.EnqueueUpserts(batch, true);
    REQUIRE(fut.get() > 0);

    SearchRequest req = MakeSearchRequest(batch[10].vec, 5);
    pomai::ai::Budget budget{};
    budget.ops_budget = 5000;
    budget.bucket_budget = 10;
    budget.allow_exact_refine = true;
    auto resp = shard.Search(req, budget);

    shard.Stop();

    REQUIRE_FALSE(resp.items.empty());
    REQUIRE(resp.items.front().id == batch[10].id);
}
