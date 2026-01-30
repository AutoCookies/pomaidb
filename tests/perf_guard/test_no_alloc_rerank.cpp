#include <catch2/catch.hpp>

#include <pomai/core/seed.h>
#include <pomai/util/memory_manager.h>

#include "tests/common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Search rerank does not grow tracked memory", "[perf][rerank]")
{
    MemoryManager::Instance().ResetUsageForTesting();

    const std::size_t dim = 8;
    Seed seed(dim);
    auto batch = MakeBatch(20, dim, 0.2f);
    seed.ApplyUpserts(batch);

    auto snap = seed.MakeSnapshot();
    SearchRequest req = MakeSearchRequest(batch[0].vec, 10);

    auto warm = Seed::SearchSnapshot(snap, req);
    (void)warm;

    auto before = MemoryManager::Instance().TotalUsage();
    for (int i = 0; i < 50; ++i)
        Seed::SearchSnapshot(snap, req);
    auto after = MemoryManager::Instance().TotalUsage();

    REQUIRE(after == before);
}
