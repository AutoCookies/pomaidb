#include <catch2/catch.hpp>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

#include "tests/common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("WAL replay applies expected vectors", "[storage][wal]")
{
    TempDir dir;
    const std::size_t dim = 4;

    {
        Wal wal("shard-0", dir.str(), dim);
        wal.Start();
        auto batch = MakeBatch(5, dim, 0.3f);
        wal.AppendUpserts(batch, true);
        wal.Stop();
    }

    Seed seed(dim);
    Wal wal_reader("shard-0", dir.str(), dim);
    auto stats = wal_reader.ReplayToSeed(seed);

    REQUIRE(stats.records_applied == 1);
    REQUIRE(stats.vectors_applied == 5);
    REQUIRE(seed.Count() == 5);
}
