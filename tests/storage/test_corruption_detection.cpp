#include <catch2/catch.hpp>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

#include "tests/common/test_utils.h"

#include <filesystem>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("WAL replay truncates corrupt tail", "[storage][corruption]")
{
    TempDir dir;
    const std::size_t dim = 4;

    {
        Wal wal("shard-0", dir.str(), dim);
        wal.Start();
        auto batch = MakeBatch(3, dim, 0.2f);
        wal.AppendUpserts(batch, true);
        wal.Stop();
    }

    auto wal_path = std::filesystem::path(dir.str()) / "shard-0.wal";
    REQUIRE(std::filesystem::exists(wal_path));

    auto size = std::filesystem::file_size(wal_path);
    REQUIRE(size > 16);
    std::filesystem::resize_file(wal_path, size - 8);

    Seed seed(dim);
    Wal wal_reader("shard-0", dir.str(), dim);
    auto stats = wal_reader.ReplayToSeed(seed);

    REQUIRE(stats.truncated_bytes > 0);
    REQUIRE(stats.vectors_applied == 3);
    REQUIRE(seed.Count() == 3);
}
