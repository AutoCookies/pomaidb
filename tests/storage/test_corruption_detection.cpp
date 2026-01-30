#include <catch2/catch.hpp>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

#include "common/test_utils.h"

#include <filesystem>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

//
TEST_CASE("WAL replay truncates corrupt tail", "[storage][corruption]")
{
    TempDir dir;
    const std::size_t dim = 4;

    {
        Wal wal("shard-0", dir.str(), dim);
        wal.Start();
        // Ghi 2 record riêng biệt
        wal.AppendUpserts(MakeBatch(2, dim, 0.1f), true); // Record 1: OK
        wal.AppendUpserts(MakeBatch(3, dim, 0.2f), true); // Record 2: Sẽ bị hỏng
        wal.Stop();
    }

    auto wal_path = std::filesystem::path(dir.str()) / "shard-0.wal";
    auto size = std::filesystem::file_size(wal_path);
    // Cắt bỏ 8 byte cuối của record thứ 2
    std::filesystem::resize_file(wal_path, size - 8);

    Seed seed(dim);
    Wal wal_reader("shard-0", dir.str(), dim);
    auto stats = wal_reader.ReplayToSeed(seed);

    // Hệ thống phải nạp được Record 1 (2 vectors) và bỏ qua Record 2 bị hỏng
    REQUIRE(stats.truncated_bytes > 0);
    REQUIRE(stats.vectors_applied == 2);
    REQUIRE(seed.Count() == 2);
}
