#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <fstream>

#include "core/routing/routing_persist.h"

namespace {

POMAI_TEST(RoutingPersist_SaveLoadRoundtrip) {
    const std::string dir = pomai::test::TempDir("routing_roundtrip");
    pomai::core::routing::RoutingTable t;
    t.epoch = 7;
    t.k = 2;
    t.dim = 3;
    t.centroids = {1,2,3, 4,5,6};
    t.owner_shard = {0,1};
    t.counts = {10,20};

    POMAI_EXPECT_OK(pomai::core::routing::SaveRoutingTableAtomic(dir, t, true));
    auto loaded = pomai::core::routing::LoadRoutingTable(dir);
    POMAI_EXPECT_TRUE(loaded.has_value());
    POMAI_EXPECT_EQ(loaded->epoch, static_cast<std::uint64_t>(7));
    POMAI_EXPECT_EQ(loaded->centroids.size(), static_cast<std::size_t>(6));
    POMAI_EXPECT_EQ(loaded->owner_shard[1], static_cast<std::uint32_t>(1));
}

POMAI_TEST(RoutingPersist_TruncatedFileTreatedMissing) {
    const std::string dir = pomai::test::TempDir("routing_trunc");
    pomai::core::routing::RoutingTable t;
    t.epoch = 1;
    t.k = 1;
    t.dim = 2;
    t.centroids = {1,2};
    t.owner_shard = {0};
    t.counts = {1};
    POMAI_EXPECT_OK(pomai::core::routing::SaveRoutingTableAtomic(dir, t, false));

    const auto path = pomai::core::routing::RoutingPath(dir);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write("bad", 3);
    out.close();

    auto loaded = pomai::core::routing::LoadRoutingTable(dir);
    POMAI_EXPECT_TRUE(!loaded.has_value());
}

} // namespace
