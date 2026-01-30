#include <catch2/catch.hpp>

#include <pomai/core/membrane.h>

#include "tests/common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Membrane router configures centroids", "[core][membrane]")
{
    TempDir dir;
    std::vector<std::unique_ptr<Shard>> shards;
    CompactionConfig compaction{};
    shards.push_back(std::make_unique<Shard>("shard-0", 4, 64, dir.str(), compaction));
    shards.push_back(std::make_unique<Shard>("shard-1", 4, 64, dir.str(), compaction));

    MembraneRouter::FilterConfig filter_cfg = MembraneRouter::FilterConfig::Default();
    MembraneRouter membrane(std::move(shards),
                            WhisperConfig{},
                            4,
                            Metric::L2,
                            1,
                            500,
                            128,
                            1000,
                            filter_cfg,
                            []() {});

    std::vector<Vector> centroids;
    centroids.push_back(MakeVector(4, 0.5f));
    centroids.push_back(MakeVector(4, 1.5f));
    membrane.ConfigureCentroids(centroids);

    REQUIRE(membrane.HasCentroids());
    auto snap = membrane.SnapshotCentroids();
    REQUIRE(snap.size() == centroids.size());
}
