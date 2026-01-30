#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>
#include <pomai/storage/verify.h>
#include <pomai/storage/snapshot.h>

#include "tests/common/test_utils.h"

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("pomai_verify equivalent succeeds", "[tools][verify]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 6, 1);

    PomaiDB db(opts);
    db.Start();
    auto batch = MakeBatch(10, 6, 0.1f, 2);
    db.UpsertBatch(batch, true).get();
    REQUIRE(db.RequestCheckpoint().get());
    db.Stop();

    pomai::storage::Manifest manifest;
    std::string err;
    REQUIRE(pomai::storage::VerifyAll(dir.str(), manifest, &err));

    std::string snapshot_path = manifest.checkpoint_path;
    if (!snapshot_path.empty() && snapshot_path.front() != '/')
        snapshot_path = dir.str() + "/" + snapshot_path;
    pomai::storage::SnapshotData snapshot;
    REQUIRE(pomai::storage::ReadSnapshotFile(snapshot_path, snapshot, &err));
    REQUIRE(snapshot.schema.dim == opts.dim);
}
