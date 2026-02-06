#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <memory>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"

namespace {

POMAI_TEST(RoutingEngine_RoutedIngestStillSupportsPointOps) {
    pomai::DBOptions opt;
    opt.path = pomai::test::TempDir("routing_point_ops");
    opt.dim = 4;
    opt.shard_count = 3;
    opt.routing_enabled = true;
    opt.routing_k = 2;
    opt.routing_warmup_mult = 1;

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

    pomai::MembraneSpec spec;
    spec.name = "default";
    spec.dim = opt.dim;
    spec.shard_count = opt.shard_count;
    POMAI_EXPECT_OK(db->CreateMembrane(spec));
    POMAI_EXPECT_OK(db->OpenMembrane("default"));

    std::vector<float> a = {1,1,1,1};
    std::vector<float> b = {9,9,9,9};
    POMAI_EXPECT_OK(db->Put("default", 101, a));
    POMAI_EXPECT_OK(db->Put("default", 202, b));
    POMAI_EXPECT_OK(db->Freeze("default"));

    std::vector<float> out;
    POMAI_EXPECT_OK(db->Get("default", 101, &out));
    POMAI_EXPECT_EQ(out[0], 1.0f);

    bool exists = false;
    POMAI_EXPECT_OK(db->Exists("default", 202, &exists));
    POMAI_EXPECT_TRUE(exists);

    POMAI_EXPECT_OK(db->Delete("default", 202));
    POMAI_EXPECT_OK(db->Freeze("default"));
    POMAI_EXPECT_OK(db->Exists("default", 202, &exists));
    POMAI_EXPECT_TRUE(!exists);

    POMAI_EXPECT_OK(db->Close());
}

} // namespace
