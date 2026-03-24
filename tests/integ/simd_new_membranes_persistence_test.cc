#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(SimdNewMembranes_Persistence_Restart) {
    auto path = pomai::test::TempDir("simd_membranes_persist");
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        MembraneSpec sp; sp.name = "sp"; sp.kind = MembraneKind::kSpatial; sp.shard_count = 1;
        MembraneSpec mesh; mesh.name = "mesh"; mesh.kind = MembraneKind::kMesh; mesh.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(sp)); POMAI_EXPECT_OK(db->OpenMembrane("sp"));
        POMAI_EXPECT_OK(db->CreateMembrane(mesh)); POMAI_EXPECT_OK(db->OpenMembrane("mesh"));
        POMAI_EXPECT_OK(db->SpatialPut("sp", 11, 20.0, 21.0));
        std::vector<float> verts = {0,0,0, 1,0,0, 0,1,0};
        POMAI_EXPECT_OK(db->MeshPut("mesh", 1, verts));
        POMAI_EXPECT_OK(db->Close());
    }
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        std::vector<SpatialPoint> out;
        POMAI_EXPECT_OK(db->SpatialRadiusSearch("sp", 20.0, 21.0, 100.0, &out));
        POMAI_EXPECT_TRUE(!out.empty());
        double vol = 0.0;
        POMAI_EXPECT_OK(db->MeshVolume("mesh", 1, &vol));
        POMAI_EXPECT_TRUE(vol >= 0.0);
        MeshQueryOptions hi;
        hi.detail = MeshDetailPreference::kHighDetail;
        POMAI_EXPECT_OK(db->MeshVolume("mesh", 1, hi, &vol));
        POMAI_EXPECT_TRUE(vol >= 0.0);
        POMAI_EXPECT_OK(db->Close());
    }
}

} // namespace

