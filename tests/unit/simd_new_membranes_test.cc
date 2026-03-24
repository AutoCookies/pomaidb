#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(SimdNewMembranes_BasicOps) {
    auto path = pomai::test::TempDir("simd_membranes_basic");
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    MembraneSpec spatial; spatial.name = "sp"; spatial.kind = MembraneKind::kSpatial; spatial.shard_count = 1;
    MembraneSpec mesh; mesh.name = "mesh"; mesh.kind = MembraneKind::kMesh; mesh.shard_count = 1;
    MembraneSpec sparse; sparse.name = "sps"; sparse.kind = MembraneKind::kSparse; sparse.shard_count = 1;
    MembraneSpec bitset; bitset.name = "bs"; bitset.kind = MembraneKind::kBitset; bitset.shard_count = 1;
    POMAI_EXPECT_OK(db->CreateMembrane(spatial)); POMAI_EXPECT_OK(db->OpenMembrane("sp"));
    POMAI_EXPECT_OK(db->CreateMembrane(mesh)); POMAI_EXPECT_OK(db->OpenMembrane("mesh"));
    POMAI_EXPECT_OK(db->CreateMembrane(sparse)); POMAI_EXPECT_OK(db->OpenMembrane("sps"));
    POMAI_EXPECT_OK(db->CreateMembrane(bitset)); POMAI_EXPECT_OK(db->OpenMembrane("bs"));

    POMAI_EXPECT_OK(db->SpatialPut("sp", 1, 10.0, 10.0));
    POMAI_EXPECT_OK(db->SpatialPut("sp", 2, 10.001, 10.001));
    std::vector<SpatialPoint> near;
    POMAI_EXPECT_OK(db->SpatialRadiusSearch("sp", 10.0, 10.0, 300.0, &near));
    POMAI_EXPECT_TRUE(!near.empty());

    std::vector<float> m1 = {0,0,0, 1,0,0, 0,1,0};
    std::vector<float> m2 = {0,0,0, 1,0,0, 0,1,0};
    POMAI_EXPECT_OK(db->MeshPut("mesh", 1, m1));
    POMAI_EXPECT_OK(db->MeshPut("mesh", 2, m2));
    double rmsd = 0.0;
    POMAI_EXPECT_OK(db->MeshRmsd("mesh", 1, 2, &rmsd));
    POMAI_EXPECT_TRUE(rmsd <= 1e-6);
    MeshQueryOptions hi_detail;
    hi_detail.detail = MeshDetailPreference::kHighDetail;
    POMAI_EXPECT_OK(db->MeshRmsd("mesh", 1, 2, hi_detail, &rmsd));
    POMAI_EXPECT_TRUE(rmsd <= 1e-6);

    SparseEntry se1{{1,3,8}, {1.0f, 2.0f, 1.0f}};
    SparseEntry se2{{1,2,8}, {1.0f, 1.0f, 3.0f}};
    POMAI_EXPECT_OK(db->SparsePut("sps", 1, se1));
    POMAI_EXPECT_OK(db->SparsePut("sps", 2, se2));
    uint32_t inter = 0;
    POMAI_EXPECT_OK(db->SparseIntersect("sps", 1, 2, &inter));
    POMAI_EXPECT_EQ(inter, 2u);

    std::vector<uint8_t> b1 = {0xF0, 0x0F};
    std::vector<uint8_t> b2 = {0xFF, 0x00};
    POMAI_EXPECT_OK(db->BitsetPut("bs", 1, b1));
    POMAI_EXPECT_OK(db->BitsetPut("bs", 2, b2));
    double h = 0.0;
    POMAI_EXPECT_OK(db->BitsetHamming("bs", 1, 2, &h));
    POMAI_EXPECT_TRUE(h > 0.0);

    POMAI_EXPECT_OK(db->Close());
}

} // namespace

