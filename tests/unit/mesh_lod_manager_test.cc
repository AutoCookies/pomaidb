#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <memory>
#include <vector>

#include "core/mesh/mesh_engine.h"
#include "core/mesh/mesh_lod_manager.h"
#include "pomai/options.h"

namespace {

std::vector<float> BuildGridMesh(std::size_t triangles) {
    std::vector<float> xyz;
    xyz.reserve(triangles * 9u);
    for (std::size_t i = 0; i < triangles; ++i) {
        const float x = static_cast<float>(i % 32u);
        const float y = static_cast<float>(i / 32u);
        xyz.insert(xyz.end(), {x, y, 0.0f, x + 1.0f, y, 0.0f, x, y + 1.0f, 0.0f});
    }
    return xyz;
}

POMAI_TEST(MeshLodManager_BuildMonotonic) {
    pomai::DBOptions opts;
    auto mesh = BuildGridMesh(256);

    std::vector<pomai::core::MeshLodLevel> lods;
    POMAI_EXPECT_OK(pomai::core::MeshLodManager::BuildLods(mesh, opts, &lods));
    POMAI_EXPECT_TRUE(!lods.empty());
    POMAI_EXPECT_EQ(lods.front().level, 0u);
    for (std::size_t i = 1; i < lods.size(); ++i) {
        POMAI_EXPECT_TRUE(lods[i].xyz.size() < lods[i - 1].xyz.size());
    }
}

POMAI_TEST(MeshLodManager_MeshEnginePersistsLodLog) {
    const auto path = pomai::test::TempDir("mesh_lod_manager_unit");
    pomai::DBOptions opts;
    auto mesh = BuildGridMesh(128);

    {
        pomai::core::MeshEngine engine(path, 0, opts);
        POMAI_EXPECT_OK(engine.Open());
        POMAI_EXPECT_OK(engine.Put(7, mesh));
        POMAI_EXPECT_OK(engine.ProcessLodJobs(8));
        double vol = 0.0;
        POMAI_EXPECT_OK(engine.Volume(7, &vol));
        POMAI_EXPECT_TRUE(vol >= 0.0);
        POMAI_EXPECT_OK(engine.Close());
    }

    {
        pomai::core::MeshEngine engine(path, 0, opts);
        POMAI_EXPECT_OK(engine.Open());
        double auto_vol = 0.0;
        pomai::MeshQueryOptions q;
        q.detail = pomai::MeshDetailPreference::kAutoLatencyFirst;
        POMAI_EXPECT_OK(engine.Volume(7, q, &auto_vol));
        double hi_vol = 0.0;
        q.detail = pomai::MeshDetailPreference::kHighDetail;
        POMAI_EXPECT_OK(engine.Volume(7, q, &hi_vol));
        POMAI_EXPECT_TRUE(auto_vol >= 0.0);
        POMAI_EXPECT_TRUE(hi_vol >= 0.0);
        POMAI_EXPECT_OK(engine.Close());
    }
}

}  // namespace

