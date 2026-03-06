// Integration tests for edge-like workloads: IndexParams::ForEdge(), repeated
// freeze/compact cycles, and small resource limits.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <vector>

#include "pomai/database.h"
#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace {

std::vector<float> MakeVec(std::uint32_t dim, float base) {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
        v[i] = base + static_cast<float>(i) * 0.001f;
    return v;
}

POMAI_TEST(Embedded_Edge_ForEdgeParams_SearchCorrect) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("edge_for_edge");
    opt.dim = 24;
    opt.metric = pomai::MetricType::kL2;
    opt.fsync = pomai::FsyncPolicy::kNever;
    opt.index_params = pomai::IndexParams::ForEdge();

    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));

    for (pomai::VectorId id = 0; id < 100; ++id) {
        auto v = MakeVec(opt.dim, static_cast<float>(id) * 0.1f);
        POMAI_EXPECT_OK(db.AddVector(id, v));
    }
    POMAI_EXPECT_OK(db.Freeze());

    pomai::SearchResult out;
    std::vector<float> q = MakeVec(opt.dim, 5.0f);
    POMAI_EXPECT_OK(db.Search(q, 10, &out));
    POMAI_EXPECT_TRUE(!out.hits.empty());
    POMAI_EXPECT_TRUE(out.hits.size() <= 10);

    POMAI_EXPECT_OK(db.Close());
}

POMAI_TEST(Embedded_Edge_RepeatedFreezeCycles) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("edge_freeze_cycles");
    opt.dim = 16;
    opt.fsync = pomai::FsyncPolicy::kNever;
    opt.index_params = pomai::IndexParams::ForEdge();
    opt.max_memtable_mb = 1;
    opt.auto_freeze_on_pressure = true;

    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));

    for (int cycle = 0; cycle < 5; ++cycle) {
        for (pomai::VectorId i = 0; i < 30; ++i) {
            pomai::VectorId id = static_cast<pomai::VectorId>(cycle * 1000 + i);
            auto v = MakeVec(opt.dim, static_cast<float>(id));
            POMAI_EXPECT_OK(db.AddVector(id, v));
        }
        POMAI_EXPECT_OK(db.Freeze());
    }

    pomai::SearchResult out;
    std::vector<float> q = MakeVec(opt.dim, 0.0f);
    POMAI_EXPECT_OK(db.Search(q, 20, &out));
    POMAI_EXPECT_TRUE(out.hits.size() <= 20);
    POMAI_EXPECT_OK(db.Close());
}

POMAI_TEST(Embedded_Edge_SmallDimSmallBatch) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("edge_small");
    opt.dim = 8;
    opt.fsync = pomai::FsyncPolicy::kNever;
    opt.index_params = pomai::IndexParams::ForEdge();

    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));

    std::vector<pomai::VectorId> ids;
    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < 20; ++i) {
        ids.push_back(static_cast<pomai::VectorId>(i));
        vecs.push_back(MakeVec(opt.dim, static_cast<float>(i)));
    }
    std::vector<std::span<const float>> spans;
    for (const auto& v : vecs) spans.push_back(v);
    POMAI_EXPECT_OK(db.AddVectorBatch(ids, spans));
    POMAI_EXPECT_OK(db.Freeze());

    pomai::SearchResult out;
    std::vector<float> q = MakeVec(opt.dim, 10.0f);
    POMAI_EXPECT_OK(db.Search(q, 5, &out));
    POMAI_EXPECT_TRUE(out.hits.size() <= 5);
    POMAI_EXPECT_OK(db.Close());
}

} // namespace
