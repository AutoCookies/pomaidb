// TSAN test: stress embedded Database with backpressure (low memtable limit)
// to ensure no data races when auto-freeze and TryFreezeIfPressured run.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <vector>

#include "pomai/database.h"
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

POMAI_TEST(Tsan_EmbeddedBackpressure_WriteAndFreeze) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("tsan_backpressure");
    opt.dim = 32;
    opt.fsync = pomai::FsyncPolicy::kNever;
    opt.max_memtable_mb = 1;
    opt.pressure_threshold_percent = 70;
    opt.auto_freeze_on_pressure = true;
    opt.index_params = pomai::IndexParams::ForEdge();

    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));

    for (int round = 0; round < 3; ++round) {
        for (pomai::VectorId id = 0; id < 80; ++id) {
            pomai::VectorId gid = static_cast<pomai::VectorId>(round * 1000 + id);
            auto v = MakeVec(opt.dim, static_cast<float>(gid));
            pomai::Status st = db.AddVector(gid, v);
            POMAI_EXPECT_TRUE(st.ok() || st.code() == pomai::ErrorCode::kResourceExhausted);
        }
        POMAI_EXPECT_OK(db.TryFreezeIfPressured());
        POMAI_EXPECT_OK(db.Freeze());
    }

    pomai::SearchResult out;
    std::vector<float> q = MakeVec(opt.dim, 0.0f);
    POMAI_EXPECT_OK(db.Search(q, 10, &out));
    POMAI_EXPECT_TRUE(out.hits.size() <= 10);
    POMAI_EXPECT_OK(db.Close());
}

} // namespace
