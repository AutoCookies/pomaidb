// Integration tests for embedded Database backpressure: memtable limits,
// auto_freeze_on_pressure, TryFreezeIfPressured, and ResourceExhausted semantics.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <vector>

#include "pomai/database.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace {

std::vector<float> MakeVec(std::uint32_t dim, float base) {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
        v[i] = base + static_cast<float>(i) * 0.001f;
    return v;
}

POMAI_TEST(Embedded_Backpressure_AutoFreezeReducesPressure) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("backpressure_auto");
    opt.dim = 32;
    opt.metric = pomai::MetricType::kL2;
    opt.fsync = pomai::FsyncPolicy::kNever;
    opt.max_memtable_mb = 2;
    opt.pressure_threshold_percent = 80;
    opt.auto_freeze_on_pressure = true;

    pomai::Database db;
    pomai::Status st = db.Open(opt);
    POMAI_EXPECT_OK(st);

    for (pomai::VectorId id = 0; id < 500; ++id) {
        auto v = MakeVec(opt.dim, static_cast<float>(id) * 0.01f);
        st = db.AddVector(id, v);
        POMAI_EXPECT_TRUE(st.ok() || st.code() == pomai::ErrorCode::kResourceExhausted);
        if (!st.ok()) break;
    }
    pomai::Status close_st = db.Close();
    POMAI_EXPECT_OK(close_st);
}

POMAI_TEST(Embedded_Backpressure_TryFreezeIfPressured_UnderPressure) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("backpressure_try");
    opt.dim = 64;
    opt.fsync = pomai::FsyncPolicy::kNever;
    opt.max_memtable_mb = 1;
    opt.pressure_threshold_percent = 50;
    opt.auto_freeze_on_pressure = true;

    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));

    for (pomai::VectorId id = 0; id < 200; ++id) {
        auto v = MakeVec(opt.dim, static_cast<float>(id));
        pomai::Status st = db.AddVector(id, v);
        POMAI_EXPECT_TRUE(st.ok() || st.code() == pomai::ErrorCode::kResourceExhausted);
    }
    pomai::Status try_st = db.TryFreezeIfPressured();
    POMAI_EXPECT_OK(try_st);
    POMAI_EXPECT_OK(db.Close());
}

POMAI_TEST(Embedded_Backpressure_GetMemTableBytesUsed) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("backpressure_bytes");
    opt.dim = 16;
    opt.fsync = pomai::FsyncPolicy::kNever;

    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));
    std::size_t used0 = db.GetMemTableBytesUsed();
    auto v = MakeVec(opt.dim, 1.0f);
    POMAI_EXPECT_OK(db.AddVector(1, v));
    std::size_t used1 = db.GetMemTableBytesUsed();
    POMAI_EXPECT_TRUE(used1 >= used0);
    POMAI_EXPECT_OK(db.Close());
}

} // namespace
