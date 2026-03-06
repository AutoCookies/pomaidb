// Integration tests for embedded Database error paths: invalid arguments,
// null outputs, and not-opened semantics.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <memory>
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

POMAI_TEST(Embedded_Error_OpenDimZero) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("err_dim0");
    opt.dim = 0;
    pomai::Database db;
    pomai::Status st = db.Open(opt);
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
}

POMAI_TEST(Embedded_Error_OpenPathEmpty) {
    pomai::EmbeddedOptions opt;
    opt.path = "";
    opt.dim = 8;
    pomai::Database db;
    pomai::Status st = db.Open(opt);
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
}

POMAI_TEST(Embedded_Error_AddVectorNotOpened) {
    pomai::Database db;
    std::vector<float> v(8, 1.0f);
    pomai::Status st = db.AddVector(1, v);
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
}

POMAI_TEST(Embedded_Error_SearchNotOpened) {
    pomai::Database db;
    std::vector<float> q(8, 0.0f);
    pomai::SearchResult out;
    pomai::Status st = db.Search(q, 10, &out);
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
}

POMAI_TEST(Embedded_Error_SearchNullOut) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("err_search_null");
    opt.dim = 8;
    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));
    std::vector<float> q(8, 0.0f);
    pomai::Status st = db.Search(q, 10, nullptr);
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
    POMAI_EXPECT_OK(db.Close());
}

POMAI_TEST(Embedded_Error_GetSnapshotNullOut) {
    pomai::EmbeddedOptions opt;
    opt.path = pomai::test::TempDir("err_snap_null");
    opt.dim = 8;
    pomai::Database db;
    POMAI_EXPECT_OK(db.Open(opt));
    pomai::Status st = db.GetSnapshot(nullptr);
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
    POMAI_EXPECT_OK(db.Close());
}

POMAI_TEST(Embedded_Error_TryFreezeNotOpened) {
    pomai::Database db;
    pomai::Status st = db.TryFreezeIfPressured();
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
}

POMAI_TEST(Embedded_Error_FlushNotOpened) {
    pomai::Database db;
    pomai::Status st = db.Flush();
    POMAI_EXPECT_TRUE(!st.ok());
    POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
}

} // namespace
