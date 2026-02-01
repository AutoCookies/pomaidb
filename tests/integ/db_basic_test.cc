#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"

namespace
{

    std::vector<float> MakeVec(std::uint32_t dim, float base)
    {
        std::vector<float> v(dim);
        for (std::uint32_t i = 0; i < dim; ++i)
            v[i] = base + static_cast<float>(i) * 0.001f;
        return v;
    }

    POMAI_TEST(DB_Basic_OpenPutSearchDelete)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-db_basic_test");
        opt.dim = 8;
        opt.shard_count = 4;
        opt.fsync = pomai::FsyncPolicy::kAlways;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        const auto v1 = MakeVec(opt.dim, 1.0f);
        const auto v2 = MakeVec(opt.dim, 2.0f);

        POMAI_EXPECT_OK(db->Put(100, v1));
        POMAI_EXPECT_OK(db->Put(200, v2));
        POMAI_EXPECT_OK(db->Flush());

        // Search(v1) topk=2: kỳ vọng id=100 đứng đầu
        pomai::SearchResult r;
        POMAI_EXPECT_OK(db->Search(v1, /*topk*/ 2, &r));
        POMAI_EXPECT_TRUE(r.hits.size() >= 1);
        POMAI_EXPECT_EQ(r.hits[0].id, static_cast<pomai::VectorId>(100));

        // Delete rồi search lại không thấy 100 trong hits
        POMAI_EXPECT_OK(db->Delete(100));
        POMAI_EXPECT_OK(db->Flush());

        r.Clear();
        POMAI_EXPECT_OK(db->Search(v1, /*topk*/ 2, &r));
        for (const auto &h : r.hits)
            POMAI_EXPECT_TRUE(h.id != static_cast<pomai::VectorId>(100));

        POMAI_EXPECT_OK(db->Close());
    }

} // namespace
