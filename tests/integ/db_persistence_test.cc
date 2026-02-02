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
            v[i] = base + static_cast<float>(i) * 0.01f;
        return v;
    }

    POMAI_TEST(DB_Persistence_Reopen_ReplaysWal)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-db_persistence_test");
        opt.dim = 16;
        opt.shard_count = 4;
        opt.fsync = pomai::FsyncPolicy::kAlways;

        // 1) Open -> Put -> Flush -> Close
        {
            std::unique_ptr<pomai::DB> db;
            POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

            auto v = MakeVec(opt.dim, 3.14f);
            POMAI_EXPECT_OK(db->Put(777, v));
            POMAI_EXPECT_OK(db->Flush());
            POMAI_EXPECT_OK(db->Close());
        }

        // 2) Reopen -> Search phải thấy id=777
        {
            std::unique_ptr<pomai::DB> db;
            POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

            auto q = MakeVec(opt.dim, 3.14f);
            pomai::SearchResult r;
            POMAI_EXPECT_OK(db->Search(q, /*topk*/ 10, &r));

            bool found = false;
            for (const auto &h : r.hits)
            {
                if (h.id == static_cast<pomai::VectorId>(777))
                {
                    found = true;
                    break;
                }
            }
            POMAI_EXPECT_TRUE(found);

            POMAI_EXPECT_OK(db->Close());
        }
    }

} // namespace
