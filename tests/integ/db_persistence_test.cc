#include "tests/common/test_main.h"
#include <cstdint>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"
#include "tests/common/db_test_util.h"
#include "tests/common/test_tmpdir.h"

POMAI_TEST(DB_ReopenPersistsThroughWalReplay)
{
    pomai::DBOptions opt;
    opt.path = pomai::test::TempDir("pomai-integ-persist");
    opt.dim = 8;
    opt.shard_count = 4;

    {
        auto db = pomai::test::OpenDB(opt);
        POMAI_EXPECT_TRUE(db != nullptr);

        std::vector<float> v(opt.dim, 0.0f);
        v[3] = 1.0f;

        for (std::uint64_t id = 1; id <= 200; ++id)
        {
            POMAI_EXPECT_OK(db->Put(id, v));
        }
        POMAI_EXPECT_OK(db->Flush());
        POMAI_EXPECT_OK(db->Close());
    }

    {
        auto db = pomai::test::OpenDB(opt);
        POMAI_EXPECT_TRUE(db != nullptr);

        std::vector<float> q(opt.dim, 0.0f);
        q[3] = 1.0f;

        pomai::SearchResult out;
        POMAI_EXPECT_OK(db->Search(q, 5, &out));
        POMAI_EXPECT_TRUE(!out.hits.empty());

        POMAI_EXPECT_OK(db->Close());
    }
}
