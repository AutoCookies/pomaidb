#include "tests/common/test_main.h"
#include <cstdint>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"
#include "tests/common/db_test_util.h"
#include "tests/common/test_tmpdir.h"

POMAI_TEST(DB_OpenPutSearchDelete)
{
    pomai::DBOptions opt;
    opt.path = pomai::test::TempDir("pomai-integ-basic");
    opt.dim = 4;
    opt.shard_count = 2;

    auto db = pomai::test::OpenDB(opt);
    POMAI_EXPECT_TRUE(db != nullptr);

    std::vector<float> v1 = {1, 0, 0, 0};
    std::vector<float> v2 = {0, 1, 0, 0};

    POMAI_EXPECT_OK(db->Put(100, v1));
    POMAI_EXPECT_OK(db->Put(200, v2));
    POMAI_EXPECT_OK(db->Flush());

    pomai::SearchResult out;
    POMAI_EXPECT_OK(db->Search(v1, /*topk*/ 2, &out));
    POMAI_EXPECT_TRUE(out.hits.size() >= 1);
    POMAI_EXPECT_EQ(out.hits[0].id, 100u);

    POMAI_EXPECT_OK(db->Delete(100));
    POMAI_EXPECT_OK(db->Flush());

    pomai::SearchResult out2;
    POMAI_EXPECT_OK(db->Search(v1, /*topk*/ 10, &out2));
    // After delete, best hit should not be 100 anymore (or hits empty)
    if (!out2.hits.empty())
    {
        POMAI_EXPECT_TRUE(out2.hits[0].id != 100u);
    }

    POMAI_EXPECT_OK(db->Close());
}
