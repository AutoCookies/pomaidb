#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <filesystem>
#include <vector>
#include <cmath>

namespace pomai
{
    namespace fs = std::filesystem;

    POMAI_TEST(ReadYourWrites_ActiveMemTable)
    {
        std::string test_dir = pomai::test::TempDir("pomai_consistency_1");
        DBOptions opt;
        opt.path = test_dir;
        opt.dim = 2;
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        // 1. Put (Active)
        std::vector<float> vec = {1.0f, 2.0f};
        POMAI_EXPECT_OK(db->Put(100, vec));

        // 2. Get (Must see it instantly)
        std::vector<float> out;
        POMAI_EXPECT_OK(db->Get(100, &out));
        POMAI_EXPECT_EQ(out.size(), vec.size());
        POMAI_EXPECT_EQ(out[0], vec[0]);
        POMAI_EXPECT_EQ(out[1], vec[1]);

        // 3. Exists (Must be true)
        bool exists = false;
        POMAI_EXPECT_OK(db->Exists(100, &exists));
        POMAI_EXPECT_TRUE(exists);

        // 4. Search (Must find it)
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 1, &res));
        POMAI_EXPECT_EQ(res.hits.size(), static_cast<size_t>(1));
        POMAI_EXPECT_EQ(res.hits[0].id, static_cast<VectorId>(100));
        // Score: 1*1 + 2*2 = 5
        POMAI_EXPECT_TRUE(std::abs(res.hits[0].score - 5.0f) < 0.001f);
        
        fs::remove_all(test_dir);
    }

    POMAI_TEST(TombstoneDominatesActive)
    {
        std::string test_dir = pomai::test::TempDir("pomai_consistency_2");
        DBOptions opt;
        opt.path = test_dir;
        opt.dim = 2;
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        std::vector<float> vec = {1.0f, 2.0f};
        POMAI_EXPECT_OK(db->Put(100, vec));

        // Verify it exists
        bool exists = false;
        db->Exists(100, &exists);
        POMAI_EXPECT_TRUE(exists);

        // Delete (Active Tombstone)
        POMAI_EXPECT_OK(db->Delete(100));

        // Verify Get returns NotFound
        std::vector<float> out;
        Status st = db->Get(100, &out);
        POMAI_EXPECT_TRUE(st.code() == ErrorCode::kNotFound);

        // Verify Exists returns false
        db->Exists(100, &exists);
        POMAI_EXPECT_TRUE(!exists);

        // Verify Search ignores it
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 1, &res));
        POMAI_EXPECT_TRUE(res.hits.empty());
        fs::remove_all(test_dir);
    }

    POMAI_TEST(NewestWins_Update)
    {
        std::string test_dir = pomai::test::TempDir("pomai_consistency_3");
        DBOptions opt;
        opt.path = test_dir;
        opt.dim = 2;
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        std::vector<float> v1 = {1.0f, 2.0f};
        std::vector<float> v2 = {3.0f, 4.0f};

        // Put v1
        POMAI_EXPECT_OK(db->Put(100, v1));
        
        // Put v2 (Update)
        POMAI_EXPECT_OK(db->Put(100, v2));

        // Get should return v2
        std::vector<float> out;
        POMAI_EXPECT_OK(db->Get(100, &out));
        POMAI_EXPECT_EQ(out.size(), v2.size());
        POMAI_EXPECT_EQ(out[0], v2[0]);
        POMAI_EXPECT_EQ(out[1], v2[1]);

        // Search for v1 should find 100 but using v2's content
        // v1 query = {1, 2}
        // v2 doc = {3, 4} -> 3+8 = 11.
        
        SearchResult res;
        // Search with v1 query
        POMAI_EXPECT_OK(db->Search(v1, 1, &res));
        POMAI_EXPECT_EQ(res.hits.size(), static_cast<size_t>(1));
        POMAI_EXPECT_EQ(res.hits[0].id, static_cast<VectorId>(100));
        POMAI_EXPECT_TRUE(std::abs(res.hits[0].score - 11.0f) < 0.001f);
        fs::remove_all(test_dir);
    }
}
