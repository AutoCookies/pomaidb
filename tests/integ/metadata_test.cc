#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include <vector>
#include "pomai/pomai.h"
#include <filesystem>

namespace
{
    using namespace pomai;

    POMAI_TEST(MetadataTest_BasicPutGet)
    {
        std::string path = test::TempDir("meta_basic");
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        opt.fsync = FsyncPolicy::kNever;

        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        std::vector<float> vec = {0.1f, 0.2f, 0.3f, 0.4f};
        VectorId id = 100;
        Metadata meta("tenant_A");

        POMAI_EXPECT_OK(db->Put(id, vec, meta));

        // 1. Get with metadata
        std::vector<float> out;
        Metadata out_meta;
        POMAI_EXPECT_OK(db->Get(id, &out, &out_meta));
        POMAI_EXPECT_EQ(out.size(), 4UL); // Fixed signed/unsigned mismatch
        if (out.size() > 0) POMAI_EXPECT_EQ(out[0], 0.1f);
        POMAI_EXPECT_EQ(out_meta.tenant, "tenant_A");

        // 2. Get without metadata
        out.clear();
        POMAI_EXPECT_OK(db->Get(id, &out));
        POMAI_EXPECT_EQ(out.size(), 4UL);
        
        POMAI_EXPECT_OK(db->Close());
        std::filesystem::remove_all(path);
    }

    POMAI_TEST(MetadataTest_Overwrite)
    {
        std::string path = test::TempDir("meta_overwrite");
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;

        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        std::vector<float> vec = {0.1f, 0.1f, 0.1f, 0.1f};
        VectorId id = 101;
        
        // Put "A"
        POMAI_EXPECT_OK(db->Put(id, vec, Metadata("A")));
        {
            std::vector<float> out;
            Metadata m;
            POMAI_EXPECT_OK(db->Get(id, &out, &m));
            POMAI_EXPECT_EQ(m.tenant, "A");
        }

        // Overwrite "B"
        POMAI_EXPECT_OK(db->Put(id, vec, Metadata("B")));
        {
            std::vector<float> out;
            Metadata m;
            POMAI_EXPECT_OK(db->Get(id, &out, &m));
            POMAI_EXPECT_EQ(m.tenant, "B");
        }

        // Overwrite without metadata (empty) -> Should result in empty metadata
        POMAI_EXPECT_OK(db->Put(id, vec));
        {
            std::vector<float> out;
            Metadata m;
            POMAI_EXPECT_OK(db->Get(id, &out, &m));
            POMAI_EXPECT_EQ(m.tenant.empty(), true);
        }
        POMAI_EXPECT_OK(db->Close());
        std::filesystem::remove_all(path);
    }

    POMAI_TEST(MetadataTest_Persistence)
    {
        std::string path = test::TempDir("meta_persist");
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;

        {
            std::unique_ptr<DB> db;
            POMAI_EXPECT_OK(DB::Open(opt, &db));
            std::vector<float> vec = {1.0f, 1.0f, 1.0f, 1.0f};
            
            POMAI_EXPECT_OK(db->Put(1, vec, Metadata("T1")));
            POMAI_EXPECT_OK(db->Put(2, vec, Metadata("T2")));
            POMAI_EXPECT_OK(db->Put(3, vec)); // Empty meta
            POMAI_EXPECT_OK(db->Close());
        }

        {
            std::unique_ptr<DB> db;
            POMAI_EXPECT_OK(DB::Open(opt, &db));

            {
                std::vector<float> out;
                Metadata m;
                POMAI_EXPECT_OK(db->Get(1, &out, &m));
                POMAI_EXPECT_EQ(m.tenant, "T1");
            }
            {
                std::vector<float> out;
                Metadata m;
                POMAI_EXPECT_OK(db->Get(2, &out, &m));
                POMAI_EXPECT_EQ(m.tenant, "T2");
            }
            {
                // Verify empty stays empty
                std::vector<float> out;
                Metadata m;
                POMAI_EXPECT_OK(db->Get(3, &out, &m));
                POMAI_EXPECT_EQ(m.tenant.empty(), true);
            }
            POMAI_EXPECT_OK(db->Close());
        }
        std::filesystem::remove_all(path);
    }

    POMAI_TEST(MetadataTest_DeleteClears)
    {
        std::string path = test::TempDir("meta_del");
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;

        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));

        std::vector<float> vec = {0.5f, 0.5f, 0.5f, 0.5f};
        VectorId id = 500;
        
        POMAI_EXPECT_OK(db->Put(id, vec, Metadata("DeletedTenant")));
        POMAI_EXPECT_OK(db->Delete(id));

        std::vector<float> out;
        Metadata m;
        // Should be Not Found (check status code)
        Status st = db->Get(id, &out, &m);
        if (st.ok()) {
            std::cerr << "Expected Error but got OK for deleted item\n";
            std::abort();
        }
        
        // Reinsert clean
        POMAI_EXPECT_OK(db->Put(id, vec));
        POMAI_EXPECT_OK(db->Get(id, &out, &m));
        POMAI_EXPECT_EQ(m.tenant.empty(), true);
        
        POMAI_EXPECT_OK(db->Close());
        std::filesystem::remove_all(path);
    }
}

