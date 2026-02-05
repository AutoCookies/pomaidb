#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <memory>
#include <vector>
#include <thread>
#include <chrono>

namespace {

using namespace pomai;

static std::vector<float> MakeVec(std::uint32_t dim, float val) {
    return std::vector<float>(dim, val);
}

POMAI_TEST(Consistency_ReadYourWrites_Immediate) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("consistency_ryw_imm");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));
    // Default membrane created implicitly? No, API usually requires creation?
    // db_basic_test explicit creates default. 
    // Manager::Open creates default if not exists?
    // Let's create explicitly to be safe.
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec); 
    db->OpenMembrane("default");

    std::vector<float> v1 = MakeVec(4, 1.0f);
    
    // 1. Put
    POMAI_EXPECT_OK(db->Put("default", 1, v1));

    // 2. Get IMMEDIATELY (No Freeze)
    std::vector<float> out;
    POMAI_EXPECT_OK(db->Get("default", 1, &out));
    POMAI_EXPECT_EQ(out.size(), 4u);
    POMAI_EXPECT_EQ(out[0], 1.0f);

    // 3. Exists IMMEDIATELY
    bool exists = false;
    POMAI_EXPECT_OK(db->Exists("default", 1, &exists));
    POMAI_EXPECT_TRUE(exists);

    // 4. Search IMMEDIATELY
    SearchResult res;
    POMAI_EXPECT_OK(db->Search("default", v1, 10, &res));
    bool found = false;
    for(const auto& h : res.hits) {
        if(h.id == 1) found = true;
    }
    POMAI_EXPECT_TRUE(found);

    // 5. Delete
    POMAI_EXPECT_OK(db->Delete("default", 1));

    // 6. Get IMMEDIATELY (Should be NotFound)
    // Note: Active MemTombstone should hide it.
    Status st = db->Get("default", 1, &out);
    POMAI_EXPECT_TRUE(!st.ok()); 
    // Is it NotFound?
    POMAI_EXPECT_TRUE(st.code() == ErrorCode::kNotFound);

    // 7. Exists IMMEDIATELY
    POMAI_EXPECT_OK(db->Exists("default", 1, &exists));
    POMAI_EXPECT_TRUE(!exists);

    // 8. Search IMMEDIATELY (Should not find)
    res.Clear();
    POMAI_EXPECT_OK(db->Search("default", v1, 10, &res));
    found = false;
    for(const auto& h : res.hits) {
        if(h.id == 1) found = true;
    }
    POMAI_EXPECT_TRUE(!found);
    
    db->Close();
}

POMAI_TEST(Consistency_Update_LatestWins) {
    DBOptions opt;
    // unique path
    opt.path = pomai::test::TempDir("consistency_update_v2");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec; spec.name = "default"; spec.dim=4; spec.shard_count=1;
    db->CreateMembrane(spec); db->OpenMembrane("default");

    auto v1 = MakeVec(4, 1.0f);
    auto v2 = MakeVec(4, 2.0f);

    // Put v1
    db->Put("default", 1, v1);
    
    // Freeze to flush to frozen/segment (simulation of older version)
    db->Freeze("default");

    // Put v2 (Active)
    db->Put("default", 1, v2);

    // Get should return v2
    std::vector<float> out;
    db->Get("default", 1, &out);
    POMAI_EXPECT_EQ(out[0], 2.0f);

    // Search should match v2 better? 
    // And should only return one hit for ID 1.
    SearchResult res;
    // Search with v2
    db->Search("default", v2, 10, &res);
    int count = 0;
    for(const auto& h : res.hits) {
        if (h.id == 1) {
            count++;
            // If V2, score should be ~DOT(2,2) = 4*2*2? No, 2.0*2.0*4 = 16.
            // If V1, 1.0*2.0*4 = 8.
            // POMAI_EXPECT_TRUE(h.score > 10.0f);
        }
    }
    POMAI_EXPECT_EQ(count, 1); // Should unify duplicates
}

POMAI_TEST(Consistency_Tombstone_Hides_Frozen) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("consistency_tomb");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec; spec.name = "default"; spec.dim=4; spec.shard_count=1;
    db->CreateMembrane(spec); db->OpenMembrane("default");

    auto v1 = MakeVec(4, 1.0f);
    db->Put("default", 1, v1);
    db->Freeze("default"); // 1 is in frozen/segment

    db->Delete("default", 1); // Delete in Active
    
    // Get should not find it
    std::vector<float> out;
    Status st = db->Get("default", 1, &out);
    POMAI_EXPECT_TRUE(st.code() == ErrorCode::kNotFound);

    // Search should not find it
    SearchResult res;
    db->Search("default", v1, 10, &res);
    bool found = false;
    for(const auto& h : res.hits) if(h.id == 1) found = true;
    POMAI_EXPECT_TRUE(!found);
}

} // namespace
