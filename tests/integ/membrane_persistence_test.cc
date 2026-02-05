#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <memory>
#include <vector>

namespace {

using namespace pomai;

POMAI_TEST(MembranePersistence_CreateRestart) {
    auto path = pomai::test::TempDir("membrane_persist_create");
    
    // Phase 1: Create membrane and insert data
    {
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        opt.fsync = FsyncPolicy::kAlways; // Ensure durability
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        // Create custom membrane
        MembraneSpec spec;
        spec.name = "custom";
        spec.dim = 4;
        spec.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(spec));
        POMAI_EXPECT_OK(db->OpenMembrane("custom"));
        
        // Insert data
        std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f};
        POMAI_EXPECT_OK(db->Put("custom", 100, v1));
        POMAI_EXPECT_OK(db->Freeze("custom"));
        
        POMAI_EXPECT_OK(db->Close());
    }
    
    // Phase 2: Reopen and verify membrane exists
    {
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        // List membranes
        std::vector<std::string> membranes;
        POMAI_EXPECT_OK(db->ListMembranes(&membranes));
        
        // Should have default + custom
        POMAI_EXPECT_TRUE(membranes.size() >= 2);
        bool found_custom = false;
        for (const auto& name : membranes) {
            if (name == "custom") found_custom = true;
        }
        POMAI_EXPECT_TRUE(found_custom);
        
        // Verify data persisted
        std::vector<float> out;
        POMAI_EXPECT_OK(db->Get("custom", 100, &out));
        POMAI_EXPECT_EQ(out.size(), 4u);
        POMAI_EXPECT_EQ(out[0], 1.0f);
        
        POMAI_EXPECT_OK(db->Close());
    }
}

POMAI_TEST(MembranePersistence_DropRestartAbsent) {
    auto path = pomai::test::TempDir("membrane_persist_drop");
    
    // Phase 1: Create and drop membrane
    {
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        opt.fsync = FsyncPolicy::kAlways;
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        MembraneSpec spec;
        spec.name = "temp";
        spec.dim = 4;
        spec.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(spec));
        POMAI_EXPECT_OK(db->OpenMembrane("temp"));
        
        // Drop it
        POMAI_EXPECT_OK(db->CloseMembrane("temp"));
        POMAI_EXPECT_OK(db->DropMembrane("temp"));
        
        POMAI_EXPECT_OK(db->Close());
    }
    
    // Phase 2: Reopen and verify membrane is absent
    {
        DBOptions opt;
        opt.path = path;
        opt.dim = 4;
        opt.shard_count = 1;
        
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        std::vector<std::string> membranes;
        POMAI_EXPECT_OK(db->ListMembranes(&membranes));
        
        // Should NOT have temp
        for (const auto& name : membranes) {
            POMAI_EXPECT_TRUE(name != "temp");
        }
        
        POMAI_EXPECT_OK(db->Close());
    }
}

} // namespace
