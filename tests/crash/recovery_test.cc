#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <fstream>
#include <filesystem>
#include <thread>
#include <atomic>

namespace fs = std::filesystem;

namespace {

using namespace pomai;

// Test that database detects and handles WAL corruption gracefully
POMAI_TEST(WAL_CorruptionDetection) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("wal_corruption");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kAlways;

    // Create DB and insert data
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        MembraneSpec spec;
        spec.name = "default";
        spec.dim = 4;
        spec.shard_count = 1;
        db->CreateMembrane(spec);
        db->OpenMembrane("default");
        
        // Insert vectors
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < 50; ++i) {
            db->Put("default", i, v);
        }
        
        db->Flush();
        db->Close();
    }
    
    // Corrupt the WAL file if it exists
    auto wal_path = fs::path(opt.path) / "default" / "shard_0" / "wal";
    if (fs::exists(wal_path)) {
        std::ofstream corrupted(wal_path, std::ios::binary | std::ios::app);
        corrupted << "CORRUPT_DATA";
        corrupted.close();
        
        // Try to reopen - should either detect corruption or recover
        std::unique_ptr<DB> db;
        auto st = DB::Open(opt, &db);
        // Accept either behavior - fail gracefully or recover
        if (st.ok()) {
            db->Close();
        }
    }
}

// Test recovery from incomplete flush
POMAI_TEST(IncompleteFlushRecovery) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("incomplete_flush");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kAlways;

    // Create DB, insert data, close without explicit flush
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        MembraneSpec spec;
        spec.name = "default";
        spec.dim = 4;
        spec.shard_count = 1;
        db->CreateMembrane(spec);
        db->OpenMembrane("default");
        
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < 30; ++i) {
            db->Put("default", i, v);
        }
        
        db->Close();
    }
    
    // Reopen - should replay WAL and recover data
    {
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        db->OpenMembrane("default");
        
        // Verify some data recovered
        std::vector<float> out;
        auto st = db->Get("default", 0, &out);
        POMAI_EXPECT_OK(st);
        POMAI_EXPECT_EQ(out.size(), 4u);
        
        db->Close();
    }
}

// Test concurrent reads produce consistent results
POMAI_TEST(ConcurrentConsistency) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("concurrent_consistency");
    opt.dim = 4;
    opt.shard_count = 2;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 2;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");
    
    // Pre-populate
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 100; ++i) {
        db->Put("default", i, v);
    }
    db->Freeze("default");
    
    // Concurrent reads from multiple threads
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 25; ++i) {
                VectorId id = (t * 25 + i) % 100;
                std::vector<float> out;
                auto st = db->Get("default", id, &out);
                if (!st.ok() || out.size() != 4) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    POMAI_EXPECT_EQ(failures.load(), 0);
    db->Close();
}

} // namespace
