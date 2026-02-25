#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <memory>
#include <vector>
#include <thread>
#include <atomic>

namespace {

using namespace pomai;

// Test concurrent operations on a single membrane
POMAI_TEST(ShardConcurrency_ParallelPuts) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("shard_concurrency_puts");
    opt.dim = 4;
    opt.shard_count = 1; // Single shard to stress mailbox
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));
    
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Spawn multiple threads doing puts
    constexpr int num_threads = 4;
    constexpr int puts_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < puts_per_thread; ++i) {
                VectorId id = t * puts_per_thread + i;
                std::vector<float> v = {
                    static_cast<float>(id), 
                    static_cast<float>(id + 1),
                    static_cast<float>(id + 2), 
                    static_cast<float>(id + 3)
                };
                Status st = db->Put("default", id, v);
                if (!st.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    POMAI_EXPECT_EQ(failures.load(), 0);
    
    // Verify data
    db->Freeze("default");
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < puts_per_thread; ++i) {
            VectorId id = t * puts_per_thread + i;
            std::vector<float> out;
            Status st = db->Get("default", id, &out);
            POMAI_EXPECT_OK(st);
            POMAI_EXPECT_EQ(out.size(), 4u);
            if (out.size() == 4) {
                // SQ8 across 4000 elements can mean max diff of ~15.0
                POMAI_EXPECT_TRUE(std::abs(out[0] - static_cast<float>(id)) < 20.0f);
            }
        }
    }
    
    db->Close();
}

// Test mixed operations (Put, Get, Delete, Search) concurrently
POMAI_TEST(ShardConcurrency_MixedOperations) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("shard_concurrency_mixed");
    opt.dim = 4;
    opt.shard_count = 2; // Multi-shard
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 2;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Pre-populate some data
    for (int i = 0; i < 100; ++i) {
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        db->Put("default", i, v);
    }
    db->Freeze("default");

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    // Writer thread
    threads.emplace_back([&]() {
        for (int i = 100; i < 200 && !stop.load(); ++i) {
            std::vector<float> v = {5.0f, 6.0f, 7.0f, 8.0f};
            db->Put("default", i, v);
        }
    });

    // Reader threads
    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&]() {
            std::vector<float> out;
            for (int i = 0; i < 50 && !stop.load(); ++i) {
                db->Get("default", i % 100, &out);
            }
        });
    }

    // Search thread
    threads.emplace_back([&]() {
        std::vector<float> query = {1.0f, 2.0f, 3.0f, 4.0f};
        SearchResult res;
        for (int i = 0; i < 20 && !stop.load(); ++i) {
            db->Search("default", query, 10, &res);
        }
    });

    // Let them run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& th : threads) {
        th.join();
    }
    
    db->Close();
}

} // namespace
