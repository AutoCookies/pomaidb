#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <memory>
#include <vector>

namespace {

using namespace pomai;

static std::vector<float> MakeVec(std::uint32_t dim, float val) {
    return std::vector<float>(dim, val);
}

// Note: This test is somewhat artificial since we can't easily force
// a single shard to fail in the current architecture without sophisticated
// fault injection. Instead, we verify the API surface exists and that
// SearchResult.errors is accessible.
//
// A more realistic test would:
// 1. Use a test harness that allows shard replacement
// 2. Replace one shard with a mock that always fails
// 3. Verify results from other shards + error from failed shard
//
// For now, we verify the basic API contract.

POMAI_TEST(PartialFailure_SearchResultHasErrorsField) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("partial_failure_api");
    opt.dim = 4;
    opt.shard_count = 2; // Multi-shard setup
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));
    
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 2;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert data into both shards
    auto v1 = MakeVec(4, 1.0f);
    auto v2 = MakeVec(4, 2.0f);
    
    // ID routing: assuming simple modulo sharding
    // We want to insert into both shards
    db->Put("default", 1, v1); // Shard 1 % 2 = 1
    db->Put("default", 2, v2); // Shard 2 % 2 = 0
    
    // Search should succeed
    SearchResult res;
    POMAI_EXPECT_OK(db->Search("default", v1, 10, &res));
    
    // In normal operation, errors should be empty
    POMAI_EXPECT_TRUE(res.errors.empty());
    
    // Verify hits are present
    POMAI_EXPECT_TRUE(!res.hits.empty());
    
    db->Close();
}

POMAI_TEST(PartialFailure_EmptyErrorsOnSuccess) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("partial_empty_errors");
    opt.dim = 4;
    opt.shard_count = 4;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 4;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert across all shards
    for (int i = 0; i < 100; ++i) {
        auto v = MakeVec(4, static_cast<float>(i));
        db->Put("default", i, v);
    }

    // Search
    SearchResult res;
    auto query = MakeVec(4, 50.0f);
    POMAI_EXPECT_OK(db->Search("default", query, 10, &res));
    
    // All shards healthy, so errors should be empty
    POMAI_EXPECT_EQ(res.errors.size(), 0u);
    
    // Should have hits
    POMAI_EXPECT_TRUE(res.hits.size() > 0);
}

} // namespace
