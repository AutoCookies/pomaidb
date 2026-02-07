#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <algorithm>

using namespace pomai;

POMAI_TEST(SearchFilterTest_BasicFilter) {
    auto tmp = test::TempDir("filter_test");
    pomai::DBOptions opt;
    opt.path = tmp;
    opt.dim = 2;
    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    std::vector<float> vec = {0.1f, 0.1f};
    
    // 0 -> Tenant A
    POMAI_EXPECT_OK(db->Put(0, vec, Metadata("A")));
    // 1 -> Tenant B
    POMAI_EXPECT_OK(db->Put(1, vec, Metadata("B")));
    // 2 -> Tenant A
    POMAI_EXPECT_OK(db->Put(2, vec, Metadata("A")));
    // 3 -> No Tenant (Empty)
    POMAI_EXPECT_OK(db->Put(3, vec));

    // Case 1: Filter tenant=A
    {
        SearchOptions opts;
        opts.filters.push_back(Filter("tenant", "A"));
        
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 10, opts, &res));
        
        POMAI_EXPECT_EQ(res.hits.size(), 2UL);
        
        std::vector<VectorId> ids;
        for(const auto& h : res.hits) ids.push_back(h.id);
        std::sort(ids.begin(), ids.end());
        
        POMAI_EXPECT_EQ(ids[0], 0);
        POMAI_EXPECT_EQ(ids[1], 2);
    }

    // Case 2: Filter tenant=B
    {
        SearchOptions opts;
        opts.filters.push_back(Filter("tenant", "B"));
        
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 10, opts, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1UL);
        POMAI_EXPECT_EQ(res.hits[0].id, 1);
    }
    
    // Case 3: Empty Filter (All)
    {
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 10, &res)); // Uses default empty options
        POMAI_EXPECT_EQ(res.hits.size(), 4UL);
    }
}

POMAI_TEST(SearchFilterTest_UpdateRespectsFilter) {
    auto tmp = test::TempDir("filter_update_test");
    pomai::DBOptions opt;
    opt.path = tmp;
    opt.dim = 2;
    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    std::vector<float> vec = {0.1f, 0.1f};
    
    // 0 -> Tenant A
    POMAI_EXPECT_OK(db->Put(0, vec, Metadata("A")));
    
    // Search A -> Found
    SearchOptions optsA;
    optsA.filters.push_back(Filter("tenant", "A"));
    {
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 10, optsA, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1UL);
    }
    
    // Update 0 -> Tenant B
    POMAI_EXPECT_OK(db->Put(0, vec, Metadata("B")));
    
    // Search A -> Not Found
    {
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 10, optsA, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 0UL);
    }
    
    // Search B -> Found
    SearchOptions optsB;
    optsB.filters.push_back(Filter("tenant", "B"));
    {
        SearchResult res;
        POMAI_EXPECT_OK(db->Search(vec, 10, optsB, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1UL);
    }
}

POMAI_TEST(SearchFilterTest_TombstoneHidesEvenIfFilterMatches) {
    auto tmp = test::TempDir("filter_del_test");
    pomai::DBOptions opt;
    opt.path = tmp;
    opt.dim = 2;
    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));
    
    std::vector<float> vec = {0.1f, 0.1f};
    POMAI_EXPECT_OK(db->Put(0, vec, Metadata("A")));
    
    POMAI_EXPECT_OK(db->Delete(0));
    
    SearchOptions optsA;
    optsA.filters.push_back(Filter("tenant", "A"));
    
    SearchResult res;
    POMAI_EXPECT_OK(db->Search(vec, 10, optsA, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 0UL);
}

POMAI_TEST(SearchFilterTest_Persistence) {
    auto tmp = test::TempDir("filter_persist_test");
    std::vector<float> vec = {0.1f, 0.1f};
    
    // 1. Write data with metadata and Freeze
    {
        pomai::DBOptions opt;
        opt.path = tmp;
        opt.dim = 2;
        std::unique_ptr<DB> db;
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        
        // Use explicit membrane to ensure Freeze works
        pomai::MembraneSpec spec;
        spec.name = "test_mem";
        spec.dim = 2;
        spec.shard_count = 1;
        POMAI_EXPECT_OK(db->CreateMembrane(spec));
        POMAI_EXPECT_OK(db->OpenMembrane("test_mem"));
        
        POMAI_EXPECT_OK(db->Put("test_mem", 0, vec, Metadata("A")));
        POMAI_EXPECT_OK(db->Put("test_mem", 1, vec, Metadata("B")));
        
        // Force creation of Segments
        POMAI_EXPECT_OK(db->Freeze("test_mem"));
    }
    
    // 2. Reopen and Search Segments
    {
        pomai::DBOptions opt;
        opt.path = tmp;
        opt.dim = 2;
        std::unique_ptr<DB> db;
        // Open should load segments with metadata
        POMAI_EXPECT_OK(DB::Open(opt, &db));
        POMAI_EXPECT_OK(db->OpenMembrane("test_mem"));
        
        // First: Verify unfiltered search finds both vectors
        SearchResult res_all;
        POMAI_EXPECT_OK(db->Search("test_mem", vec, 10, &res_all));
        POMAI_EXPECT_EQ(res_all.hits.size(), 2UL); // Should find both 0 and 1
        
        // Filter A -> Should find 0, skip 1
        SearchOptions optsA;
        optsA.filters.push_back(Filter("tenant", "A"));
        
        SearchResult res;
        POMAI_EXPECT_OK(db->Search("test_mem", vec, 10, optsA, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1UL);
        POMAI_EXPECT_EQ(res.hits[0].id, 0);
        
        // Filter B -> Should find 1, skip 0
        SearchOptions optsB;
        optsB.filters.push_back(Filter("tenant", "B"));
        
        POMAI_EXPECT_OK(db->Search("test_mem", vec, 10, optsB, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 1UL);
        POMAI_EXPECT_EQ(res.hits[0].id, 1);
        
        // Filter C -> Should find nothing
        SearchOptions optsC;
        optsC.filters.push_back(Filter("tenant", "C"));
        POMAI_EXPECT_OK(db->Search("test_mem", vec, 10, optsC, &res));
        POMAI_EXPECT_EQ(res.hits.size(), 0UL);
    }
}
