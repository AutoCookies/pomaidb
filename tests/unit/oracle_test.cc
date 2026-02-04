#include "tests/common/test_main.h"

#include <vector>
#include <memory>
#include <cmath>

#include "tests/common/bruteforce_oracle.h"
#include "table/memtable.h"
#include "pomai/status.h"

using namespace pomai;
using namespace pomai::table;

// Simple unit test for Oracle correctness & determinism
POMAI_TEST(BruteForceOracle_Basic)
{
    uint32_t dim = 2;
    // Arena block size small
    auto mem = std::make_unique<MemTable>(dim, 1024);

    // Insert verifying vectors
    // 1: (1, 0)
    // 2: (0, 1)
    // 3: (1, 1)
    std::vector<float> v1 = {1.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f};
    std::vector<float> v3 = {1.0f, 1.0f};

    POMAI_EXPECT_OK(mem->Put(1, v1));
    POMAI_EXPECT_OK(mem->Put(2, v2));
    POMAI_EXPECT_OK(mem->Put(3, v3));

    // Query: (1, 0)
    // Expected scores (Dot): 
    // 1: 1.0
    // 2: 0.0
    // 3: 1.0
    //
    // Top-k = 3.
    // Order: 
    // Score 1.0: ID 1 vs ID 3.
    // Tie-break: ID 1 < ID 3. So 1 first.
    // 3 second.
    // 2 third.
    
    std::vector<std::shared_ptr<SegmentReader>> segments; // empty for now

    std::vector<float> query = {1.0f, 0.0f};

    auto run_once = [&]() {
        return pomai::test::BruteForceSearch(query, 3, mem.get(), segments);
    };

    auto res1 = run_once();
    
    POMAI_EXPECT_EQ(res1.size(), 3u);
    POMAI_EXPECT_EQ(res1[0].id, 1u);
    POMAI_EXPECT_EQ(res1[0].score, 1.0f);
    POMAI_EXPECT_EQ(res1[1].id, 3u);
    POMAI_EXPECT_EQ(res1[1].score, 1.0f);
    POMAI_EXPECT_EQ(res1[2].id, 2u);
    POMAI_EXPECT_EQ(res1[2].score, 0.0f);

    // Determinism check
    auto res2 = run_once();
    POMAI_EXPECT_EQ(res1.size(), res2.size());
    for(size_t i=0; i<res1.size(); ++i) {
        POMAI_EXPECT_EQ(res1[i].id, res2[i].id);
        POMAI_EXPECT_EQ(res1[i].score, res2[i].score);
    }
}
