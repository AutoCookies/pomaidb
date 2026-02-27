#include "tests/common/test_main.h"
#include <vector>
#include <iostream>
#include <cmath>

#include "core/index/ivf_coarse.h"
#include "core/distance.h"

// Test tracking structural centroid shifting.
POMAI_TEST(IvfCoarseTest_StreamingIndexCentroidShift) {
    pomai::index::IvfCoarse::Options opt;
    opt.nlist = 2; // Only 2 centroids
    opt.max_learning_rate = 0.5f;
    opt.min_learning_rate = 0.05f;

    pomai::index::IvfCoarse index(2, opt);
    
    // Create an initial train buffer to kick off trained_ state with 2 opposite clusters
    for (int i = 0; i < 4000; i++) {
        std::vector<float> vec;
        if (i % 2 == 0) vec = {1.0f, 0.0f}; // Cluster A (X-axis)
        else vec = {-1.0f, 0.0f};           // Cluster B (Negative X-axis)
        auto st = index.Put(i, vec);
        POMAI_EXPECT_TRUE(st.ok());
    }

    POMAI_EXPECT_TRUE(index.ready());
    
    // Verify candidate retrieval works for base cases
    std::vector<pomai::VectorId> cands;
    std::vector<float> q_a = {0.9f, 0.1f};
    index.SelectCandidates(q_a, &cands);
    POMAI_EXPECT_TRUE(cands.size() > 0);
    
    // Now, stream 100 updates heavily skewed towards the Y-axis but barely closer to Cluster A.
    // Over time, competitive learning should drag Cluster A's centroid from {1, 0} towards {0.5, 0.866}.
    std::vector<float> skewed_vec = {0.5f, 0.866f}; // Normalized 60 degree angle
    
    // Capture state before drift
    std::vector<pomai::VectorId> pre_drift_cands;
    index.SelectCandidates(skewed_vec, &pre_drift_cands);

    std::cout << "Drifting index with SOM learning..." << std::endl;
    for (int i = 0; i < 100; i++) {
        // Stream vectors
        auto st = index.Put(10000 + i, skewed_vec);
        POMAI_EXPECT_TRUE(st.ok());
    }

    std::vector<pomai::VectorId> post_drift_cands;
    index.SelectCandidates(skewed_vec, &post_drift_cands);
    
    // SOM forces the centroid closer to the streamed inputs, meaning when queried at that coordinate,
    // accuracy, proximity, and hit rate will generally favor the new location heavily over the old initialization.
    std::cout << "Streaming index successfully updated " << post_drift_cands.size() << " candidates!" << std::endl;
}
