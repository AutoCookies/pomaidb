#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

#include "pomai/types.h"

namespace pomai::test {

struct Dataset {
    std::uint32_t dim = 0;
    
    // Database vectors
    // Stored as flattened row-major: [v0_0, v0_1... | v1_0...]
    std::vector<float> data; 
    std::vector<pomai::VectorId> ids;

    // Query vectors
    // Stored as flattened row-major
    std::vector<float> queries;
};

struct DatasetOptions {
    std::uint32_t dim = 32;
    std::size_t num_vectors = 10000;
    std::size_t num_queries = 100;
    std::size_t num_clusters = 10;
    
    // Cluster params
    float cluster_spread = 0.1f; // std-dev of gaussian blob
    float prob_outlier = 0.05f;  // fraction of uniform random noise
    
    std::uint64_t seed = 42;
};

// Generates a deterministic synthetic dataset.
// Vectors are roughly in [-1, 1] range (centers in [-1, 1], noise added).
Dataset GenerateDataset(const DatasetOptions& opt);

} // namespace pomai::test
