#include "tests/recall/recall_dataset.h"

#include <random>
#include <algorithm>
#include <cmath>

namespace pomai::test {

Dataset GenerateDataset(const DatasetOptions& opt) {
    Dataset ds;
    ds.dim = opt.dim;
    ds.data.resize(opt.num_vectors * opt.dim);
    ds.ids.resize(opt.num_vectors);
    ds.queries.resize(opt.num_queries * opt.dim);

    std::mt19937_64 rng(opt.seed);
    std::uniform_real_distribution<float> dist_uniform(-1.0f, 1.0f);
    std::uniform_int_distribution<size_t> dist_cluster(0, opt.num_clusters - 1);
    std::normal_distribution<float> dist_gauss(0.0f, opt.cluster_spread);
    std::bernoulli_distribution dist_outlier(opt.prob_outlier);

    // 1. Generate cluster centers
    std::vector<float> centers(opt.num_clusters * opt.dim);
    for (float& x : centers) {
        x = dist_uniform(rng);
    }

    // Helper to write vector
    auto gen_vec = [&](float* dst) {
        if (opt.num_clusters > 0 && !dist_outlier(rng)) {
            // Cluster mode
            size_t cid = dist_cluster(rng);
            const float* c = &centers[cid * opt.dim];
            for (uint32_t i = 0; i < opt.dim; ++i) {
                dst[i] = c[i] + dist_gauss(rng);
            }
        } else {
            // Outlier / Uniform mode
            for (uint32_t i = 0; i < opt.dim; ++i) {
                dst[i] = dist_uniform(rng);
            }
        }
    };

    // 2. Generate Data
    for (size_t i = 0; i < opt.num_vectors; ++i) {
        ds.ids[i] = static_cast<pomai::VectorId>(i + 1); // 1-based IDs
        gen_vec(&ds.data[i * opt.dim]);
        
        // Ensure non-zero norms if needed, but likelihood is low.
        // Normalize? 
        // Pomai doesn't mandate normalization, but many indexes assume it for cosine.
        // Let's leave unnormalized to stress test dot product + large/small vectors.
    }

    // 3. Generate Queries
    // Queries follow same distribution (roughly).
    for (size_t i = 0; i < opt.num_queries; ++i) {
        gen_vec(&ds.queries[i * opt.dim]);
    }

    return ds;
}

} // namespace pomai::test
