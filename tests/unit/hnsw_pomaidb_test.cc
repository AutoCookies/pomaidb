#include "tests/common/test_main.h"
#include "third_party/pomaidb_hnsw/hnsw.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace {

float Distance(pomai::hnsw::storage_idx_t i1, pomai::hnsw::storage_idx_t i2, const std::vector<float>& vectors, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; ++i) {
        float d = vectors[i1 * dim + i] - vectors[i2 * dim + i];
        sum += d * d;
    }
    return sum;
}

POMAI_TEST(NativeHNSWInternal_Basic) {
    int dim = 4;
    int M = 4;
    pomai::hnsw::HNSW index(M, 16);
    
    std::vector<float> vectors = {
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        10, 10, 10, 10
    };
    
    pomai::hnsw::HNSW::DistanceComputer dist_func = [&](pomai::hnsw::storage_idx_t i1, pomai::hnsw::storage_idx_t i2) {
        return Distance(i1, i2, vectors, dim);
    };

    for (int i = 0; i < 5; ++i) {
        index.add_point(i, -1, dist_func);
    }
    
    std::vector<float> query = {1.1f, 1.1f, 1.1f, 1.1f};
    pomai::hnsw::HNSW::QueryDistanceComputer qdis = [&](pomai::hnsw::storage_idx_t target_id) {
        float sum = 0;
        for (int i = 0; i < dim; ++i) {
            float d = query[i] - vectors[target_id * dim + i];
            sum += d * d;
        }
        return sum;
    };
    
    std::vector<pomai::hnsw::storage_idx_t> out_ids;
    std::vector<float> out_dists;
    index.search(qdis, 2, 16, out_ids, out_dists);
    
    POMAI_EXPECT_TRUE(!out_ids.empty());
    POMAI_EXPECT_EQ(out_ids[0], 1); // Point [1,1,1,1] is closest to [1.1, 1.1, 1.1, 1.1]
}

POMAI_TEST(NativeHNSWInternal_Persistence) {
    int dim = 4;
    int M = 4;
    const char* filename = "hnsw_persist_test.bin";
    
    std::vector<float> vectors = {
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2
    };

    {
        pomai::hnsw::HNSW index(M, 16);
        pomai::hnsw::HNSW::DistanceComputer dist_func = [&](pomai::hnsw::storage_idx_t i1, pomai::hnsw::storage_idx_t i2) {
            return Distance(i1, i2, vectors, dim);
        };
        for (int i = 0; i < 3; ++i) index.add_point(i, -1, dist_func);
        
        FILE* f = fopen(filename, "wb");
        index.save(f);
        fclose(f);
    }
    
    {
        pomai::hnsw::HNSW index;
        FILE* f = fopen(filename, "rb");
        index.load(f);
        fclose(f);
        
        POMAI_EXPECT_EQ(index.M, M);
        
        std::vector<float> query = {2.1f, 2.1f, 2.1f, 2.1f};
        pomai::hnsw::HNSW::QueryDistanceComputer qdis = [&](pomai::hnsw::storage_idx_t target_id) {
            float sum = 0;
            for (int i = 0; i < dim; ++i) {
                float d = query[i] - vectors[target_id * dim + i];
                sum += d * d;
            }
            return sum;
        };
        
        std::vector<pomai::hnsw::storage_idx_t> out_ids;
        std::vector<float> out_dists;
        index.search(qdis, 1, 16, out_ids, out_dists);
        
        POMAI_EXPECT_EQ(out_ids[0], 2);
    }
    remove(filename);
}

} // namespace
