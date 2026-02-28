#include "tests/common/test_main.h"
#include "core/simd/vector_batch.h"
#include "core/storage/compression/alp_compressor.h"
#include "core/distance.h"
#include "core/storage/wal_index.h"
#include "core/storage/io_provider.h"
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cmath>

using namespace pomai::core;

POMAI_TEST(EdgeArchitecture_ALPCompression) {
    std::cout << "Running EdgeArchitecture_ALPCompression..." << std::endl;
    std::vector<float> input = {1.23456f, 2.34567f, 3.45678f, 4.56789f};
    std::vector<int64_t> encoded(input.size());
    std::vector<float> decoded(input.size());

    auto cfg = ALPCompressor::Encode(input, encoded);
    
    ALPCompressor::Decode(encoded, decoded, cfg);

    for (size_t i = 0; i < input.size(); ++i) {
        float diff = std::abs(input[i] - decoded[i]);
        POMAI_EXPECT_TRUE(diff < 0.0001f);
    }
}

POMAI_TEST(EdgeArchitecture_VectorizedBatchSearch) {
    std::cout << "Running EdgeArchitecture_VectorizedBatchSearch..." << std::endl;
    InitDistance();
    uint32_t dim = 128;
    uint32_t batch_size = 10;
    
    FloatBatch batch(batch_size, dim);
    batch.set_size(batch_size);
    
    std::vector<float> query(dim, 1.0f);
    for (uint32_t i = 0; i < batch_size; ++i) {
        float* v = batch.data() + (i * dim);
        for (uint32_t j = 0; j < dim; ++j) v[j] = static_cast<float>(i);
    }

    std::vector<float> results(batch_size);
    SearchBatch(query, batch, DistanceMetrics::DOT, results.data());

    for (uint32_t i = 0; i < batch_size; ++i) {
        POMAI_EXPECT_EQ(results[i], static_cast<float>(i * dim));
    }
}

POMAI_TEST(EdgeArchitecture_SectorPadding) {
    std::cout << "Running EdgeArchitecture_SectorPadding..." << std::endl;
    std::string path = "/tmp/sector_pad_test.log";
    {
        int fd = ::open(path.c_str(), O_TRUNC | O_WRONLY | O_CREAT, 0644);
        pomai::storage::SectorAlignedWritableFile file(fd);
        
        // Write 100 bytes
        std::string data(100, 'A');
        POMAI_EXPECT_OK(file.Append(pomai::Slice(data)));
        
        // Sync should trigger padding to 4096
        POMAI_EXPECT_OK(file.Sync());
    }
    
    // Verify file size
    auto size = std::filesystem::file_size(path);
    POMAI_EXPECT_EQ(size % 4096, 0);
    POMAI_EXPECT_EQ(size, 4096);
    
    std::filesystem::remove(path);
}

POMAI_TEST(EdgeArchitecture_WALIndex) {
    std::cout << "Running EdgeArchitecture_WALIndex..." << std::endl;
    std::string db_name = "testdb";
    
    {
        WALIndex index(db_name);
        index.MarkCommitted(12345, 10);
    }
    
    // Re-open in a "different" process (sharing SHM)
    {
        WALIndex index(db_name);
        POMAI_EXPECT_EQ(index.header()->last_committed_offset.load(), 12345);
        POMAI_EXPECT_EQ(index.header()->n_frames.load(), 10);
    }
}
