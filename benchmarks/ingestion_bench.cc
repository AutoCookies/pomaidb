// Quick Ingestion Throughput Benchmark
// Measures raw Put() performance without search overhead

#include "pomai/pomai.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono;

int main(int argc, char** argv) {
    bool use_batch = false;
    uint32_t batch_size = 1000;
    uint32_t num_vectors = 1000000;
    uint32_t dim = 128;
    
    // Simple arg parsing
    for (int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--batch") {
            use_batch = true;
        } else if (arg == "--batch-size" && i+1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else if (i == 1 && arg[0] != '-') {
            num_vectors = std::atoi(argv[i]);
        } else if (i == 2 && arg[0] != '-') {
            dim = std::atoi(argv[i]);
        }
    }
    
    printf("=============================================================\n");
    printf("              Ingestion Throughput Benchmark\n");
    printf("=============================================================\n");
    printf("Vectors:    %u\n", num_vectors);
    printf("Dims:       %u\n", dim);
    printf("Mode:       %s\n", use_batch ? "BATCH" : "SEQUENTIAL");
    if (use_batch) printf("Batch Size: %u\n", batch_size);
    printf("=============================================================\n\n");
    
    // Generate data
    printf("[1/3] Generating %u vectors...\n", num_vectors);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<std::vector<float>> vectors(num_vectors);
    for (auto& vec : vectors) {
        vec.resize(dim);
        for (auto& val : vec) val = dist(rng);
    }
    printf("Generated %.2f MB of data\n\n", 
           (num_vectors * dim * sizeof(float)) / 1024.0 / 1024.0);
    
    // Setup DB
    printf("[2/3] Setting up database...\n");
    std::filesystem::remove_all("/tmp/ingestion_bench");
    
    pomai::DBOptions opts;
    opts.path = "/tmp/ingestion_bench";
    opts.dim = dim;
    opts.shard_count = std::thread::hardware_concurrency();
    opts.fsync = pomai::FsyncPolicy::kNever;  // Disable for max throughput
    
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opts, &db);
    if (!st.ok()) {
        fprintf(stderr, "Open failed: %s\n", st.message().c_str());
        return 1;
    }
    printf("DB opened with %u shards\n\n", opts.shard_count);
    
    // Benchmark ingestion
    printf("[3/3] Ingesting %u vectors...\n", num_vectors);
    
    auto start = high_resolution_clock::now();
    
    if (use_batch) {
        std::vector<pomai::VectorId> ids_batch;
        std::vector<std::span<const float>> vecs_batch;
        ids_batch.reserve(batch_size);
        vecs_batch.reserve(batch_size);
        
        for (uint32_t i = 0; i < num_vectors; ++i) {
            ids_batch.push_back(i);
            vecs_batch.push_back(std::span<const float>(vectors[i]));
            
            if (ids_batch.size() >= batch_size || i == num_vectors - 1) {
                st = db->PutBatch(ids_batch, vecs_batch);
                if (!st.ok()) {
                     fprintf(stderr, "PutBatch failed at %u: %s\n", i, st.message().c_str());
                     return 1;
                }
                ids_batch.clear();
                vecs_batch.clear();
            }
            
            if ((i + 1) % 10000 == 0) {
                printf("  Progress: %u/%u\r", i + 1, num_vectors);
                fflush(stdout);
            }
        }
    } else {
        for (uint32_t i = 0; i < num_vectors; ++i) {
            st = db->Put(i, vectors[i]);
            if (!st.ok()) {
                fprintf(stderr, "Put failed at %u: %s\n", i, st.message().c_str());
                return 1;
            }
            
            if ((i + 1) % 10000 == 0) {
                printf("  Progress: %u/%u\r", i + 1, num_vectors);
                fflush(stdout);
            }
        }
    }
    
    auto end = high_resolution_clock::now();
    
    printf("\n\n");
    printf("=============================================================\n");
    printf("                      RESULTS\n");
    printf("=============================================================\n");
    
    double elapsed_sec = duration<double>(end - start).count();
    double throughput_vecs = num_vectors / elapsed_sec;
    double throughput_mb = (num_vectors * dim * sizeof(float)) / elapsed_sec / 1024.0 / 1024.0;
    
    printf("Total Time:       %.3f sec\n", elapsed_sec);
    printf("Throughput:       %.0f vectors/sec\n", throughput_vecs);
    printf("                  %.2f MB/sec\n", throughput_mb);
    printf("Avg Latency:      %.2f µs/vector\n", (elapsed_sec / num_vectors) * 1e6);
    printf("\n");
    
    // Estimated overhead breakdown
    printf("ESTIMATED BOTTLENECKS:\n");
    double wal_overhead_mb = throughput_mb;  // Assume WAL = main write
    printf("  WAL writes:     ~%.2f MB/sec (dominant)\n", wal_overhead_mb);
    printf("  MemTable:       ~%.0f ops/sec (fast)\n", throughput_vecs);
    printf("\n");
    
    // Comparison
    printf("OPTIMIZATION POTENTIAL:\n");
    if (!use_batch) {
        printf("  Current:        %.0f vectors/sec\n", throughput_vecs);
        printf("  With batching:  ~%.0f vectors/sec (5-10x)\n", throughput_vecs * 7);
    } else {
        printf("  Sequential:     (previous run)\n");
        printf("  Batching:       %.0f vectors/sec\n", throughput_vecs);
    }
    printf("  Theoretical:    ~100,000 vectors/sec (SSD limit)\n");
    printf("=============================================================\n");
    
    // Cleanup
    db->Close();
    std::filesystem::remove_all("/tmp/ingestion_bench");
    
    return 0;
}
