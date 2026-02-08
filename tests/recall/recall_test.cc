#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "tests/recall/recall_dataset.h"
#include "tests/common/bruteforce_oracle.h"

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <set>
#include <algorithm>

#include "core/shard/runtime.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "table/segment.h"
#include "pomai/status.h"

using namespace pomai;
using namespace pomai::table;
using namespace pomai::core;

namespace {

struct RecallMetrics {
    double recall_avg = 0.0;
    double recall_min = 1.0;
    double recall_p5 = 0.0; // 5th percentile
    std::vector<double> recalls;
};

// Compute recall intersection
double ComputeRecall(const std::vector<SearchHit>& result, 
                     const std::vector<SearchHit>& ground_truth, 
                     uint32_t k) {
    if (ground_truth.empty()) return 1.0; // degenerate?
    
    // Exact ID match
    std::set<VectorId> gt_ids;
    for (size_t i = 0; i < std::min((size_t)k, ground_truth.size()); ++i) {
        gt_ids.insert(ground_truth[i].id);
    }
    
    size_t hits = 0;
    for (const auto& h : result) {
        if (gt_ids.count(h.id)) hits++;
    }
    
    return (double)hits / (double)gt_ids.size();
}

POMAI_TEST(Recall_Clustered_Basic) {
    // 1. Generate Data
    pomai::test::DatasetOptions dopt;
    dopt.dim = 32;
    dopt.num_vectors = 5000;
    dopt.num_queries = 50;
    dopt.num_clusters = 5;
    dopt.seed = 12345;
    
    std::cout << "[RecallTest] Generating dataset..." << std::endl;
    auto ds = pomai::test::GenerateDataset(dopt);
    
    // 2. Setup Shard
    std::string path = pomai::test::TempDir("recall_test_harness");
    uint32_t shard_id = 0;
    
    auto wal = std::make_unique<storage::Wal>(path, shard_id, 1u << 20, FsyncPolicy::kNever);
    POMAI_EXPECT_OK(wal->Open());
    
    auto mem = std::make_unique<MemTable>(dopt.dim, 1u << 20);

    // Enable Parallelism
    pomai::util::ThreadPool pool(4);
    
    pomai::IndexParams index_opts;
    ShardRuntime rt(shard_id, path, dopt.dim, pomai::MembraneKind::kVector, std::move(wal),
                    std::move(mem), 1024, index_opts, &pool);
    POMAI_EXPECT_OK(rt.Start());
    
    // Keep a separate MemTable for Oracle that is NOT managed by ShardRuntime
    auto oracle_mem = std::make_unique<MemTable>(dopt.dim, 1u << 20);
    
    // 3. Ingest (Split into 5 segments to test parallelism)
    size_t chunk_size = dopt.num_vectors / 5;
    std::cout << "[RecallTest] Ingesting " << dopt.num_vectors << " vectors (5 segments)..." << std::endl;
    
    for(size_t i=0; i<dopt.num_vectors; ++i) {
        pomai::VectorId id = ds.ids[i];
        std::span<const float> vec(&ds.data[i * dopt.dim], dopt.dim);
        
        POMAI_EXPECT_OK(rt.Put(id, vec));
        POMAI_EXPECT_OK(oracle_mem->Put(id, vec));

        if ((i + 1) % chunk_size == 0) {
             core::FreezeCmd fcmd;
             auto fut = fcmd.done.get_future();
             POMAI_EXPECT_OK(rt.Enqueue(core::Command{std::move(fcmd)}));
             POMAI_EXPECT_OK(fut.get());
        }
    }
    
    // 4. Run Queries
    std::cout << "[RecallTest] Running " << dopt.num_queries << " queries..." << std::endl;
    
    uint32_t k = 10;
    std::vector<double> recalls;
    recalls.reserve(dopt.num_queries);
    
    std::vector<std::shared_ptr<SegmentReader>> empty_segments;
    
    for(size_t i=0; i<dopt.num_queries; ++i) {
        std::span<const float> query(&ds.queries[i * dopt.dim], dopt.dim);
        
        // Oracle uses separate memtable
        auto gt = pomai::test::BruteForceSearch(query, k, oracle_mem.get(), empty_segments);
        
        // System
        std::vector<SearchHit> res;
        POMAI_EXPECT_OK(rt.Search(query, k, &res));
        
        double rec = ComputeRecall(res, gt, k);
        recalls.push_back(rec);
        
        if (rec < 0.5) {
             // Verbose failure (Phase 2C - Reporting)
             std::cout << "[RecallTest] Query " << i << " FAILED recall=" << rec 
                       << " Candidates=" << res.size() << " GT_Score[0]=" << (gt.empty() ? 0 : gt[0].score) << "\n";
        }
    }
    
    // 5. Aggregate
    double sum = 0;
    double min_r = 1.0;
    for(double r : recalls) {
        sum += r;
        if (r < min_r) min_r = r;
    }
    double avg = sum / static_cast<double>(recalls.size());
    
    std::cout << "------------------------------------------------\n";
    std::cout << "RECALL REPORT\n";
    std::cout << "Dataset: " << dopt.num_vectors << " vectors, " << dopt.dim << " dim, " << dopt.num_clusters << " clusters\n";
    std::cout << "Top-K: " << k << "\n";
    std::cout << "Mean Recall: " << avg << "\n";
    std::cout << "Min Recall:  " << min_r << "\n";
    std::cout << "------------------------------------------------\n";
    

    // Phase 3: Enforce Gates
    std::cout << "[RecallTest] Average Recall: " << avg << " (Target: 0.93)" << std::endl;
    POMAI_EXPECT_TRUE(avg >= 0.93);
    POMAI_EXPECT_TRUE(min_r >= 0.10);
}

POMAI_TEST(Recall_Uniform_Hard) {
     // 1. Generate Data (Uniform)
    pomai::test::DatasetOptions dopt;
    dopt.dim = 32;
    dopt.num_vectors = 10000; // Larger
    dopt.num_queries = 50;
    dopt.num_clusters = 0; // Uniform
    dopt.seed = 999;
    
    std::cout << "[RecallTest] Generating UNIFORM dataset..." << std::endl;
    auto ds = pomai::test::GenerateDataset(dopt);
    
    // 2. Setup Shard
    std::string path = pomai::test::TempDir("recall_test_uniform");
    uint32_t shard_id = 0;
    
    auto wal = std::make_unique<storage::Wal>(path, shard_id, 1u << 20, FsyncPolicy::kNever);
    POMAI_EXPECT_OK(wal->Open());
    
    auto mem = std::make_unique<MemTable>(dopt.dim, 1u << 20);
    
    ShardRuntime rt(shard_id, path, dopt.dim, pomai::MembraneKind::kVector, std::move(wal),
                    std::move(mem), 1024, pomai::IndexParams{});
    POMAI_EXPECT_OK(rt.Start());
    
    auto oracle_mem = std::make_unique<MemTable>(dopt.dim, 1u << 20);

    // 3. Ingest
    std::cout << "[RecallTest] Ingesting..." << std::endl;
    for(size_t i=0; i<dopt.num_vectors; ++i) {
        std::span<const float> vec(&ds.data[i * dopt.dim], dopt.dim);
        POMAI_EXPECT_OK(rt.Put(ds.ids[i], vec));
        POMAI_EXPECT_OK(oracle_mem->Put(ds.ids[i], vec));
    }
    // Must Freeze
    core::FreezeCmd fcmd;
    auto fut = fcmd.done.get_future();
    POMAI_EXPECT_OK(rt.Enqueue(core::Command{std::move(fcmd)}));
    POMAI_EXPECT_OK(fut.get());
    
    // 4. Query
    std::vector<std::shared_ptr<SegmentReader>> empty_segments;
    uint32_t k = 10;
    double sum = 0;
    std::vector<double> latencies_us;
    latencies_us.reserve(dopt.num_queries);
    
    std::cout << "[RecallTest] Querying..." << std::endl;
    for(size_t i=0; i<dopt.num_queries; ++i) {
        std::span<const float> query(&ds.queries[i * dopt.dim], dopt.dim);
        auto gt = pomai::test::BruteForceSearch(query, k, oracle_mem.get(), empty_segments);
        
        auto t0 = std::chrono::steady_clock::now();
        std::vector<SearchHit> res;
        POMAI_EXPECT_OK(rt.Search(query, k, &res));
        auto t1 = std::chrono::steady_clock::now();
        
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies_us.push_back(us);
        
        sum += ComputeRecall(res, gt, k);
    }
    
    std::sort(latencies_us.begin(), latencies_us.end());
    double p50 = latencies_us[latencies_us.size() / 2];
    double p95 = latencies_us[latencies_us.size() * 95 / 100];
    
    double avg = sum / static_cast<double>(dopt.num_queries);
    std::cout << "------------------------------------------------\n";
    std::cout << "RECALL REPORT (Uniform)\n";
    std::cout << "Mean Recall: " << avg << "\n";
    std::cout << "Latency p50: " << p50 << " us\n";
    std::cout << "Latency p95: " << p95 << " us\n";
    std::cout << "------------------------------------------------\n";

    // Uniform is harder. Target 0.60 (relaxed for now).
    POMAI_EXPECT_TRUE(avg >= 0.60);
}

} // namespace
