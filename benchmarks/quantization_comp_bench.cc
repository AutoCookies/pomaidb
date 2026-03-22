#include "pomai/pomai.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <unordered_set>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct BenchResult {
    std::string name;
    double throughput = 0.0;
    double p50_ms = 0.0;
    double accuracy = 0.0;
};

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

void BenchType(const std::string& name, pomai::QuantizationType qtype, 
               const std::vector<std::vector<float>>& data, 
               const std::vector<std::vector<float>>& queries,
               const std::vector<std::vector<uint64_t>>& ground_truth,
               uint32_t dim, BenchResult* res) {
    
    std::string path = "./bench_db_q_" + name;
    fs::remove_all(path);
    
    pomai::DBOptions opts;
    opts.path = path;
    opts.dim = dim;
    opts.shard_count = 1;
    opts.index_params.quant_type = qtype;
    opts.index_params.adaptive_threshold = 1000000; // Force brute force on segments for accuracy demo
    opts.metric = pomai::MetricType::kInnerProduct;
    
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opts, &db);
    if (!st.ok()) return;
    
    // Ingestion
    for (size_t i = 0; i < data.size(); ++i) {
        db->Put(i, data[i]);
    }
    db->Flush();
    
    // Search
    std::vector<double> latencies;
    uint32_t correct_t1 = 0;
    uint32_t correct_t10 = 0;
    auto t0 = Clock::now();
    for (size_t i = 0; i < queries.size(); ++i) {
        pomai::SearchResult sres;
        auto st0 = Clock::now();
        db->Search(queries[i], 10, &sres);
        auto st1 = Clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(st1 - st0).count());
        
        if (!sres.hits.empty()) {
            if (sres.hits[0].id == ground_truth[i][0]) correct_t1++;
            
            std::unordered_set<uint64_t> gt_set(ground_truth[i].begin(), ground_truth[i].end());
            uint32_t matches = 0;
            for (const auto& hit : sres.hits) {
                if (gt_set.count(hit.id)) matches++;
            }
            correct_t10 += matches;
        }
    }
    auto t1 = Clock::now();
    
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    res->name = name;
    res->throughput = queries.size() / elapsed;
    std::sort(latencies.begin(), latencies.end());
    res->p50_ms = latencies[latencies.size() / 2];
    res->accuracy = static_cast<double>(correct_t1) / queries.size();
    double recall10 = static_cast<double>(correct_t10) / (queries.size() * 10);
    
    printf("%-10s | %12.2f | %10.4f | %9.2f%% | %9.2f%%\n", name.c_str(), res->throughput, res->p50_ms, res->accuracy * 100, recall10 * 100);
    
    fs::remove_all(path);
}

int main() {
    const uint32_t dim = 256;
    const uint32_t n_vec = 10000;
    const uint32_t n_query = 100;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<std::vector<float>> data(n_vec, std::vector<float>(dim));
    for(auto& v : data) {
        float norm = 0;
        for(auto& val : v) { val = dist(rng); norm += val*val; }
        norm = std::sqrt(norm);
        for(auto& val : v) val /= norm;
    }
    
    std::vector<std::vector<float>> queries(n_query, std::vector<float>(dim));
    for(auto& v : queries) {
        float norm = 0;
        for(auto& val : v) { val = dist(rng); norm += val*val; }
        norm = std::sqrt(norm);
        for(auto& val : v) val /= norm;
    }
    
    std::vector<std::vector<uint64_t>> ground_truth(n_query, std::vector<uint64_t>(10));
    for(uint32_t i = 0; i < n_query; ++i) {
        std::vector<std::pair<float, uint64_t>> scores;
        for(uint32_t j = 0; j < n_vec; ++j) {
            scores.emplace_back(Dot(queries[i], data[j]), j);
        }
        std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
        for(int k=0; k<10; ++k) ground_truth[i][k] = scores[k].second;
    }
    
    BenchResult r;
    
    printf("%-10s | %-12s | %-10s | %-10s | %-10s\n", "Type", "Throughput", "P50 (ms)", "Recall@1", "Recall@10");
    printf("-----------|--------------|------------|-----------|-----------\n");
    
    BenchType("Float32", pomai::QuantizationType::kNone, data, queries, ground_truth, dim, &r);
    BenchType("Int8 (SQ8)", pomai::QuantizationType::kSq8, data, queries, ground_truth, dim, &r);
    BenchType("FP16", pomai::QuantizationType::kFp16, data, queries, ground_truth, dim, &r);
    BenchType("1-Bit", pomai::QuantizationType::kBit, data, queries, ground_truth, dim, &r);
    
    return 0;
}
