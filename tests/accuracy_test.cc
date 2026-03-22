#include "tests/common/test_main.h"
#include "pomai/pomai.h"
#include <random>
#include <vector>
#include <iostream>
#include <filesystem>

namespace {

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

void TestAccuracy(pomai::QuantizationType qtype, const std::string& name) {
    const uint32_t dim = 128;
    const uint32_t n_vec = 1000;
    const uint32_t n_query = 50;
    const std::string path = "./test_accuracy_" + name;
    std::filesystem::remove_all(path);

    pomai::DBOptions opts;
    opts.path = path;
    opts.dim = dim;
    opts.shard_count = 1;
    opts.index_params.quant_type = qtype;
    opts.metric = pomai::MetricType::kInnerProduct;

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opts, &db));

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> data(n_vec, std::vector<float>(dim));
    for (auto& v : data) {
        float norm = 0;
        for (auto& val : v) { val = dist(rng); norm += val * val; }
        norm = std::sqrt(norm);
        for (auto& val : v) val /= norm;
    }

    for (size_t i = 0; i < n_vec; ++i) {
        POMAI_EXPECT_OK(db->Put(i, data[i]));
    }
    POMAI_EXPECT_OK(db->Flush());

    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_query; ++i) {
        std::vector<float> q(dim);
        float norm = 0;
        for (auto& val : q) { val = dist(rng); norm += val * val; }
        norm = std::sqrt(norm);
        for (auto& val : q) val /= norm;

        // Ground truth
        float best_s = -1e9;
        uint64_t best_id = 0;
        for (uint32_t j = 0; j < n_vec; ++j) {
            float s = Dot(q, data[j]);
            if (s > best_s) { best_s = s; best_id = j; }
        }

        pomai::SearchResult res;
        POMAI_EXPECT_OK(db->Search(q, 1, &res));
        if (!res.hits.empty() && res.hits[0].id == best_id) {
            hits++;
        }
    }

    double accuracy = static_cast<double>(hits) / n_query;
    std::cout << "[Accuracy Test] " << name << ": " << (accuracy * 100.0) << "%" << std::endl;
    POMAI_EXPECT_TRUE(accuracy > 0.95);

    std::filesystem::remove_all(path);
}

POMAI_TEST(QuantizationAccuracy_None) {
    TestAccuracy(pomai::QuantizationType::kNone, "None");
}

POMAI_TEST(QuantizationAccuracy_Sq8) {
    TestAccuracy(pomai::QuantizationType::kSq8, "Sq8");
}

POMAI_TEST(QuantizationAccuracy_Fp16) {
    TestAccuracy(pomai::QuantizationType::kFp16, "Fp16");
}

POMAI_TEST(QuantizationAccuracy_Bit) {
    TestAccuracy(pomai::QuantizationType::kBit, "Bit");
}

} // namespace
