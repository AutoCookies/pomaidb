/*
 * Standalone integration test for VectorStore + SoA + SimHash + prefilter.
 *
 * - Uses VectorStore::upsert(...) to insert N vectors so the same code path
 *   that computes fingerprints and writes into the SoA mapping is exercised.
 * - After inserts, reads the SoA fingerprint block and runs prefilter::collect_candidates_threshold.
 * - Computes exact top-K (by L2) from the in-memory DB copy and measures recall
 *   and scan reduction achieved by the prefilter.
 *
 * Exit codes:
 *  0 : success (acceptance criteria met)
 *  1 : failure (criteria not met or error)
 *
 * Build: compile with your project build rules (add this file into test target).
 * Example (rough, depends on your project layout):
 *   g++ -std=c++17 -I. tests/prefilter/vector_store_prefilter_integration.cc \
 *       src/ai/simhash.cc src/ai/prefilter.cc src/ai/vector_store_soa.cc src/ai/vector_store.cc \
 *       src/ai/fingerprint.cc src/ai/refine.cc ... -o vector_store_prefilter_test
 *
 * Run: ./vector_store_prefilter_test
 */

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <filesystem>

#include "src/ai/fingerprint.h" // use factory so we match the same seed/projection used by VectorStore
#include "src/ai/prefilter.h"
#include "src/core/config.h"
#include "src/ai/vector_store.h"
#include "src/ai/vector_store_soa.h"

using namespace pomai::ai;

// Helper: generate clustered dataset
static std::vector<float> make_clustered_dataset(size_t N, size_t dim, size_t clusters, float cluster_spread, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, cluster_spread);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    // create random cluster centers
    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (size_t c = 0; c < clusters; ++c)
        for (size_t d = 0; d < dim; ++d)
            centers[c][d] = uni(rng) * 10.0f; // spread centers

    std::vector<float> data;
    data.resize(N * dim);

    for (size_t i = 0; i < N; ++i)
    {
        size_t c = static_cast<size_t>(rng() % clusters);
        for (size_t d = 0; d < dim; ++d)
        {
            float v = centers[c][d] + nd(rng);
            data[i * dim + d] = v;
        }
    }

    return data;
}

// exact top-K L2 (returns indices)
static std::vector<size_t> exact_topk_l2(const float *query, size_t dim,
                                         const std::vector<float> &db, size_t db_count, size_t K)
{
    struct Item { float dist; size_t idx; };
    std::vector<Item> items;
    items.reserve(db_count);
    for (size_t i = 0; i < db_count; ++i)
    {
        double acc = 0.0;
        const float *vec = db.data() + i * dim;
        for (size_t d = 0; d < dim; ++d)
        {
            double diff = static_cast<double>(query[d]) - static_cast<double>(vec[d]);
            acc += diff * diff;
        }
        items.push_back({static_cast<float>(acc), i});
    }
    if (items.size() > K)
    {
        std::nth_element(items.begin(), items.begin() + static_cast<ptrdiff_t>(K), items.end(),
                         [](const Item &a, const Item &b) { return a.dist < b.dist; });
        items.resize(K);
    }
    std::sort(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.dist < b.dist; });

    std::vector<size_t> out;
    out.reserve(items.size());
    for (auto &it : items)
        out.push_back(it.idx);
    return out;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    // Parameters (adjust for local/CI as desired)
    const size_t N = 20000;       // number of DB vectors
    const size_t dim = 128;       // vector dimensionality
    const size_t clusters = 50;   // number of clusters
    const float cluster_spread = 0.5f;
    const uint64_t seed = 1234567ULL;
    const size_t queries = 30;    // number of queries to test
    const size_t K = 100;         // top-K to evaluate recall for

    std::cout << "[vector_store prefilter test] N=" << N << " dim=" << dim << " clusters=" << clusters << " queries=" << queries << " K=" << K << "\n";

    // Generate clustered dataset (keep host copy for exact L2)
    auto db = make_clustered_dataset(N, dim, clusters, cluster_spread, seed);

    // Create and init VectorStore
    VectorStore store;
    if (!store.init(dim, N + 16, /*M*/8, /*ef_construction*/50, /*arena*/nullptr))
    {
        std::cerr << "Failed to init VectorStore\n";
        return 1;
    }

    // Create SoA mapping on disk (temporary path) and attach
    const std::string soa_path = "tmp_prefilter_soa.mmap";
    // remove if exists
    std::error_code ec;
    std::filesystem::remove(soa_path, ec);

    uint16_t fp_bits = static_cast<uint16_t>(pomai::config::runtime.fingerprint_bits);
    if (fp_bits == 0)
        fp_bits = 512;

    auto soa = pomai::ai::soa::VectorStoreSoA::create_new(soa_path, N, static_cast<uint32_t>(dim), /*pq_m*/0, /*pq_k*/0, fp_bits, /*ppe*/0, std::string());
    if (!soa)
    {
        std::cerr << "Failed to create SoA mapping\n";
        return 1;
    }
    // Keep a raw pointer to the SoA for direct fingerprint access in the test
    pomai::ai::soa::VectorStoreSoA *soa_ptr = soa.get();
    store.attach_soa(std::move(soa));

    // Insert all vectors via VectorStore::upsert to exercise the exact integration path
    for (size_t i = 0; i < N; ++i)
    {
        std::string key = "vec_" + std::to_string(i);
        const float *vec = db.data() + i * dim;
        bool ok = store.upsert(key.c_str(), key.size(), vec);
        if (!ok)
        {
            std::cerr << "upsert failed for index " << i << "\n";
            return 1;
        }
    }

    // Ensure SoA publish completed: fingerprint_ptr(0) should be non-null
    const uint8_t *db_fp_base = soa_ptr->fingerprint_ptr(0);
    if (!db_fp_base)
    {
        std::cerr << "fingerprint_ptr(0) returned nullptr after inserts; aborting\n";
        return 1;
    }

    // Build a FingerprintEncoder using the SAME seed used by the VectorStore path.
    // VectorStore uses FingerprintEncoder::createSimHash(dim, bits) with the default seed
    // (123456789ULL) when it creates the encoder. Use the same seed here so query fps
    // match the DB fingerprints stored by VectorStore.
    const uint64_t fingerprint_seed = 123456789ULL;
    std::unique_ptr<FingerprintEncoder> fp_enc;
    try
    {
        fp_enc = FingerprintEncoder::createSimHash(dim, fp_bits, fingerprint_seed);
    }
    catch (...)
    {
        std::cerr << "Failed to create FingerprintEncoder for test\n";
        return 1;
    }
    if (!fp_enc)
    {
        std::cerr << "FingerprintEncoder factory returned null\n";
        return 1;
    }
    size_t fp_bytes = fp_enc->bytes();

    // Prefilter threshold from runtime config (fallback)
    uint32_t hamming_thresh = static_cast<uint32_t>(pomai::config::runtime.prefilter_hamming_threshold);
    if (hamming_thresh == 0)
        hamming_thresh = 128;

    std::mt19937_64 rng(seed ^ 0xdeadbeef);
    std::uniform_int_distribution<size_t> uni_idx(0, N - 1);

    double sum_recall = 0.0;
    double sum_reduction = 0.0;

    for (size_t qi = 0; qi < queries; ++qi)
    {
        size_t qidx = uni_idx(rng);
        const float *qvec = db.data() + qidx * dim;

        // exact top-K
        auto exact_topk = exact_topk_l2(qvec, dim, db, N, K);
        std::unordered_set<size_t> exact_set(exact_topk.begin(), exact_topk.end());

        // compute query fingerprint using same encoder (matching seed/projections)
        std::vector<uint8_t> qfp(fp_bytes);
        fp_enc->compute(qvec, qfp.data());

        // run prefilter against SoA fingerprint block (contiguous rows)
        std::vector<size_t> candidates;
        pomai::ai::prefilter::collect_candidates_threshold(qfp.data(), fp_bytes, db_fp_base, N, hamming_thresh, candidates);

        // reduction factor
        double reduction = static_cast<double>(N) / std::max<size_t>(1, candidates.size());
        sum_reduction += reduction;

        // refine among candidates (exact L2)
        std::vector<std::pair<float, size_t>> cand_dists;
        cand_dists.reserve(candidates.size());
        for (size_t cidx : candidates)
        {
            double acc = 0.0;
            const float *vec = db.data() + cidx * dim;
            for (size_t d = 0; d < dim; ++d)
            {
                double diff = static_cast<double>(qvec[d]) - static_cast<double>(vec[d]);
                acc += diff * diff;
            }
            cand_dists.emplace_back(static_cast<float>(acc), cidx);
        }
        std::sort(cand_dists.begin(), cand_dists.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        size_t taken = std::min<size_t>(K, cand_dists.size());
        size_t found = 0;
        for (size_t i = 0; i < taken; ++i)
        {
            if (exact_set.find(cand_dists[i].second) != exact_set.end())
                ++found;
        }

        double recall = 0.0;
        if (K > 0)
            recall = static_cast<double>(found) / static_cast<double>(K);
        sum_recall += recall;

        std::printf("[query %zu] qidx=%zu candidates=%zu reduction=%.2f recall=%.4f\n",
                    qi, qidx, candidates.size(), reduction, recall);
    }

    double avg_recall = sum_recall / static_cast<double>(queries);
    double avg_reduction = sum_reduction / static_cast<double>(queries);

    std::cout << "[vector_store prefilter test] avg_recall=" << avg_recall << " avg_reduction=" << avg_reduction << "\n";

    const double required_recall = 0.95;
    const double required_reduction = 10.0;

    bool pass = true;
    if (avg_recall < required_recall)
    {
        std::cerr << "FAIL: avg_recall " << avg_recall << " < required " << required_recall << "\n";
        pass = false;
    }
    if (avg_reduction < required_reduction)
    {
        std::cerr << "FAIL: avg_reduction " << avg_reduction << " < required " << required_reduction << "\n";
        pass = false;
    }

    // cleanup
    std::filesystem::remove(soa_path, ec);

    if (pass)
    {
        std::cout << "OK: VectorStore+SoA prefilter acceptance criteria met\n";
        return 0;
    }
    else
    {
        std::cerr << "ERROR: VectorStore+SoA prefilter acceptance criteria NOT met\n";
        return 1;
    }
}