/*
 * Standalone integration test for SimHash + prefilter pipeline (no gtest).
 *
 * - Generates a clustered dataset.
 * - Computes SimHash fingerprints for the DB and queries.
 * - Uses pomai::ai::prefilter::collect_candidates_threshold to select candidates.
 * - Verifies average recall for top-K (K=100) >= 0.95 and average reduction >= 10x.
 *
 * Exit codes:
 *  0 : success (acceptance criteria met)
 *  1 : failure (criteria not met)
 *
 * This test is intended to be run as a standalone executable (no test framework).
 */

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <cmath>

#include "src/ai/simhash.h"
#include "src/ai/prefilter.h"
#include "src/core/config.h"

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

    // Parameters tuned for reasonable CI runtime.
    const size_t N = 20000;       // number of DB vectors
    const size_t dim = 128;       // vector dimensionality
    const size_t clusters = 50;   // number of clusters
    const float cluster_spread = 0.5f;
    const uint64_t seed = 1234567ULL;
    const size_t queries = 30;    // number of queries to test
    const size_t K = 100;         // top-K to evaluate recall for

    std::cout << "[prefilter test] N=" << N << " dim=" << dim << " clusters=" << clusters << " queries=" << queries << " K=" << K << "\n";

    // Generate clustered dataset
    auto db = make_clustered_dataset(N, dim, clusters, cluster_spread, seed);

    // Build SimHash encoder using runtime/config default bits
    size_t fp_bits = static_cast<size_t>(pomai::config::runtime.fingerprint_bits);
    if (fp_bits == 0)
        fp_bits = 512;

    SimHash encoder(static_cast<size_t>(dim), fp_bits, /*seed=*/98765);

    size_t fp_bytes = encoder.bytes();

    // Compute fingerprints for entire DB into a contiguous byte buffer
    std::vector<uint8_t> db_fp;
    db_fp.resize(N * fp_bytes);
    for (size_t i = 0; i < N; ++i)
    {
        const float *vec = db.data() + i * dim;
        uint8_t *dst = db_fp.data() + i * fp_bytes;
        encoder.compute(vec, dst);
    }

    // Prefilter threshold from runtime config (fallback if 0)
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

        // compute query fingerprint
        std::vector<uint8_t> qfp(fp_bytes);
        encoder.compute(qvec, qfp.data());

        // run prefilter on contiguous DB fingerprints
        std::vector<size_t> candidates;
        pomai::ai::prefilter::collect_candidates_threshold(qfp.data(), fp_bytes, db_fp.data(), N, hamming_thresh, candidates);

        // reduction factor
        double reduction = static_cast<double>(N) / std::max<size_t>(1, candidates.size());
        sum_reduction += reduction;

        // refine: compute exact L2 among candidates only (if none, treat as zero recall)
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
        // sort and take top K among candidates
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

    std::cout << "[prefilter test] avg_recall=" << avg_recall << " avg_reduction=" << avg_reduction << "\n";

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

    if (pass)
    {
        std::cout << "OK: prefilter acceptance criteria met\n";
        return 0;
    }
    else
    {
        std::cerr << "ERROR: prefilter acceptance criteria NOT met\n";
        return 1;
    }
}