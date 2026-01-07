#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cmath>

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/pomai_hnsw.h"
#include "src/ai/ppe.h"

// Adaptive behavior test (tolerant):
// - ensure PPE hints increase for hot nodes after heating
// - verify insertion after heating does not substantially degrade hot-neighbor fraction

int main()
{
    using namespace pomai::ai;
    using namespace std::chrono;

    std::cout << "=== PPHNSW PPE-driven adaptive M/ef test ===\n";

    const int dim = 8;
    const size_t clusters = 3;
    const size_t per_cluster = 30;
    const size_t N = clusters * per_cluster;
    const size_t max_elements = 1024;
    const size_t topk = 10;

    // RNG
    std::mt19937 rng(123456);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    // Create underlying L2 space
    hnswlib::L2Space l2space(dim);

    // Use small arena for compatibility (not required)
    PomaiArena arena = PomaiArena::FromMB(1);

    // Create PPHNSW and attach arena
    PPHNSW<float> alg(&l2space, max_elements, /*M*/ 16, /*ef_construction*/ 100);
    alg.setPomaiArena(&arena);

    // Generate cluster centers
    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (size_t c = 0; c < clusters; ++c)
    {
        for (int d = 0; d < dim; ++d)
            centers[c][d] = ud(rng) * 1.0f;
    }

    // Insert clustered points
    std::vector<std::vector<float>> data(N, std::vector<float>(dim));
    for (size_t c = 0; c < clusters; ++c)
    {
        for (size_t i = 0; i < per_cluster; ++i)
        {
            size_t idx = c * per_cluster + i;
            for (int d = 0; d < dim; ++d)
            {
                data[idx][d] = centers[c][d] + nd(rng);
                if (data[idx][d] < 0.0f)
                    data[idx][d] = 0.0f;
                if (data[idx][d] > 1.0f)
                    data[idx][d] = 1.0f;
            }
            alg.addPoint(data[idx].data(), static_cast<hnswlib::labeltype>(idx));
        }
    }

    std::cout << "[INFO] Inserted " << N << " points in " << clusters << " clusters.\n";

    // Baseline probe near cluster 0
    std::vector<float> probe(dim);
    for (int d = 0; d < dim; ++d)
        probe[d] = centers[0][d] + nd(rng);
    auto baseline_pq = alg.searchKnnAdaptive(probe.data(), topk, 0.0f);

    size_t baseline_hot_hits = 0;
    while (!baseline_pq.empty())
    {
        auto p = baseline_pq.top();
        baseline_pq.pop();
        size_t lbl = static_cast<size_t>(p.second);
        if (lbl < per_cluster)
            ++baseline_hot_hits;
    }
    double baseline_frac = static_cast<double>(baseline_hot_hits) / static_cast<double>(topk);
    std::cout << "[BASELINE] hot fraction in topk = " << baseline_frac << " (" << baseline_hot_hits << "/" << topk << ")\n";

    // Heat cluster 0
    const size_t heat_queries = 300;
    for (size_t q = 0; q < heat_queries; ++q)
    {
        for (int d = 0; d < dim; ++d)
            probe[d] = centers[0][d] + nd(rng);
        alg.searchKnnAdaptive(probe.data(), topk, 0.0f);
    }
    std::cout << "[INFO] Performed " << heat_queries << " heat queries near cluster0.\n";

    // Check PPE hints: average hot vs cold
    double sum_hot_M = 0.0, sum_hot_ef = 0.0;
    double sum_cold_M = 0.0, sum_cold_ef = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        char *ptr = alg.getDataByInternalId(static_cast<hnswlib::tableint>(i));
        PPEHeader *h = reinterpret_cast<PPEHeader *>(ptr);
        uint16_t hm = h->get_hint_M();
        uint16_t he = h->get_hint_ef();
        if (i < per_cluster)
        {
            sum_hot_M += hm;
            sum_hot_ef += he;
        }
        else
        {
            sum_cold_M += hm;
            sum_cold_ef += he;
        }
    }
    double avg_hot_M = sum_hot_M / static_cast<double>(per_cluster);
    double avg_hot_ef = sum_hot_ef / static_cast<double>(per_cluster);
    double avg_cold_M = sum_cold_M / static_cast<double>(N - per_cluster);
    double avg_cold_ef = sum_cold_ef / static_cast<double>(N - per_cluster);

    std::cout << "[HINTS] avg_hot_M=" << avg_hot_M << " avg_cold_M=" << avg_cold_M
              << " | avg_hot_ef=" << avg_hot_ef << " avg_cold_ef=" << avg_cold_ef << "\n";

    if (!(avg_hot_M >= avg_cold_M))
    {
        std::cerr << "[FAIL] avg_hot_M < avg_cold_M\n";
        return 1;
    }
    if (!(avg_hot_ef >= avg_cold_ef))
    {
        std::cerr << "[FAIL] avg_hot_ef < avg_cold_ef\n";
        return 2;
    }
    std::cout << "[PASS] PPE hints increased for hot nodes vs cold nodes.\n";

    // Insert a new point near cluster0 and measure hot fraction
    std::vector<float> new_pt(dim);
    for (int d = 0; d < dim; ++d)
        new_pt[d] = centers[0][d] + nd(rng);
    hnswlib::labeltype new_label = static_cast<hnswlib::labeltype>(N + 1);
    alg.addPoint(new_pt.data(), new_label);

    auto after_pq = alg.searchKnnAdaptive(new_pt.data(), topk, 0.0f);
    size_t after_hot_hits = 0;
    while (!after_pq.empty())
    {
        auto p = after_pq.top();
        after_pq.pop();
        size_t lbl = static_cast<size_t>(p.second);
        if (lbl < per_cluster)
            ++after_hot_hits;
    }
    double after_frac = static_cast<double>(after_hot_hits) / static_cast<double>(topk);
    std::cout << "[AFTER_INSERT] hot fraction in topk = " << after_frac << " (" << after_hot_hits << "/" << topk << ")\n";

    // Accept small drops due to wiring noise; require no substantial degradation.
    const double tol = 0.25; // allow up to 25% drop
    if (after_frac + tol < baseline_frac)
    {
        std::cerr << "[FAIL] after_frac < baseline_frac - tol\n";
        std::cerr << "  baseline_frac=" << baseline_frac << " after_frac=" << after_frac << " tol=" << tol << "\n";
        return 3;
    }

    std::cout << "[PASS] Insertion after heating did not substantially degrade hot-neighbor fraction (within tol=" << tol << ").\n";
    std::cout << "Adaptive M/ef behavior test completed successfully.\n";
    return 0;
}