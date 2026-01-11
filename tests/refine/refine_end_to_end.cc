/*
 * tests/refine/refine_end_to_end.cc
 *
 * End-to-end integration test exercising:
 *  - VectorStore init (trains PQ)
 *  - SoA creation with codebooks/pq blocks + fingerprints
 *  - Upserts through VectorStore (PQ encode + write into SoA + publish)
 *  - Search pipeline: prefilter -> PQ approx eval -> refine::refine_topk_l2
 *
 * This test is deterministic: it uses fixed RNG seeds for dataset, PQ training
 * and fingerprint projection so stored fingerprints match query fingerprints.
 *
 * Exit: 0 on success, non-zero on failure.
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
#include <thread>
#include <chrono>
#include <cstring>
#include <iomanip>

#include "src/ai/vector_store.h"
#include "src/ai/vector_store_soa.h"
#include "src/core/config.h"
#include "src/ai/fingerprint.h"

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
    (void)argc; (void)argv;

    // Test parameters (balanced for speed/reliability in CI)
    const size_t N = 2000;
    const size_t dim = 64;       // divisible by pq_m below
    const size_t clusters = 20;
    const float cluster_spread = 0.5f;
    const uint64_t dataset_seed = 1234567ULL;
    const size_t queries = 20;
    const size_t K = 10;

    std::cerr << "[refine_e2e] N=" << N << " dim=" << dim << " clusters=" << clusters
              << " queries=" << queries << " K=" << K << "\n";

    // deterministic dataset
    auto db = make_clustered_dataset(N, dim, clusters, cluster_spread, dataset_seed);

    // Create & init VectorStore
    VectorStore store;
    if (!store.init(dim, N + 16, /*M*/8, /*ef_construction*/50, /*arena*/nullptr))
    {
        std::cerr << "Failed to init VectorStore\n";
        return 1;
    }

    // Prepare SoA mapping with PQ blocks reserved so PQ codebooks/raw codes can be stored.
    const std::string soa_path = "tmp_refine_soa.mmap";
    std::error_code ec;
    std::filesystem::remove(soa_path, ec);

    uint16_t pq_m = 8;   // must divide dim (64)
    uint16_t pq_k = 256; // trained PQ k
    uint16_t fp_bits = static_cast<uint16_t>(pomai::config::runtime.fingerprint_bits);
    if (fp_bits == 0) fp_bits = 512; // default

    auto soa = pomai::ai::soa::VectorStoreSoA::create_new(soa_path, N, static_cast<uint32_t>(dim),
                                                          pq_m, pq_k, fp_bits, /*ppe*/0, std::string());
    if (!soa)
    {
        std::cerr << "Failed to create SoA mapping\n";
        return 1;
    }
    pomai::ai::soa::VectorStoreSoA *soa_ptr = soa.get();
    store.attach_soa(std::move(soa));

    // Upsert all vectors via VectorStore to exercise the full upsert path
    for (size_t i = 0; i < N; ++i)
    {
        std::string key = "v_" + std::to_string(i);
        const float *vec = db.data() + i * dim;
        bool ok = store.upsert(key.c_str(), key.size(), vec);
        if (!ok)
        {
            std::cerr << "upsert failed for index " << i << "\n";
            return 1;
        }
    }

    // Give a short moment for background demoter / async tasks to settle (if any)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Diagnostics: check how many labels/ids/fingerprints were actually written
    size_t map_size = store.size();
    size_t ids_nonzero = 0;
    size_t fps_published = 0;
    for (size_t i = 0; i < N; ++i)
    {
        uint64_t id = soa_ptr->id_entry_at(i);
        if (id != 0) ++ids_nonzero;
        const uint8_t *fp = soa_ptr->fingerprint_ptr(i);
        if (fp) ++fps_published;
    }

    // Compute expected fingerprint bytes per slot from public API fingerprint_bits()
    size_t fp_bytes_expected = 0;
    if (soa_ptr->fingerprint_bits() > 0)
        fp_bytes_expected = static_cast<size_t>((soa_ptr->fingerprint_bits() + 7) / 8);

    std::cerr << "[diagnostics] store.size()=" << map_size
              << " ids_nonzero=" << ids_nonzero
              << " fps_published=" << fps_published
              << " soa_fp_bytes_per=" << fp_bytes_expected << "\n";

    if (map_size == 0)
    {
        std::cerr << "ERROR: store.size() == 0 after upserts\n";
        return 1;
    }
    if (ids_nonzero == 0)
    {
        std::cerr << "ERROR: No non-zero ids found in SoA after upserts\n";
        return 1;
    }
    if (fps_published == 0)
    {
        std::cerr << "ERROR: No fingerprints published in SoA; prefilter path will produce no candidates\n";
        return 1;
    }

    // --- Sanity: compare stored fingerprint for slot 0 with a local encoder using same seed ---
    {
        const float *vec0 = db.data();
        const uint64_t sanity_seed = 123456789ULL; // must match VectorStore's seed
        std::unique_ptr<FingerprintEncoder> local_enc;
        try { local_enc = FingerprintEncoder::createSimHash(dim, fp_bits, sanity_seed); }
        catch (...) { local_enc.reset(); }
        if (!local_enc)
        {
            std::cerr << "[sanity] failed to create local FingerprintEncoder\n";
        }
        else
        {
            std::vector<uint8_t> local_fp(fp_bytes_expected);
            local_enc->compute(vec0, local_fp.data());
            const uint8_t *stored_fp0 = soa_ptr->fingerprint_ptr(0);
            if (!stored_fp0)
            {
                std::cerr << "[sanity] stored_fp0 == nullptr\n";
            }
            else
            {
                bool same = (std::memcmp(local_fp.data(), stored_fp0, fp_bytes_expected) == 0);
                std::cerr << "[sanity] fingerprint(0) match local_enc? " << (same ? "YES" : "NO") << "\n";
                if (!same)
                {
                    std::cerr << "[sanity] local/stored first 16 bytes (hex):\n";
                    for (size_t i = 0; i < std::min<size_t>(16, fp_bytes_expected); ++i)
                        std::fprintf(stderr, " %02x/%02x", local_fp[i], stored_fp0[i]);
                    std::fprintf(stderr, "\n");
                }
            }
        }
    }

    // Use same deterministic fingerprint seed used by VectorStore factory (default)
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

    // Now run queries and verify refine returns reasonable results.
    std::mt19937_64 rng(dataset_seed ^ 0xfeed1234ULL);
    std::uniform_int_distribution<size_t> uni_idx(0, N - 1);

    size_t success_count = 0;
    for (size_t qi = 0; qi < queries; ++qi)
    {
        size_t qidx = uni_idx(rng);
        const float *qvec = db.data() + qidx * dim;

        // Ground-truth exact top-K
        auto exact_idx = exact_topk_l2(qvec, dim, db, N, K);
        std::unordered_set<size_t> exact_set(exact_idx.begin(), exact_idx.end());

        // Run VectorStore search (this uses prefilter -> pq_eval -> refine)
        auto res = store.search(qvec, dim, K);

        if (res.empty())
        {
            std::cerr << "[query " << qi << "] search returned empty results (unexpected)\n";
            continue;
        }

        // Parse returned keys to indices (we used "v_<idx>" keys)
        std::vector<size_t> returned_idx;
        returned_idx.reserve(res.size());
        for (auto &pr : res)
        {
            const std::string &k = pr.first;
            if (k.size() > 2 && k[0] == 'v' && k[1] == '_')
            {
                try {
                    size_t id = static_cast<size_t>(std::stoull(k.substr(2)));
                    returned_idx.push_back(id);
                } catch (...) {
                    // skip parse errors
                }
            }
            else
            {
                // fallback: parse numeric labels if present
                try {
                    size_t id = static_cast<size_t>(std::stoull(k));
                    returned_idx.push_back(id);
                } catch (...) {
                    // skip
                }
            }
        }

        // Compute recall vs exact K
        size_t found = 0;
        for (size_t r : returned_idx)
            if (exact_set.find(r) != exact_set.end())
                ++found;

        double recall = static_cast<double>(found) / static_cast<double>(K);
        std::printf("[query %2zu] qidx=%5zu returned=%zu recall=%.3f\n", qi, qidx, returned_idx.size(), recall);

        if (recall >= 0.9)
            ++success_count;
    }

    double pass_frac = static_cast<double>(success_count) / static_cast<double>(queries);
    std::cout << "[refine_e2e] pass_frac=" << pass_frac << " (need >=0.8)\n";

    // Cleanup
    std::filesystem::remove(soa_path, ec);

    if (pass_frac >= 0.8)
    {
        std::cout << "OK: refine end-to-end test passed\n";
        return 0;
    }
    else
    {
        std::cerr << "FAIL: refine end-to-end test failed\n";
        return 1;
    }
}