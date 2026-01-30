#include <pomai/api/pomai_db.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <thread>

using namespace pomai;

// Helper: Brute-force chuẩn để tính Ground Truth trên toàn bộ tập dữ liệu
static std::vector<Id> CalculateGroundTruth(const std::vector<float> &all_data,
                                            std::size_t N, std::size_t dim,
                                            const std::vector<float> &query,
                                            std::size_t k)
{
    struct Pair
    {
        float d;
        Id id;
    };
    std::vector<Pair> dists(N);

    for (std::size_t i = 0; i < N; ++i)
    {
        float d = 0;
        const float *v = &all_data[i * dim];
        for (std::size_t j = 0; j < dim; ++j)
        {
            float diff = v[j] - query[j];
            d += diff * diff;
        }
        dists[i] = {d, (Id)i};
    }

    std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                      [](const Pair &a, const Pair &b)
                      { return a.d < b.d; });

    std::vector<Id> out;
    for (std::size_t i = 0; i < k; ++i)
        out.push_back(dists[i].id);
    return out;
}

int main(int argc, char **argv)
{
    std::cout << ":: POMAI PRODUCTION-GRADE BENCHMARK (IVF-SQ8 READY) ::\n";

    // 1. Cấu hình tham số chuẩn Big Tech
    size_t N = 500'000;
    size_t dim = 512;
    size_t shards = 2;
    size_t probe_p = 4; // Tăng probe để đạt Recall > 90%
    size_t train_size = 50'000;
    size_t query_count = 100;
    size_t topk = 10;
    size_t rerank_k = 0;
    std::uint32_t graph_ef = 0;

    for (int i = 1; i < argc; ++i)
    {
        if (std::strncmp(argv[i], "--rerank_k=", 11) == 0)
        {
            rerank_k = static_cast<size_t>(std::stoull(argv[i] + 11));
        }
        else if (std::strncmp(argv[i], "--graph_ef=", 11) == 0)
        {
            graph_ef = static_cast<std::uint32_t>(std::stoul(argv[i] + 11));
        }
        else if (std::strncmp(argv[i], "--queries=", 10) == 0)
        {
            query_count = static_cast<size_t>(std::stoull(argv[i] + 10));
        }
        else if (std::strncmp(argv[i], "--topk=", 7) == 0)
        {
            topk = static_cast<size_t>(std::stoull(argv[i] + 7));
        }
    }

    DbOptions opt;
    opt.dim = dim;
    opt.shards = shards;
    opt.wal_dir = "./data/bench";
    opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;

    std::filesystem::remove_all(opt.wal_dir);
    PomaiDB db(opt);
    db.Start();
    db.SetProbeCount(probe_p);

    // Lưu trữ toàn bộ dữ liệu trong RAM để tính Ground Truth (tốn ~2GB cho 1M vectors)
    std::vector<float> all_vectors(N * dim);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);

    auto t_start = std::chrono::high_resolution_clock::now();

    // 2. Giai đoạn 1: Ingest dữ liệu mẫu và Train Centroids
    std::cout << "[Phase 1] Training centroids with " << train_size << " vectors...\n";
    for (size_t i = 0; i < train_size; ++i)
    {
        std::vector<UpsertRequest> batch(1);
        batch[0].id = i;
        batch[0].vec.data.resize(dim);
        for (size_t d = 0; d < dim; ++d)
        {
            float v = val(rng);
            batch[0].vec.data[d] = v;
            all_vectors[i * dim + d] = v;
        }
        db.UpsertBatch(std::move(batch), false).get();
    }
    // Kích hoạt training đồng bộ để Shards nhận diện đúng cấu trúc vùng (Voronoi)
    std::this_thread::sleep_for(std::chrono::seconds(2));
    db.RecomputeCentroids(shards * 32, train_size).get();

    // 3. Giai đoạn 2: Ingest toàn bộ dữ liệu còn lại
    std::cout << "[Phase 2] Full ingestion of " << (N - train_size) << " vectors...\n";
    for (size_t base = train_size; base < N; base += 2000)
    {
        size_t chunk = std::min<size_t>(2000, N - base);
        std::vector<UpsertRequest> batch(chunk);
        for (size_t i = 0; i < chunk; ++i)
        {
            size_t idx = base + i;
            batch[i].id = idx;
            batch[i].vec.data.resize(dim);
            for (size_t d = 0; d < dim; ++d)
            {
                float v = val(rng);
                batch[i].vec.data[d] = v;
                all_vectors[idx * dim + d] = v;
            }
        }
        db.UpsertBatch(std::move(batch), false).get();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ingest_s = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Ingest completed in " << ingest_s << "s (" << (N / ingest_s) << " ops/s)\n";

    // 4. Giai đoạn 3: Tính toán Ground Truth & Search
    std::cout << "[Phase 3] Running " << query_count << " queries with Ground Truth...\n";
    std::vector<double> latencies;
    double recall_acc = 0;

    for (size_t qi = 0; qi < query_count; ++qi)
    {
        // Lấy query ngẫu nhiên từ tập dữ liệu (đã biết ID)
        size_t target_id = std::uniform_int_distribution<size_t>(0, N - 1)(rng);
        std::vector<float> query_vec(dim);
        std::copy(all_vectors.begin() + target_id * dim,
                  all_vectors.begin() + (target_id + 1) * dim, query_vec.begin());

        // Tính kết quả chuẩn (Brute-force)
        auto ground_truth = CalculateGroundTruth(all_vectors, N, dim, query_vec, topk);

        SearchRequest req;
        req.topk = topk;
        req.candidate_k = rerank_k;
        req.graph_ef = graph_ef;
        req.metric = Metric::L2;
        req.query.data = query_vec;

        auto q0 = std::chrono::high_resolution_clock::now();
        auto resp = db.Search(req);
        auto q1 = std::chrono::high_resolution_clock::now();

        latencies.push_back(std::chrono::duration<double, std::micro>(q1 - q0).count());

        // So khớp ID
        size_t hits = 0;
        for (auto tid : ground_truth)
        {
            for (const auto &item : resp.items)
            {
                if (item.id == tid)
                {
                    hits++;
                    break;
                }
            }
        }
        recall_acc += (double)hits / topk;
    }

    // 5. Xuất kết quả
    std::sort(latencies.begin(), latencies.end());
    std::cout << "================ Search Benchmark Results (1M Vectors) ================\n";
    std::cout << "Recall@10  : " << (recall_acc / query_count * 100.0) << "%\n";
    std::cout << "Latency p50: " << latencies[query_count / 2] << " us\n";
    std::cout << "Latency p95: " << latencies[query_count * 95 / 100] << " us\n";
    std::cout << "Latency p99: " << latencies[query_count * 99 / 100] << " us\n";
    SearchRequest summary_req;
    summary_req.topk = topk;
    summary_req.candidate_k = rerank_k;
    summary_req.graph_ef = graph_ef;
    std::size_t effective_rerank_k = NormalizeCandidateK(summary_req);
    std::uint32_t effective_graph_ef = NormalizeGraphEf(summary_req, effective_rerank_k);
    std::cout << "Candidate_k: " << effective_rerank_k << "\n";
    std::cout << "Graph_ef   : " << effective_graph_ef << "\n";
    std::cout << "=======================================================================\n";

    db.Stop();
    return 0;
}
