#include "pomai_db.h"
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

static inline float L2Sqr(const std::vector<float> &a, const std::vector<float> &b)
{
    float s = 0.0f;
    std::size_t dim = a.size();
    for (std::size_t i = 0; i < dim; ++i)
    {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Brute-force top-k over small datasets (used only on reservoir).
// Accepts data as vector of (vector, original_id) pairs and returns global Ids.
static std::vector<Id> TrueTopK(const std::vector<std::pair<std::vector<float>, Id>> &data,
                                const std::vector<float> &q,
                                std::size_t k)
{
    const std::size_t N = data.size();
    std::vector<std::pair<float, Id>> dist;
    dist.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        dist.emplace_back(L2Sqr(data[i].first, q), data[i].second);
    if (k >= N)
    {
        std::sort(dist.begin(), dist.end(),
                  [](auto &a, auto &b)
                  { return a.first < b.first; });
    }
    else
    {
        std::nth_element(dist.begin(), dist.begin() + k, dist.end(),
                         [](auto &a, auto &b)
                         { return a.first < b.first; });
        dist.resize(k);
        std::sort(dist.begin(), dist.end(), [](auto &a, auto &b)
                  { return a.first < b.first; });
    }
    std::vector<Id> out;
    out.reserve(std::min(k, dist.size()));
    for (auto &p : dist)
        out.push_back(p.second);
    return out;
}

int main(int argc, char **argv)
{
    std::cout << ":: POMAI EMBEDDED SAFE BENCHMARK (FIXED) ::\n";

    // Configurable parameters (tune for your machine)
    size_t N = 1'000'000;  // total vectors to ingest
    std::size_t dim = 512; // vector dim
    std::size_t shards = 8;
    std::size_t chunk_size = 2'000;
    std::size_t reservoir_size = 20'000;
    bool durable_per_chunk = false;

    if (argc > 1)
        N = static_cast<size_t>(std::stoull(argv[1]));
    if (argc > 2)
        dim = static_cast<size_t>(std::stoull(argv[2]));
    if (argc > 3)
        shards = static_cast<size_t>(std::stoull(argv[3]));

    std::cout << "Params: N=" << N << " dim=" << dim << " shards=" << shards
              << " chunk=" << chunk_size << " reservoir=" << reservoir_size << "\n";

    DbOptions opt;
    opt.dim = dim;
    opt.metric = Metric::L2;
    opt.shards = shards;
    opt.shard_queue_capacity = 65536;
    opt.wal_dir = "./data/bench";
    opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;

    std::filesystem::remove_all(opt.wal_dir);
    std::filesystem::create_directories(opt.wal_dir);

    PomaiDB db(opt);
    db.Start();

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);

    // reservoir now stores pair<vector, original_id>
    std::vector<std::pair<std::vector<float>, Id>> reservoir;
    reservoir.reserve(reservoir_size);

    uint64_t seen = 0;
    std::vector<UpsertRequest> chunk;
    chunk.reserve(chunk_size);

    auto t_start = std::chrono::high_resolution_clock::now();

    try
    {
        for (size_t base = 0; base < N; base += chunk_size)
        {
            size_t this_chunk = std::min(chunk_size, N - base);

            chunk.clear();
            chunk.resize(this_chunk);

            for (size_t i = 0; i < this_chunk; ++i)
            {
                size_t global_idx = base + i;
                chunk[i].id = global_idx;
                chunk[i].vec.data.resize(dim);
                for (size_t d = 0; d < dim; ++d)
                    chunk[i].vec.data[d] = val(rng);

                // reservoir sampling storing original id
                if (seen < reservoir_size)
                {
                    reservoir.emplace_back(chunk[i].vec.data, (Id)global_idx);
                }
                else
                {
                    std::uniform_int_distribution<uint64_t> dist(0, seen);
                    uint64_t j = dist(rng);
                    if (j < reservoir_size)
                        reservoir[static_cast<size_t>(j)] = std::make_pair(chunk[i].vec.data, (Id)global_idx);
                }
                ++seen;
            }

            auto fut = db.UpsertBatch(std::move(chunk), durable_per_chunk);
            fut.get();

            if (((base / chunk_size) % 50) == 0)
            {
                size_t done = std::min(N, base + this_chunk);
                double pct = 100.0 * double(done) / double(N);
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = now - t_start;
                std::cout << "[Ingest] " << done << "/" << N << " (" << std::fixed << std::setprecision(1)
                          << pct << "%) elapsed=" << elapsed.count() << "s\n";
            }
        }
    }
    catch (const std::bad_alloc &e)
    {
        std::cerr << "[Fatal] Out of memory during ingest: " << e.what() << "\n";
        db.Stop();
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Error] ingestion failed: " << e.what() << "\n";
        db.Stop();
        return 2;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> ingest_time = t_end - t_start;
    std::cout << "Ingest completed: " << N << " vectors in " << ingest_time.count() << "s ("
              << static_cast<size_t>(N / ingest_time.count()) << " ops/s)\n";

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // SEARCH BENCHMARK (uses reservoir)
    if (reservoir.empty())
    {
        std::cerr << "[Error] reservoir empty; cannot run search/recall\n";
        db.Stop();
        return 3;
    }

    const size_t Q = 200;
    const size_t topk = 10;
    std::uniform_int_distribution<size_t> qp(0, reservoir.size() - 1);

    std::vector<double> lat_us;
    lat_us.reserve(Q);
    size_t correct1 = 0;
    double recall10_acc = 0.0;

    std::cout << "[Run] Running " << Q << " queries (approximate recall over reservoir size " << reservoir.size() << ")...\n";

    for (size_t qi = 0; qi < Q; ++qi)
    {
        size_t qi_idx = qp(rng);
        auto &entry = reservoir[qi_idx];
        auto &qvec = entry.first;

        auto truek = TrueTopK(reservoir, qvec, topk); // returns global IDs now

        SearchRequest req;
        req.topk = topk;
        req.query.data = qvec;

        auto q0 = std::chrono::high_resolution_clock::now();
        SearchResponse resp = db.Search(req);
        auto q1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> d = q1 - q0;
        lat_us.push_back(d.count());

        std::vector<Id> ids;
        ids.reserve(resp.items.size());
        for (auto &it : resp.items)
            ids.push_back(it.id);

        if (!ids.empty() && ids[0] == truek[0])
            ++correct1;

        size_t hits = 0;
        for (auto tid : truek)
            if (std::find(ids.begin(), ids.end(), tid) != ids.end())
                ++hits;
        recall10_acc += double(hits) / double(topk);
    }

    std::sort(lat_us.begin(), lat_us.end());
    double avg = std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / lat_us.size();
    double p50 = lat_us[lat_us.size() * 50 / 100];
    double p95 = lat_us[std::min(lat_us.size() - 1, lat_us.size() * 95 / 100)];
    double p99 = lat_us[std::min(lat_us.size() - 1, lat_us.size() * 99 / 100)];
    double r1 = double(correct1) / double(Q);
    double r10 = recall10_acc / double(Q);

    std::cout << "================ Search Benchmark Results ================\n";
    std::cout << "Queries        : " << Q << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency avg    : " << avg << " us\n";
    std::cout << "Latency p50    : " << p50 << " us\n";
    std::cout << "Latency p95    : " << p95 << " us\n";
    std::cout << "Latency p99    : " << p99 << " us\n";
    std::cout << "Approx Recall@1: " << (r1 * 100.0) << " % (vs reservoir)\n";
    std::cout << "Approx Recall@10: " << (r10 * 100.0) << " % (vs reservoir)\n";
    std::cout << "==========================================================\n";

    db.Stop();
    return 0;
}