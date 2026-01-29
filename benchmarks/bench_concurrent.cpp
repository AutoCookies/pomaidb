#include "pomai_db.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
#include <sys/resource.h>

using namespace pomai;

static std::size_t PeakRssBytes()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return 0;
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
}

static double Percentile(const std::vector<double> &sorted, double pct)
{
    if (sorted.empty())
        return 0.0;
    std::size_t idx = static_cast<std::size_t>(pct * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

int main()
{
    const std::size_t dim = 128;
    const std::size_t total_vectors = 30000;
    const std::size_t batch_size = 500;
    const std::size_t query_pool = 2000;
    const std::chrono::seconds max_runtime(6);

    DbOptions opt;
    opt.dim = dim;
    opt.metric = Metric::L2;
    opt.shards = 4;
    opt.shard_queue_capacity = 4096;
    opt.wal_dir = "./data/bench_concurrent";
    opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;

    std::filesystem::remove_all(opt.wal_dir);
    std::filesystem::create_directories(opt.wal_dir);

    PomaiDB db(opt);
    db.Start();

    std::mt19937_64 rng_seed(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> queries;
    queries.reserve(query_pool);
    for (std::size_t i = 0; i < query_pool; ++i)
    {
        std::vector<float> v(dim);
        for (std::size_t d = 0; d < dim; ++d)
            v[d] = dist(rng_seed);
        queries.push_back(std::move(v));
    }

    std::atomic<bool> ingest_done{false};
    std::atomic<bool> stop_search{false};
    std::vector<double> latencies_us;
    latencies_us.reserve(5000);
    std::mutex lat_mu;

    auto t_start = std::chrono::steady_clock::now();

    std::thread ingest([&]()
                       {
                           std::mt19937_64 rng(123);
                           std::vector<UpsertRequest> batch;
                           batch.reserve(batch_size);
                           std::size_t id = 0;
                           for (std::size_t base = 0; base < total_vectors; base += batch_size)
                           {
                               const std::size_t this_batch = std::min(batch_size, total_vectors - base);
                               batch.clear();
                               batch.resize(this_batch);
                               for (std::size_t i = 0; i < this_batch; ++i)
                               {
                                   batch[i].id = static_cast<Id>(id++);
                                   batch[i].vec.data.resize(dim);
                                   for (std::size_t d = 0; d < dim; ++d)
                                       batch[i].vec.data[d] = dist(rng);
                               }
                               db.UpsertBatch(std::move(batch), false).get();
                               batch.clear();
                               if (std::chrono::steady_clock::now() - t_start > max_runtime)
                                   break;
                           }
                           ingest_done.store(true, std::memory_order_release);
                       });

    std::thread searcher([&]()
                         {
                             std::mt19937_64 rng(456);
                             std::uniform_int_distribution<std::size_t> pick(0, queries.size() - 1);
                             while (!stop_search.load(std::memory_order_acquire))
                             {
                                 SearchRequest req;
                                 req.topk = 10;
                                 req.query.data = queries[pick(rng)];
                                 auto t0 = std::chrono::steady_clock::now();
                                 (void)db.Search(req);
                                 auto t1 = std::chrono::steady_clock::now();
                                 std::chrono::duration<double, std::micro> d = t1 - t0;
                                 {
                                     std::lock_guard<std::mutex> lk(lat_mu);
                                     latencies_us.push_back(d.count());
                                 }
                                 if (ingest_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() - t_start > max_runtime)
                                     break;
                             }
                         });

    while (!ingest_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() - t_start < max_runtime)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    stop_search.store(true, std::memory_order_release);
    ingest.join();
    searcher.join();

    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_s = t_end - t_start;

    std::sort(latencies_us.begin(), latencies_us.end());

    double p50 = Percentile(latencies_us, 0.50);
    double p95 = Percentile(latencies_us, 0.95);
    double p99 = Percentile(latencies_us, 0.99);
    double ops = latencies_us.empty() ? 0.0 : static_cast<double>(latencies_us.size()) / total_s.count();

    std::cout << ":: POMAI CONCURRENT INGEST+SEARCH BENCH ::\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency p50 (us): " << p50 << "\n";
    std::cout << "Latency p95 (us): " << p95 << "\n";
    std::cout << "Latency p99 (us): " << p99 << "\n";
    std::cout << "Search ops/s: " << ops << "\n";
    std::cout << "Peak RSS bytes: " << PeakRssBytes() << "\n";

    db.Stop();
    return 0;
}
