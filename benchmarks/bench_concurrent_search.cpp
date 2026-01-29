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

#include "index_build_pool.h"
#include "shard.h"

using namespace pomai;
namespace fs = std::filesystem;

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

static double CpuPercent(std::chrono::steady_clock::duration wall)
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return 0.0;
    double user = static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) / 1e6;
    double sys = static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec) / 1e6;
    double wall_s = std::chrono::duration<double>(wall).count();
    if (wall_s <= 0.0)
        return 0.0;
    return ((user + sys) / wall_s) * 100.0;
}

int main()
{
    const std::size_t dim = 64;
    const std::size_t batch_size = 256;
    const std::size_t search_threads = 4;
    const std::chrono::seconds runtime(10);

    std::string wal_dir = "./data/bench_concurrent_search";
    fs::remove_all(wal_dir);
    fs::create_directories(wal_dir);

    IndexBuildPool pool(2);
    pool.Start();

    Shard shard("bench-shard", dim, 4096, wal_dir);
    shard.SetIndexBuildPool(&pool);
    shard.Start();

    std::atomic<std::size_t> ingest_ops{0};

    std::vector<double> latencies_us;
    latencies_us.reserve(20000);
    std::mutex lat_mu;

    auto t_start = std::chrono::steady_clock::now();

    std::thread ingest([&]()
                       {
                           std::mt19937_64 rng(123);
                           std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                           Id next_id = 1;
                           while (std::chrono::steady_clock::now() - t_start < runtime)
                           {
                               std::vector<UpsertRequest> batch;
                               batch.reserve(batch_size);
                               for (std::size_t i = 0; i < batch_size; ++i)
                               {
                                   UpsertRequest req;
                                   req.id = next_id++;
                                   req.vec.data.resize(dim);
                                   for (std::size_t d = 0; d < dim; ++d)
                                       req.vec.data[d] = dist(rng);
                                   batch.push_back(std::move(req));
                               }
                               shard.EnqueueUpserts(std::move(batch), false).get();
                               ingest_ops.fetch_add(batch_size, std::memory_order_relaxed);
                           }
                       });

    std::thread freezer([&]()
                       {
                           while (std::chrono::steady_clock::now() - t_start < runtime)
                           {
                               shard.RequestEmergencyFreeze();
                               std::this_thread::sleep_for(std::chrono::milliseconds(250));
                           }
                       });

    std::vector<std::thread> searchers;
    searchers.reserve(search_threads);
    for (std::size_t t = 0; t < search_threads; ++t)
    {
        searchers.emplace_back([&, t]()
                               {
                                   std::mt19937_64 rng(1000 + t);
                                   std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                                   pomai::ai::Budget budget;
                                   while (std::chrono::steady_clock::now() - t_start < runtime)
                                   {
                                       SearchRequest req;
                                       req.topk = 10;
                                       req.query.data.resize(dim);
                                       for (std::size_t d = 0; d < dim; ++d)
                                           req.query.data[d] = dist(rng);
                                       auto t0 = std::chrono::steady_clock::now();
                                       auto resp = shard.Search(req, budget);
                                       auto t1 = std::chrono::steady_clock::now();
                                       std::chrono::duration<double, std::micro> d = t1 - t0;
                                       {
                                           std::lock_guard<std::mutex> lk(lat_mu);
                                           latencies_us.push_back(d.count());
                                       }
                                       (void)resp;
                                   }
                               });
    }

    while (std::chrono::steady_clock::now() - t_start < runtime)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    ingest.join();
    freezer.join();
    for (auto &t : searchers)
        t.join();

    auto t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_s = t_end - t_start;

    std::sort(latencies_us.begin(), latencies_us.end());

    double p50 = Percentile(latencies_us, 0.50);
    double p95 = Percentile(latencies_us, 0.95);
    double p99 = Percentile(latencies_us, 0.99);
    double ops = latencies_us.empty() ? 0.0 : static_cast<double>(latencies_us.size()) / total_s.count();
    double ingest_rate = static_cast<double>(ingest_ops.load(std::memory_order_relaxed)) / total_s.count();

    std::cout << ":: POMAI CONCURRENT SEARCH+FREEZE BENCH ::\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency p50 (us): " << p50 << "\n";
    std::cout << "Latency p95 (us): " << p95 << "\n";
    std::cout << "Latency p99 (us): " << p99 << "\n";
    std::cout << "Search ops/s: " << ops << "\n";
    std::cout << "Ingest ops/s: " << ingest_rate << "\n";
    std::cout << "CPU usage (%): " << CpuPercent(t_end - t_start) << "\n";
    std::cout << "Peak RSS bytes: " << PeakRssBytes() << "\n";

    shard.Stop();
    pool.Stop();
    return 0;
}
