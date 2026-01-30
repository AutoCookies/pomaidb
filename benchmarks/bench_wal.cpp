#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

#include <pomai/storage/wal.h>

using namespace pomai;
namespace fs = std::filesystem;

struct BenchConfig
{
    int threads = 1;
    int batch = 64;
    int dim = 128;
    int duration_s = 5;
    bool wait_durable = false;
    std::string wal_dir = "/tmp/pomai_wal_bench";
};

static BenchConfig ParseArgs(int argc, char **argv)
{
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto next = [&](int &out)
        {
            if (i + 1 < argc)
                out = std::atoi(argv[++i]);
        };
        if (arg == "--threads")
            next(cfg.threads);
        else if (arg == "--batch")
            next(cfg.batch);
        else if (arg == "--dim")
            next(cfg.dim);
        else if (arg == "--duration")
            next(cfg.duration_s);
        else if (arg == "--wait-durable")
            cfg.wait_durable = true;
        else if (arg == "--wal-dir" && i + 1 < argc)
            cfg.wal_dir = argv[++i];
    }
    return cfg;
}

static std::vector<UpsertRequest> MakeBatch(int dim, int batch, Id base)
{
    std::vector<UpsertRequest> out;
    out.reserve(batch);
    for (int i = 0; i < batch; ++i)
    {
        UpsertRequest r;
        r.id = base + static_cast<Id>(i);
        r.vec.data.resize(dim);
        for (int d = 0; d < dim; ++d)
            r.vec.data[d] = static_cast<float>(i + d);
        out.push_back(std::move(r));
    }
    return out;
}

static double Percentile(std::vector<double> &vals, double p)
{
    if (vals.empty())
        return 0.0;
    std::sort(vals.begin(), vals.end());
    const size_t idx = static_cast<size_t>(p * (vals.size() - 1));
    return vals[idx];
}

int main(int argc, char **argv)
{
    BenchConfig cfg = ParseArgs(argc, argv);
    fs::remove_all(cfg.wal_dir);
    fs::create_directories(cfg.wal_dir);

    Wal wal("bench", cfg.wal_dir, static_cast<std::size_t>(cfg.dim));
    wal.Start();

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> vectors{0};
    std::vector<std::thread> workers;
    std::vector<std::vector<double>> latencies(cfg.threads);

    auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < cfg.threads; ++t)
    {
        workers.emplace_back([&, t]()
                             {
                                 Id base = static_cast<Id>(t * 1'000'000);
                                 while (!stop.load())
                                 {
                                     auto batch = MakeBatch(cfg.dim, cfg.batch, base);
                                     auto t0 = std::chrono::steady_clock::now();
                                     wal.AppendUpserts(batch, cfg.wait_durable);
                                     auto t1 = std::chrono::steady_clock::now();
                                     std::chrono::duration<double, std::micro> us = t1 - t0;
                                     latencies[t].push_back(us.count());
                                     batches.fetch_add(1);
                                     vectors.fetch_add(batch.size());
                                     base += static_cast<Id>(cfg.batch);
                                 } });
    }

    std::this_thread::sleep_for(std::chrono::seconds(cfg.duration_s));
    stop.store(true);
    for (auto &th : workers)
        th.join();

    wal.WaitDurable(wal.WrittenLsn());
    wal.Stop();

    std::vector<double> all_lat;
    for (auto &v : latencies)
        all_lat.insert(all_lat.end(), v.begin(), v.end());

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double ops_per_sec = static_cast<double>(vectors.load()) / elapsed.count();

    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    double cpu_seconds = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 +
                         usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;

    std::cout << "WAL benchmark\n";
    std::cout << "threads=" << cfg.threads << " batch=" << cfg.batch << " dim=" << cfg.dim
              << " duration_s=" << cfg.duration_s << " wait_durable=" << (cfg.wait_durable ? "true" : "false") << "\n";
    std::cout << "batches=" << batches.load() << " vectors=" << vectors.load() << "\n";
    std::cout << "throughput(upserts/s)=" << ops_per_sec << "\n";
    std::cout << "latency_us p50=" << Percentile(all_lat, 0.50)
              << " p95=" << Percentile(all_lat, 0.95)
              << " p99=" << Percentile(all_lat, 0.99) << "\n";
    std::cout << "cpu_seconds=" << cpu_seconds << "\n";
    std::cout << "hint: use `strace -c -p <pid>` or `perf stat` to verify syscall reduction\n";
    return 0;
}
