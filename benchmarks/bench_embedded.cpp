#include "pomai_db.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <filesystem>
#include <cstring>

using namespace pomai;

int main()
{
    std::cout << ":: POMAI EMBEDDED BENCHMARK (Direct Link) ::\n";

    // 1. Setup Config (Không qua YAML, config thẳng vào Struct)
    DbOptions opt;
    opt.dim = 128;
    opt.metric = Metric::L2;
    opt.shards = 4; // Tận dụng đa luồng CPU
    opt.shard_queue_capacity = 65536;
    opt.wal_dir = "./data/bench";

    // Clean start
    std::filesystem::remove_all(opt.wal_dir);
    std::filesystem::create_directories(opt.wal_dir);

    // 2. Init Engine (Zero IPC overhead)
    std::cout << "[Init] Starting Core Engine...\n";
    PomaiDB db(opt);
    db.Start();

    // 3. Prepare Data
    size_t N = 100000;
    std::cout << "[Prep] Generating " << N << " vectors (dim 128)...\n";

    std::vector<UpsertRequest> batch;
    batch.reserve(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < N; ++i)
    {
        UpsertRequest req;
        req.id = i;
        req.vec.data.resize(128);
        for (auto &x : req.vec.data)
            x = dist(rng);
        batch.push_back(std::move(req));
    }

    // 4. BENCHMARK INGEST (Pure C++ Function Call latency)
    std::cout << "[Run] Ingesting...\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Gọi thẳng vào Memory Router -> Shard Queue -> WAL -> Index
    // Không có: Serialize, Syscall send/recv, Context Switch
    auto fut = db.UpsertBatch(std::move(batch), true); // Wait Durable = true để đo cả Disk IO
    fut.wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double throughput = N / diff.count();
    std::cout << "--------------------------------------------------\n";
    std::cout << "Time       : " << diff.count() << " s\n";
    std::cout << "Throughput : " << (size_t)throughput << " ops/sec\n";
    std::cout << "Latency/Op : " << (diff.count() * 1000000 / N) << " us (kể cả WAL I/O)\n";
    std::cout << "--------------------------------------------------\n";

    // 5. Cleanup
    db.Stop();
    return 0;
}