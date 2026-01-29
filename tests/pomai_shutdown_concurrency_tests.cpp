#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "pomai_db.h"

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_shutdown_concurrency.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char *res = mkdtemp(buf.data());
    if (!res)
        throw std::runtime_error("mkdtemp failed");
    return std::string(res);
}

static void RemoveDir(const std::string &d)
{
    std::error_code ec;
    fs::remove_all(d, ec);
}

int main()
{
    std::cout << "PomaiDB shutdown concurrency test starting...\n";
    int failures = 0;

    try
    {
        const std::size_t dim = 8;
        std::string dir = MakeTempDir();

        DbOptions opt;
        opt.dim = dim;
        opt.shards = 2;
        opt.shard_queue_capacity = 256;
        opt.wal_dir = dir;
        opt.search_pool_workers = 1;
        opt.search_timeout_ms = 50;
        opt.index_build_threads = 1;
        opt.allow_sync_on_append = false;

        PomaiDB db(opt);
        db.Start();

        std::atomic<bool> stop{false};
        std::atomic<int> errors{0};

        std::thread writer([&]()
                           {
                               std::mt19937_64 rng(42);
                               std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                               Id next_id = 1;
                               while (!stop.load(std::memory_order_acquire))
                               {
                                   std::vector<UpsertRequest> batch;
                                   batch.reserve(16);
                                   for (std::size_t i = 0; i < 16; ++i)
                                   {
                                       UpsertRequest req;
                                       req.id = next_id++;
                                       req.vec.data.resize(dim);
                                       for (std::size_t d = 0; d < dim; ++d)
                                           req.vec.data[d] = dist(rng);
                                       batch.push_back(std::move(req));
                                   }
                                   try
                                   {
                                       db.UpsertBatch(std::move(batch), false).get();
                                   }
                                   catch (...)
                                   {
                                       errors.fetch_add(1, std::memory_order_relaxed);
                                   }
                               }
                           });

        std::thread checkpointer([&]()
                                 {
                                     while (!stop.load(std::memory_order_acquire))
                                     {
                                         try
                                         {
                                             db.RequestCheckpoint().get();
                                         }
                                         catch (...)
                                         {
                                             errors.fetch_add(1, std::memory_order_relaxed);
                                         }
                                         std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                     }
                                 });

        std::this_thread::sleep_for(std::chrono::seconds(2));
        stop.store(true, std::memory_order_release);
        db.Stop();

        writer.join();
        checkpointer.join();
        RemoveDir(dir);

        if (errors.load(std::memory_order_relaxed) != 0)
        {
            std::cerr << "Test FAILED: errors observed during shutdown\n";
            ++failures;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test exception: " << e.what() << "\n";
        ++failures;
    }

    if (failures == 0)
    {
        std::cout << "PomaiDB shutdown concurrency test PASS\n";
        return 0;
    }

    std::cerr << failures << " PomaiDB shutdown concurrency test FAILED\n";
    return 1;
}
