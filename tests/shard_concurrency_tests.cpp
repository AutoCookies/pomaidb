#include <atomic>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pomai/util/index_build_pool.h>
#include <pomai/core/shard.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_shard_concurrency.XXXXXX";
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
    std::cout << "Shard concurrency tests starting...\n";
    int failures = 0;

    try
    {
        const std::size_t dim = 16;
        const std::size_t batch_size = 128;
        int runtime_sec = 5;
        if (const char *env = std::getenv("POMAI_TEST_RUNTIME_SEC"))
        {
            try
            {
                runtime_sec = std::max(1, std::stoi(env));
            }
            catch (...)
            {
            }
        }
        const std::chrono::seconds runtime(runtime_sec);

        std::string dir = MakeTempDir();

        IndexBuildPool pool(1);
        pool.Start();

        Shard shard("shard-test", dim, 2048, dir);
        shard.SetIndexBuildPool(&pool);
        shard.Start();

        std::atomic<Id> max_id{0};
        std::atomic<int> errors{0};

        auto t_start = std::chrono::steady_clock::now();

        std::thread ingest([&]()
                           {
                               std::mt19937_64 rng(42);
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
                                   max_id.store(next_id - 1, std::memory_order_release);
                               }
                           });

        std::thread freezer([&]()
                           {
                               while (std::chrono::steady_clock::now() - t_start < runtime)
                               {
                                   shard.RequestEmergencyFreeze();
                                   std::this_thread::sleep_for(std::chrono::milliseconds(200));
                               }
                           });

        const std::size_t search_threads = 4;
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
                                           auto resp = shard.Search(req, budget);
                                           Id cap = max_id.load(std::memory_order_acquire);
                                           for (const auto &item : resp.items)
                                           {
                                               if (item.id > cap)
                                                   errors.fetch_add(1, std::memory_order_relaxed);
                                           }
                                       }
                                   });
        }

        ingest.join();
        freezer.join();
        for (auto &t : searchers)
            t.join();

        shard.Stop();
        pool.Stop();
        RemoveDir(dir);

        int err = errors.load(std::memory_order_relaxed);
        if (err != 0)
        {
            std::cerr << "Test FAILED: detected " << err << " invalid ids\n";
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
        std::cout << "All Shard concurrency tests PASS\n";
        return 0;
    }

    std::cerr << failures << " Shard concurrency tests FAILED\n";
    return 1;
}
