#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "shard.h"

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_tsan_smoke.XXXXXX";
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
    std::cout << "TSAN smoke test starting...\n";
    try
    {
        const std::size_t dim = 8;
        std::string dir = MakeTempDir();

        Shard shard("tsan-shard", dim, 64, dir);
        shard.Start();

        std::thread writer([&]()
                           {
                               std::mt19937_64 rng(7);
                               std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                               for (int i = 0; i < 10; ++i)
                               {
                                   std::vector<UpsertRequest> batch(4);
                                   for (std::size_t r = 0; r < batch.size(); ++r)
                                   {
                                       batch[r].id = static_cast<Id>(i * 10 + r);
                                       batch[r].vec.data.resize(dim);
                                       for (std::size_t d = 0; d < dim; ++d)
                                           batch[r].vec.data[d] = dist(rng);
                                   }
                                   shard.EnqueueUpserts(std::move(batch), false).get();
                               }
                           });

        std::thread reader([&]()
                           {
                               std::mt19937_64 rng(9);
                               std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                               pomai::ai::Budget budget;
                               for (int i = 0; i < 20; ++i)
                               {
                                   SearchRequest req;
                                   req.topk = 5;
                                   req.query.data.resize(dim);
                                   for (std::size_t d = 0; d < dim; ++d)
                                       req.query.data[d] = dist(rng);
                                   (void)shard.Search(req, budget);
                               }
                           });

        writer.join();
        reader.join();

        shard.Stop();
        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "TSAN smoke test FAILED: " << e.what() << "\n";
        return 1;
    }

    std::cout << "TSAN smoke test PASS\n";
    return 0;
}
