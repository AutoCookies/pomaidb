#include <atomic>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <unistd.h>

#include "wal.h"

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_wal_concurrency.XXXXXX";
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

static std::vector<UpsertRequest> MakeBatch(std::size_t dim, std::size_t count, Id base)
{
    std::vector<UpsertRequest> batch;
    batch.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        UpsertRequest r;
        r.id = base + static_cast<Id>(i);
        r.vec.data.resize(dim);
        for (std::size_t j = 0; j < dim; ++j)
            r.vec.data[j] = static_cast<float>(i + j);
        batch.push_back(std::move(r));
    }
    return batch;
}

int main()
{
    int failures = 0;
    try
    {
        std::string dir = MakeTempDir();
        Wal w("shard-0", dir, 4);
        w.Start();

        const int threads = 4;
        const int iterations = 50;
        std::vector<Lsn> lsns;
        std::mutex lsns_mu;

        std::vector<std::thread> workers;
        for (int t = 0; t < threads; ++t)
        {
            workers.emplace_back([&, t]()
                                 {
                                     for (int i = 0; i < iterations; ++i)
                                     {
                                         auto batch = MakeBatch(4, 2, t * 1000 + i * 10);
                                         bool wait = (i % 10 == 0);
                                         Lsn lsn = w.AppendUpserts(batch, wait);
                                         if (wait)
                                             w.WaitDurable(lsn);
                                         std::lock_guard<std::mutex> lk(lsns_mu);
                                         lsns.push_back(lsn);
                                     } });
        }

        for (auto &th : workers)
            th.join();

        w.WaitDurable(w.WrittenLsn());
        w.Stop();

        std::unordered_set<Lsn> unique;
        for (Lsn lsn : lsns)
        {
            if (!unique.insert(lsn).second)
            {
                std::cerr << "Concurrency test FAILED: duplicate LSN " << lsn << "\n";
                ++failures;
                break;
            }
        }
        if (unique.size() != lsns.size())
            ++failures;
        else
            std::cout << "Concurrency test PASS\n";

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Concurrency test exception: " << e.what() << "\n";
        ++failures;
    }

    return failures == 0 ? 0 : 2;
}
