#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <unistd.h>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_wal_durable.XXXXXX";
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

        auto b1 = MakeBatch(4, 4, 1);
        auto b2 = MakeBatch(4, 3, 100);
        Lsn l1 = w.AppendUpserts(b1, false);
        Lsn l2 = w.AppendUpserts(b2, true);
        w.WaitDurable(l1);
        w.WaitDurable(l2);
        w.Stop();

        Seed seed(4);
        WalReplayStats stats = w.ReplayToSeed(seed);
        if (stats.last_lsn < l2)
        {
            std::cerr << "Durable test FAILED: last_lsn=" << stats.last_lsn << " expected >= " << l2 << "\n";
            ++failures;
        }
        else
        {
            std::cout << "Durable test PASS\n";
        }
        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Durable test exception: " << e.what() << "\n";
        ++failures;
    }

    return failures == 0 ? 0 : 2;
}
