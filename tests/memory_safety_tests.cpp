#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pomai/util/memory_manager.h>
#include <pomai/api/pomai_db.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_mem.XXXXXX";
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
    std::cout << "Memory safety tests starting...\n";
    int failures = 0;

    try
    {
        auto &mm = MemoryManager::Instance();
        mm.ResetUsageForTesting();
        mm.SetTotalMemoryBytesForTesting(100);
        mm.AddUsage(MemoryManager::Pool::Memtable, 95);

        std::string dir = MakeTempDir();
        DbOptions opt;
        opt.dim = 2;
        opt.shards = 1;
        opt.shard_queue_capacity = 4;
        opt.wal_dir = dir;

        PomaiDB db(opt);

        std::vector<UpsertRequest> batch;
        UpsertRequest req;
        req.id = 1;
        req.vec.data = {1.0f, 2.0f};
        batch.push_back(req);

        bool threw = false;
        try
        {
            auto fut = db.UpsertBatch(std::move(batch), false);
            (void)fut.get();
        }
        catch (const std::runtime_error &e)
        {
            threw = true;
            const std::string expected = "UpsertBatch rejected: memory pressure";
            if (expected != e.what())
            {
                std::cerr << "Test FAILED: unexpected message: " << e.what() << "\n";
                ++failures;
            }
        }

        if (!threw)
        {
            std::cerr << "Test FAILED: expected memory rejection\n";
            ++failures;
        }

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test exception: " << e.what() << "\n";
        ++failures;
    }

    if (failures == 0)
    {
        std::cout << "All Memory safety tests PASS\n";
        return 0;
    }

    std::cerr << failures << " Memory safety tests FAILED\n";
    return 1;
}
