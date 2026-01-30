#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

#include <pomai/storage/wal.h>

using namespace pomai;
namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_wal_append.XXXXXX";
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
    std::cout << "WAL append regression test starting...\n";
    try
    {
        std::string dir = MakeTempDir();
        Wal w("shard-0", dir, 8);
        w.Start();

        std::vector<UpsertRequest> batch(4);
        for (std::size_t i = 0; i < batch.size(); ++i)
        {
            batch[i].id = i + 1;
            batch[i].vec.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        }

        Lsn lsn = w.AppendUpserts(batch, false);
        if (lsn == 0)
            throw std::runtime_error("AppendUpserts returned invalid LSN");

        w.Stop();
        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }

    std::cout << "WAL append regression test PASS\n";
    return 0;
}
