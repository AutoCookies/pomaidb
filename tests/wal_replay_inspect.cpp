#include <iostream>
#include <string>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

using namespace pomai;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <wal_dir> <dim>\n";
        return 2;
    }
    std::string wal_dir = argv[1];
    int dim = std::atoi(argv[2]);

    try
    {
        Wal w("shard-0", wal_dir, (size_t)dim);
        Seed seed(dim);
        WalReplayStats s = w.ReplayToSeed(seed);
        std::cout << "ReplayStats: records_applied=" << s.records_applied
                  << " vectors_applied=" << s.vectors_applied
                  << " last_lsn=" << s.last_lsn
                  << " truncated_bytes=" << s.truncated_bytes << "\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Replay inspect failed: " << e.what() << "\n";
        return 3;
    }
}