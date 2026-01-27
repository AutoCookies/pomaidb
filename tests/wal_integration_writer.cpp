#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <string>

#include "wal.h"

using namespace pomai;

// Simple writer that appends num_batches each of batch_size vectors to shard-0 in wal_dir.
// Usage: wal_integration_writer <wal_dir> <num_batches> <batch_size> <wait_durable>
// wait_durable: 0 or 1
int main(int argc, char **argv)
{
    if (argc < 5)
    {
        std::cerr << "usage: " << argv[0] << " <wal_dir> <num_batches> <batch_size> <wait_durable>\n";
        return 2;
    }
    std::string wal_dir = argv[1];
    int num_batches = std::atoi(argv[2]);
    int batch_size = std::atoi(argv[3]);
    int wait_durable = std::atoi(argv[4]);

    Wal w("shard-0", wal_dir, 8); // use dim=8 for integration
    w.Start();

    for (int b = 0; b < num_batches; ++b)
    {
        std::vector<UpsertRequest> batch;
        batch.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i)
        {
            UpsertRequest r;
            r.id = (uint64_t)b * batch_size + i + 1;
            r.vec.data.resize(8);
            for (size_t j = 0; j < 8; ++j)
                r.vec.data[j] = float(r.id + j);
            batch.push_back(std::move(r));
        }

        Lsn l = w.AppendUpserts(batch, wait_durable != 0);
        // For visibility for parent process, print progress to stdout
        std::cout << "WROTE batch " << b << " lsn=" << l << " wait_durable=" << wait_durable << "\n" << std::flush;

        // Sleep a bit to allow kill-timing in integration tests
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Clean stop
    w.Stop();
    std::cout << "Writer finished\n";
    return 0;
}