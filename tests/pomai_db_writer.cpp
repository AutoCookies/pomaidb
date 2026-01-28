#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <filesystem>
#include <cerrno>

#include "pomai_db.h"

using namespace pomai;
namespace fs = std::filesystem;

// Usage:
//   pomai_db_writer <wal_dir> <allow_sync_on_append (0|1)> <num_batches> <batch_size> <dim>
// Example:
//   ./pomai_db_writer /tmp/testdir 1 20 10 8
int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cerr << "usage: " << argv[0] << " <wal_dir> <allow_sync_on_append(0|1)> <num_batches> <batch_size> <dim>\n";
        return 2;
    }

    std::string wal_dir = argv[1];
    bool allow_sync = std::atoi(argv[2]) != 0;
    int num_batches = std::atoi(argv[3]);
    int batch_size = std::atoi(argv[4]);
    int dim = std::atoi(argv[5]);

    if (num_batches <= 0 || batch_size <= 0 || dim <= 0)
    {
        std::cerr << "invalid numeric args\n";
        return 2;
    }

    // Ensure wal_dir exists
    try
    {
        fs::create_directories(wal_dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to create wal_dir '" << wal_dir << "': " << e.what() << "\n";
        return 3;
    }

    std::cout << "Writer start: wal_dir=" << wal_dir
              << " allow_sync=" << (allow_sync ? 1 : 0)
              << " batches=" << num_batches
              << " batch_size=" << batch_size
              << " dim=" << dim << std::endl;

    // Prepare DbOptions
    DbOptions opt;
    opt.dim = static_cast<std::size_t>(dim);
    opt.metric = Metric::L2;
    opt.shards = 1;
    opt.shard_queue_capacity = 65536;
    opt.wal_dir = wal_dir;
    opt.allow_sync_on_append = allow_sync;

    // Simple logging callbacks
    PomaiDB::LogFn info = [](const std::string &m)
    { std::cout << "[DB INFO] " << m << std::endl; };
    PomaiDB::LogFn err = [](const std::string &m)
    { std::cerr << "[DB ERR] " << m << std::endl; };

    // Construct and start DB
    std::unique_ptr<PomaiDB> db;
    try
    {
        db = std::make_unique<PomaiDB>(opt, info, err);
        db->Start();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to start PomaiDB: " << e.what() << "\n";
        return 4;
    }

    // Status file for parent coordination
    const std::string status_path = wal_dir + "/writer.status";

    // Truncate/create status file
    {
        std::ofstream s(status_path, std::ios::binary | std::ios::trunc);
        if (!s)
        {
            std::cerr << "Failed to create status file: " << status_path << "\n";
            // proceed anyway; parent may timeout
        }
    }

    auto append_status = [&](int batch_idx)
    {
        int fd = ::open(status_path.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC);
        if (fd < 0)
        {
            std::cerr << "open status file failed: " << std::strerror(errno) << "\n";
            return;
        }
        std::string line = "batch " + std::to_string(batch_idx) + "\n";
        ssize_t w = ::write(fd, line.data(), (ssize_t)line.size());
        if (w < 0)
        {
            std::cerr << "write status failed: " << std::strerror(errno) << "\n";
            ::close(fd);
            return;
        }
        if (::fdatasync(fd) != 0)
        {
            std::cerr << "fdatasync status failed: " << std::strerror(errno) << "\n";
        }
        ::close(fd);
    };

    // Produce batches
    for (int b = 1; b <= num_batches; ++b)
    {
        std::vector<UpsertRequest> batch;
        batch.reserve(static_cast<std::size_t>(batch_size));
        for (int i = 0; i < batch_size; ++i)
        {
            UpsertRequest r;
            r.id = static_cast<Id>((uint64_t)b * 100000 + (uint64_t)i + 1);
            r.vec.data.resize(static_cast<std::size_t>(dim));
            for (int j = 0; j < dim; ++j)
                r.vec.data[j] = float(r.id + j);
            batch.push_back(std::move(r));
        }

        try
        {
            // Request synchronous durability from client side; PomaiDB enforces allow_sync_on_append.
            auto fut = db->UpsertBatch(std::move(batch), /*client_wait_durable=*/true);
            Lsn lsn = fut.get(); // wait for completion (may be synchronous or not depending on policy)
            std::cout << "WROTE batch=" << b << " lsn=" << lsn << " allow_sync=" << (allow_sync ? 1 : 0) << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "UpsertBatch error: " << e.what() << "\n";
        }

        // Append status marker (fsynced) so parent can deterministically kill after seeing it
        append_status(b);

        // Short sleep to allow deterministic kill timing for test harness
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Normal shutdown
    db->Stop();
    std::cout << "Writer finished normally\n";
    return 0;
}