#include <chrono>
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
    std::string tmpl = "/tmp/pomai_repro.XXXXXX";
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
    std::cout << "Repro: WAL append + centroid training\n";
    const std::size_t dim = 32;
    const std::size_t train_vectors = 50'000;
    const std::size_t batch_size = 64;

    std::string dir = MakeTempDir();
    DbOptions opt;
    opt.dim = dim;
    opt.shards = 1;
    opt.shard_queue_capacity = 128;
    opt.wal_dir = dir;
    opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;

    PomaiDB db(opt);
    db.Start();

    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<UpsertRequest> batch;
    batch.reserve(batch_size);

    for (std::size_t i = 0; i < train_vectors; ++i)
    {
        UpsertRequest req;
        req.id = i;
        req.vec.data.resize(dim);
        for (std::size_t d = 0; d < dim; ++d)
            req.vec.data[d] = dist(rng);
        batch.push_back(std::move(req));

        if (batch.size() == batch_size || i + 1 == train_vectors)
        {
            db.UpsertBatch(batch, false).get();
            batch.clear();
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    db.RecomputeCentroids(32, train_vectors).get();

    db.Stop();
    RemoveDir(dir);
    std::cout << "Repro completed\n";
    return 0;
}
