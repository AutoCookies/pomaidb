#include "pomai_db.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

using namespace pomai;

static inline float L2Sqr(const std::vector<float> &a, const std::vector<float> &b)
{
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

static std::vector<Id> TrueTopK(const std::vector<std::pair<std::vector<float>, Id>> &data,
                                const std::vector<float> &q,
                                std::size_t k)
{
    std::vector<std::pair<float, Id>> dist;
    dist.reserve(data.size());
    for (const auto &row : data)
        dist.emplace_back(L2Sqr(row.first, q), row.second);

    if (k < dist.size())
    {
        std::nth_element(dist.begin(), dist.begin() + k, dist.end(),
                         [](const auto &a, const auto &b)
                         { return a.first < b.first; });
        dist.resize(k);
    }

    std::sort(dist.begin(), dist.end(),
              [](const auto &a, const auto &b)
              { return a.first < b.first; });

    std::vector<Id> out;
    out.reserve(dist.size());
    for (const auto &p : dist)
        out.push_back(p.second);
    return out;
}

struct ScenarioResult
{
    double avg_us{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double recall10{0.0};
};

static ScenarioResult RunScenario(const std::string &label,
                                  const pomai::server::WhisperConfig &whisper,
                                  std::uint64_t rng_seed,
                                  std::size_t base_probe)
{
    const std::size_t N = 200'000;
    const std::size_t dim = 128;
    const std::size_t shards = 4;
    const std::size_t chunk_size = 2'000;
    const std::size_t reservoir_size = 5'000;
    const std::size_t queries = 120;
    const std::size_t topk = 10;

    DbOptions opt;
    opt.dim = dim;
    opt.metric = Metric::L2;
    opt.shards = shards;
    opt.shard_queue_capacity = 4096;
    opt.wal_dir = "./data/bench_phase3";
    opt.centroids_load_mode = MembraneRouter::CentroidsLoadMode::None;
    opt.whisper = whisper;

    std::filesystem::remove_all(opt.wal_dir);
    std::filesystem::create_directories(opt.wal_dir);

    PomaiDB db(opt);
    db.Start();
    db.SetProbeCount(base_probe);

    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);

    std::vector<std::pair<std::vector<float>, Id>> reservoir;
    reservoir.reserve(reservoir_size);

    std::uint64_t seen = 0;
    std::vector<UpsertRequest> chunk;
    chunk.reserve(chunk_size);

    for (std::size_t base = 0; base < N; base += chunk_size)
    {
        const std::size_t this_chunk = std::min(chunk_size, N - base);
        chunk.clear();
        chunk.resize(this_chunk);

        for (std::size_t i = 0; i < this_chunk; ++i)
        {
            const std::size_t global_idx = base + i;
            chunk[i].id = global_idx;
            chunk[i].vec.data.resize(dim);
            for (std::size_t d = 0; d < dim; ++d)
                chunk[i].vec.data[d] = val(rng);

            if (seen < reservoir_size)
            {
                reservoir.emplace_back(chunk[i].vec.data, static_cast<Id>(global_idx));
            }
            else
            {
                std::uniform_int_distribution<std::uint64_t> dist(0, seen);
                std::uint64_t j = dist(rng);
                if (j < reservoir_size)
                    reservoir[static_cast<std::size_t>(j)] = std::make_pair(chunk[i].vec.data, static_cast<Id>(global_idx));
            }
            ++seen;
        }

        db.UpsertBatch(std::move(chunk), false).get();
        chunk.clear();
    }

    const std::size_t centroids_k = shards * 8;
    db.RecomputeCentroids(centroids_k, shards * 512).get();

    std::uniform_int_distribution<std::size_t> qp(0, reservoir.size() - 1);
    for (std::size_t warm = 0; warm < 20; ++warm)
    {
        auto &qvec = reservoir[qp(rng)].first;
        SearchRequest req;
        req.topk = topk;
        req.query.data = qvec;
        db.Search(req);
    }

    std::vector<double> lat_us;
    lat_us.reserve(queries);
    double recall10_acc = 0.0;

    for (std::size_t qi = 0; qi < queries; ++qi)
    {
        auto &qvec = reservoir[qp(rng)].first;
        auto truek = TrueTopK(reservoir, qvec, topk);

        SearchRequest req;
        req.topk = topk;
        req.query.data = qvec;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto resp = db.Search(req);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> d = t1 - t0;
        lat_us.push_back(d.count());

        std::size_t hits = 0;
        for (auto tid : truek)
        {
            for (const auto &it : resp.items)
            {
                if (it.id == tid)
                {
                    ++hits;
                    break;
                }
            }
        }
        recall10_acc += static_cast<double>(hits) / static_cast<double>(topk);
    }

    std::sort(lat_us.begin(), lat_us.end());
    ScenarioResult res;
    res.avg_us = std::accumulate(lat_us.begin(), lat_us.end(), 0.0) / lat_us.size();
    res.p50_us = lat_us[lat_us.size() * 50 / 100];
    res.p95_us = lat_us[lat_us.size() * 95 / 100];
    res.recall10 = recall10_acc / static_cast<double>(queries);

    std::cout << "=== Scenario: " << label << " ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency avg (us): " << res.avg_us << "\n";
    std::cout << "Latency p50 (us): " << res.p50_us << "\n";
    std::cout << "Latency p95 (us): " << res.p95_us << "\n";
    std::cout << "Recall@10: " << (res.recall10 * 100.0) << " %\n";
    std::cout << "Stats: " << db.GetStats() << "\n";

    db.Stop();
    return res;
}

int main()
{
    std::cout << ":: POMAI PHASE 3 BENCH (ADAPTIVE NPROBE) ::\n";

    pomai::server::WhisperConfig healthy_cfg;
    healthy_cfg.latency_target_ms = 50.0f;

    pomai::server::WhisperConfig tight_cfg;
    tight_cfg.latency_target_ms = 1.0f;

    const std::size_t base_probe = 4;

    RunScenario("Healthy budget (higher recall)", healthy_cfg, 42, base_probe);
    RunScenario("Tight budget (lower latency)", tight_cfg, 42, base_probe);

    return 0;
}
