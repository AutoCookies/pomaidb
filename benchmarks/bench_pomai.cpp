#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "pomai/core/pomai.h" // PomaiDB + DbOptions
#include "pomai/types.h"
#include "pomai/status.h"

using clock_type = std::chrono::steady_clock;

static std::uint64_t NowNs()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now().time_since_epoch()).count());
}

static std::vector<float> MakeRandomVec(std::mt19937_64 &rng, std::uint32_t dim)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
        v[i] = dist(rng);
    return v;
}

static std::string MakePayload(std::size_t bytes)
{
    if (bytes == 0)
        return {};
    std::string s(bytes, 'x');
    // add tiny entropy so compression doesn’t “cheat”
    for (std::size_t i = 0; i < bytes; i += 97)
        s[i] = static_cast<char>('a' + (i % 26));
    return s;
}

struct LatStats
{
    double p50_ms{0}, p95_ms{0}, p99_ms{0}, avg_ms{0};
};

static LatStats ComputeLatStats(std::vector<std::uint64_t> &lat_ns)
{
    LatStats out{};
    if (lat_ns.empty())
        return out;
    std::sort(lat_ns.begin(), lat_ns.end());
    auto pick = [&](double q) -> double
    {
        std::size_t idx = static_cast<std::size_t>(q * (lat_ns.size() - 1));
        return static_cast<double>(lat_ns[idx]) / 1e6;
    };
    out.p50_ms = pick(0.50);
    out.p95_ms = pick(0.95);
    out.p99_ms = pick(0.99);

    long double sum = 0;
    for (auto x : lat_ns)
        sum += static_cast<long double>(x);
    out.avg_ms = static_cast<double>(sum / lat_ns.size()) / 1e6;
    return out;
}

static void PrintUsage(const char *argv0)
{
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "Options:\n"
        << "  --dir <path>           data dir (default ./pomai_bench_data)\n"
        << "  --n <int>              number of vectors (default 200000)\n"
        << "  --dim <int>            vector dim (default 512)\n"
        << "  --batch <int>          upsert batch size (default 256)\n"
        << "  --payload <int>        payload bytes per vector (default 0)\n"
        << "  --shards <int>         shard count (default 4)\n"
        << "  --queries <int>        number of queries (default 2000)\n"
        << "  --topk <int>           topK (default 10)\n"
        << "  --warmup <int>         warmup queries (default 200)\n"
        << "  --fsync never|every    WAL fsync policy (default never)\n";
}

int main(int argc, char **argv)
{
    std::filesystem::path dir = "./pomai_bench_data";
    std::uint64_t n = 200000;
    std::uint32_t dim = 512;
    std::uint32_t batch = 256;
    std::size_t payload_bytes = 0;
    std::uint32_t shards = 4;
    std::uint32_t queries = 2000;
    std::uint32_t warmup = 200;
    std::uint32_t topk = 10;
    std::string fsync = "never";

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need = [&](const char *opt)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << opt << "\n";
                std::exit(2);
            }
        };
        if (a == "--dir")
        {
            need("--dir");
            dir = argv[++i];
        }
        else if (a == "--n")
        {
            need("--n");
            n = std::stoull(argv[++i]);
        }
        else if (a == "--dim")
        {
            need("--dim");
            dim = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--batch")
        {
            need("--batch");
            batch = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--payload")
        {
            need("--payload");
            payload_bytes = static_cast<std::size_t>(std::stoull(argv[++i]));
        }
        else if (a == "--shards")
        {
            need("--shards");
            shards = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--queries")
        {
            need("--queries");
            queries = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--warmup")
        {
            need("--warmup");
            warmup = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--topk")
        {
            need("--topk");
            topk = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--fsync")
        {
            need("--fsync");
            fsync = argv[++i];
        }
        else if (a == "--help" || a == "-h")
        {
            PrintUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown arg: " << a << "\n";
            PrintUsage(argv[0]);
            return 2;
        }
    }

    // reset dir
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    pomai::DbOptions opt;
    opt.dir = dir;
    opt.num_shards = shards;
    opt.vector_dim = dim;

    if (fsync == "never")
        opt.fsync_policy = pomai::core::FsyncPolicy::Never;
    else if (fsync == "every")
        opt.fsync_policy = pomai::core::FsyncPolicy::EveryWrite;
    else
    {
        std::cerr << "Invalid --fsync. Use never|every\n";
        return 2;
    }

    pomai::PomaiDB db(opt);
    auto st = db.Start();
    if (!st.ok())
    {
        std::cerr << "DB start failed: " << st.message << "\n";
        return 1;
    }

    std::mt19937_64 rng(1234567);
    std::string payload = MakePayload(payload_bytes);

    // Pre-generate vectors (so benchmark measures DB cost, not RNG cost)
    std::vector<std::vector<float>> vecs;
    vecs.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i)
    {
        vecs.push_back(MakeRandomVec(rng, dim));
    }

    // --------------------
    // Ingest benchmark
    // --------------------
    std::cout << "== PomaiDB BENCH ==\n";
    std::cout << "n=" << n << " dim=" << dim << " shards=" << shards
              << " batch=" << batch << " payload_bytes=" << payload_bytes
              << " fsync=" << fsync << "\n";

    std::uint64_t ing_start = NowNs();
    std::uint64_t ok = 0, fail = 0;

    std::vector<pomai::UpsertItem> items;
    items.reserve(batch);

    for (std::uint64_t i = 0; i < n;)
    {
        items.clear();
        std::uint32_t take = static_cast<std::uint32_t>(std::min<std::uint64_t>(batch, n - i));
        for (std::uint32_t j = 0; j < take; ++j)
        {
            pomai::UpsertItem it;
            it.id = static_cast<pomai::VectorId>(i + j);
            it.vec.values = vecs[static_cast<std::size_t>(i + j)];
            it.payload = payload;
            items.push_back(std::move(it));
        }

        auto r = db.UpsertBatch(std::move(items));
        if (!r.status.ok())
        {
            fail += take;
        }
        else
        {
            ok += r.ok_count;
            fail += r.fail_count;
        }

        i += take;
        items = {}; // ensure moved-from vector doesn't keep capacity too huge
        items.reserve(batch);
    }

    auto flush_st = db.Flush(); // make ingest durable-ish for fsync=never too (OS flush)
    if (!flush_st.ok())
    {
        std::cerr << "Flush failed: " << flush_st.message << "\n";
    }

    std::uint64_t ing_end = NowNs();
    double ing_s = (ing_end - ing_start) / 1e9;
    double vps = (ing_s > 0) ? (static_cast<double>(ok) / ing_s) : 0;

    std::cout << "[Ingest] ok=" << ok << " fail=" << fail
              << " time=" << ing_s << "s"
              << " throughput=" << vps << " vec/s\n";

    // --------------------
    // Search benchmark
    // --------------------
    // Use queries sampled from inserted vectors (realistic “self-query”)
    std::uniform_int_distribution<std::uint64_t> pick_id(0, n - 1);

    // warmup
    for (std::uint32_t i = 0; i < warmup; ++i)
    {
        auto id = pick_id(rng);
        pomai::VectorData q;
        q.values = vecs[static_cast<std::size_t>(id)];
        auto rr = db.Search(std::move(q), topk);
        (void)rr;
    }

    std::vector<std::uint64_t> lat_ns;
    lat_ns.reserve(queries);

    std::uint64_t q_start = NowNs();

    std::uint64_t q_ok = 0, q_fail = 0;
    for (std::uint32_t i = 0; i < queries; ++i)
    {
        auto id = pick_id(rng);

        pomai::VectorData q;
        q.values = vecs[static_cast<std::size_t>(id)];

        std::uint64_t t0 = NowNs();
        auto rr = db.Search(std::move(q), topk);
        std::uint64_t t1 = NowNs();

        if (!rr.status.ok())
        {
            q_fail++;
        }
        else
        {
            q_ok++;
            lat_ns.push_back(t1 - t0);
        }
    }

    std::uint64_t q_end = NowNs();
    double q_s = (q_end - q_start) / 1e9;
    double qps = (q_s > 0) ? (static_cast<double>(queries) / q_s) : 0;

    auto stats = ComputeLatStats(lat_ns);

    std::cout << "[Search] queries=" << queries
              << " ok=" << q_ok << " fail=" << q_fail
              << " wall=" << q_s << "s"
              << " QPS=" << qps << "\n";
    std::cout << "        latency_ms: p50=" << stats.p50_ms
              << " p95=" << stats.p95_ms
              << " p99=" << stats.p99_ms
              << " avg=" << stats.avg_ms << "\n";

    // Optional: dump shard stats
    auto ss = db.Stats();
    if (ss.status.ok())
    {
        std::cout << "[ShardStats]\n";
        for (auto &s : ss.value)
        {
            std::cout << "  shard=" << s.shard_id
                      << " queue=" << s.queue_depth
                      << " wal_bytes=" << s.wal_bytes
                      << " ms_since_ckpt=" << s.ms_since_last_checkpoint
                      << " upsert_p50us=" << s.upsert_latency_us.p50
                      << " upsert_p99us=" << s.upsert_latency_us.p99
                      << " search_p50us=" << s.search_latency_us.p50
                      << " search_p99us=" << s.search_latency_us.p99
                      << "\n";
        }
    }

    db.Stop();
    return 0;
}
