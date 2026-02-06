#include "pomai/pomai.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct RunMetrics {
    double ingest_qps = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double p999_us = 0.0;
};

std::vector<float> MakeVec(std::uint64_t seed, std::uint32_t dim) {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        const std::uint64_t x = seed * 1315423911ULL + (i + 1) * 2654435761ULL;
        v[i] = static_cast<float>((x % 2001ULL) - 1000ULL) / 1000.0f;
    }
    return v;
}

double Percentile(std::vector<double> vals, double p) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    std::size_t idx = static_cast<std::size_t>(std::floor(p * static_cast<double>(vals.size() - 1)));
    return vals[idx];
}

double Median(std::vector<double> vals) {
    return Percentile(std::move(vals), 0.5);
}

RunMetrics OneRun(const std::string& db_path, std::uint32_t dim, std::uint32_t nvec, std::uint32_t nquery, std::uint32_t topk) {
    fs::remove_all(db_path);

    pomai::DBOptions opts;
    opts.path = db_path;
    opts.dim = dim;
    opts.shard_count = 1;
    opts.fsync = pomai::FsyncPolicy::kNever;

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opts, &db);
    if (!st.ok()) {
        std::cerr << "open failed: " << st.message() << "\n";
        std::exit(2);
    }

    const auto ingest_start = Clock::now();
    for (std::uint32_t i = 0; i < nvec; ++i) {
        auto v = MakeVec(i + 1, dim);
        st = db->Put(i + 1, v);
        if (!st.ok()) {
            std::cerr << "put failed: " << st.message() << "\n";
            std::exit(2);
        }
    }
    const auto ingest_end = Clock::now();

    std::vector<double> lat_us;
    lat_us.reserve(nquery);
    pomai::SearchResult out;

    for (std::uint32_t q = 0; q < nquery; ++q) {
        auto query = MakeVec(100000 + q, dim);
        const auto t0 = Clock::now();
        st = db->Search(query, topk, &out);
        const auto t1 = Clock::now();
        if (!st.ok()) {
            std::cerr << "search failed: " << st.message() << "\n";
            std::exit(2);
        }
        lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    const double ingest_sec = std::chrono::duration<double>(ingest_end - ingest_start).count();
    RunMetrics m;
    m.ingest_qps = ingest_sec > 0.0 ? static_cast<double>(nvec) / ingest_sec : 0.0;
    m.p50_us = Percentile(lat_us, 0.50);
    m.p95_us = Percentile(lat_us, 0.95);
    m.p99_us = Percentile(lat_us, 0.99);
    m.p999_us = Percentile(lat_us, 0.999);
    return m;
}

} // namespace

int main(int argc, char** argv) {
    std::string out_path = "";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--output" && i + 1 < argc) {
            out_path = argv[++i];
        }
    }

    constexpr std::uint32_t dim = 64;
    constexpr std::uint32_t nvec = 2000;
    constexpr std::uint32_t nquery = 300;
    constexpr std::uint32_t topk = 10;
    constexpr int kIters = 3;

    std::vector<double> ingest, p50, p95, p99, p999;
    ingest.reserve(kIters);
    p50.reserve(kIters);
    p95.reserve(kIters);
    p99.reserve(kIters);
    p999.reserve(kIters);

    for (int i = 0; i < kIters; ++i) {
        const auto m = OneRun("/tmp/pomai_ci_perf_" + std::to_string(i), dim, nvec, nquery, topk);
        ingest.push_back(m.ingest_qps);
        p50.push_back(m.p50_us);
        p95.push_back(m.p95_us);
        p99.push_back(m.p99_us);
        p999.push_back(m.p999_us);
    }

    const double med_ingest = Median(ingest);
    const double med_p50 = Median(p50);
    const double med_p95 = Median(p95);
    const double med_p99 = Median(p99);
    const double med_p999 = Median(p999);

    std::ostream* os = &std::cout;
    std::ofstream out;
    if (!out_path.empty()) {
        out.open(out_path);
        if (!out) {
            std::cerr << "failed to open output: " << out_path << "\n";
            return 2;
        }
        os = &out;
    }

    *os << "{\n";
    *os << "  \"config\": {\"dim\": " << dim << ", \"vectors\": " << nvec << ", \"queries\": " << nquery
        << ", \"topk\": " << topk << ", \"iterations\": " << kIters << "},\n";
    *os << "  \"metrics\": {\n";
    *os << "    \"ingest_qps\": " << med_ingest << ",\n";
    *os << "    \"search_latency_us\": {\"p50\": " << med_p50 << ", \"p95\": " << med_p95
        << ", \"p99\": " << med_p99 << ", \"p999\": " << med_p999 << "}\n";
    *os << "  }\n";
    *os << "}\n";

    return 0;
}
