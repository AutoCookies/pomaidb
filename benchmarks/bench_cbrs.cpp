#include "pomai/pomai.h"
#include "core/distance.h"
#include "core/routing/kmeans_lite.h"
#include "core/routing/routing_persist.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct CliConfig {
    std::string path = "/tmp/pomai_bench_cbrs";
    uint64_t seed = 1337;
    uint32_t shards = 4;
    uint32_t dim = 128;
    uint32_t n = 10000;
    uint32_t queries = 1000;
    uint32_t topk = 10;
    std::string dataset = "uniform";
    uint32_t clusters = 8;
    std::string routing = "cbrs";
    uint32_t probe = 0;
    uint32_t k_global = 0;
    std::string fsync = "never";
    uint32_t threads = 1;
    std::string report_json;
    std::string report_csv;
    std::string matrix;
};

struct ScenarioConfig {
    std::string name;
    CliConfig cfg;
    bool epoch_drift = false;
};

struct ResourceSample {
    long rss_kb = 0;
    long peak_rss_kb = 0;
};

struct Row {
    std::string scenario;
    std::string routing;
    std::string dataset;
    uint32_t dim = 0;
    uint32_t n = 0;
    uint32_t queries = 0;
    uint32_t topk = 0;
    uint32_t shards = 0;
    double ingest_sec = 0.0;
    double ingest_qps = 0.0;
    double query_qps = 0.0;
    double p50_us = 0.0, p90_us = 0.0, p95_us = 0.0, p99_us = 0.0, p999_us = 0.0;
    double recall1 = 0.0, recall10 = 0.0, recall100 = 0.0;
    double routed_shards_avg = 0.0, routed_shards_p95 = 0.0;
    double routed_probe_avg = 0.0, routed_probe_p95 = 0.0;
    double routed_buckets_avg = -1.0, routed_buckets_p95 = -1.0;
    long rss_open_kb = 0, rss_ingest_kb = 0, rss_query_kb = 0, peak_rss_kb = 0;
    double user_cpu_sec = 0.0, sys_cpu_sec = 0.0;
    std::string verdict;
    bool success = true;
    std::string error;
};

long ReadVmValueKB(const char* key) {
    std::ifstream in("/proc/self/status");
    std::string label;
    long value = 0;
    std::string unit;
    while (in >> label >> value >> unit) {
        if (label == key) return value;
    }
    return 0;
}

ResourceSample ReadResources() {
    ResourceSample r;
    r.rss_kb = ReadVmValueKB("VmRSS:");
    r.peak_rss_kb = ReadVmValueKB("VmHWM:");
    return r;
}

double ToSec(const timeval& tv) {
    return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1e6;
}

double Percentile(const std::vector<double>& vals, double p) {
    if (vals.empty()) return 0.0;
    std::vector<double> s = vals;
    std::sort(s.begin(), s.end());
    const size_t idx = static_cast<size_t>(std::floor(p * static_cast<double>(s.size() - 1)));
    return s[idx];
}

double RecallAtK(const std::vector<pomai::SearchHit>& approx,
                 const std::vector<pomai::SearchHit>& exact,
                 size_t k) {
    if (k == 0) return 1.0;
    if (exact.empty()) return 1.0;
    const size_t kk = std::min(k, exact.size());
    std::unordered_set<pomai::VectorId> gt;
    gt.reserve(kk * 2);
    for (size_t i = 0; i < kk; ++i) gt.insert(exact[i].id);
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, approx.size()); ++i) {
        if (gt.count(approx[i].id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(kk);
}

void Normalize(std::vector<float>& v) {
    float n2 = 0.0f;
    for (float x : v) n2 += x * x;
    const float n = std::sqrt(std::max(1e-12f, n2));
    for (float& x : v) x /= n;
}

struct Dataset {
    std::vector<std::vector<float>> base;
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<float>> drift_half; // used by epoch scenario
};

Dataset GenerateDataset(const CliConfig& cfg, bool with_drift = false) {
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    Dataset d;
    d.base.assign(cfg.n, std::vector<float>(cfg.dim));
    d.queries.assign(cfg.queries, std::vector<float>(cfg.dim));

    std::vector<std::vector<float>> centers(cfg.clusters, std::vector<float>(cfg.dim));
    for (auto& c : centers) {
        for (float& x : c) x = g(rng);
        Normalize(c);
    }

    auto sample_clustered = [&](std::vector<float>& out, uint32_t cid, float sigma) {
        for (uint32_t j = 0; j < cfg.dim; ++j) out[j] = centers[cid][j] + sigma * g(rng);
        Normalize(out);
    };

    for (uint32_t i = 0; i < cfg.n; ++i) {
        auto& v = d.base[i];
        if (cfg.dataset == "uniform") {
            for (float& x : v) x = u(rng);
            Normalize(v);
        } else if (cfg.dataset == "clustered") {
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered(v, c, 0.08f);
        } else if (cfg.dataset == "overlap") {
            const float shift = 0.03f;
            for (uint32_t c = 1; c < cfg.clusters; ++c) {
                for (uint32_t j = 0; j < cfg.dim; ++j) centers[c][j] = centers[0][j] + shift * g(rng);
                Normalize(centers[c]);
            }
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered(v, c, 0.12f);
        } else if (cfg.dataset == "skew") {
            const bool hot = (rng() % 10) != 0;
            uint32_t c = hot ? 0u : static_cast<uint32_t>(1 + (rng() % std::max(1u, cfg.clusters - 1)));
            sample_clustered(v, c, hot ? 0.05f : 0.15f);
        }
    }

    for (uint32_t i = 0; i < cfg.queries; ++i) {
        auto& q = d.queries[i];
        if (cfg.dataset == "uniform") {
            for (float& x : q) x = u(rng);
            Normalize(q);
        } else {
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered(q, c, cfg.dataset == "overlap" ? 0.12f : 0.08f);
        }
    }

    if (with_drift) {
        d.drift_half.assign(cfg.n / 2, std::vector<float>(cfg.dim));
        std::vector<std::vector<float>> drift_centers = centers;
        for (auto& c : drift_centers) {
            for (float& x : c) x += 0.35f * g(rng);
            Normalize(c);
        }
        for (auto& v : d.drift_half) {
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            for (uint32_t j = 0; j < cfg.dim; ++j) v[j] = drift_centers[c][j] + 0.08f * g(rng);
            Normalize(v);
        }
    }

    return d;
}

std::vector<pomai::SearchHit> BruteForceTopK(const std::vector<std::vector<float>>& all,
                                             std::span<const float> query,
                                             uint32_t topk,
                                             pomai::VectorId id_offset = 0) {
    std::vector<pomai::SearchHit> hits;
    hits.reserve(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
        hits.push_back({id_offset + i, pomai::core::Dot(query, all[i])});
    }
    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
    });
    if (hits.size() > topk) hits.resize(topk);
    return hits;
}

pomai::FsyncPolicy ParseFsync(const std::string& f) {
    if (f == "always") return pomai::FsyncPolicy::kAlways;
    return pomai::FsyncPolicy::kNever;
}

void WriteCsv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    out << "scenario,routing,dataset,dim,n,queries,topk,shards,ingest_sec,ingest_qps,query_qps,p50_us,p90_us,p95_us,p99_us,p999_us,recall1,recall10,recall100,routed_shards_avg,routed_shards_p95,routed_probe_avg,routed_probe_p95,routed_buckets_avg,routed_buckets_p95,rss_open_kb,rss_ingest_kb,rss_query_kb,peak_rss_kb,user_cpu_sec,sys_cpu_sec,verdict,error\n";
    for (const auto& r : rows) {
        out << r.scenario << ',' << r.routing << ',' << r.dataset << ',' << r.dim << ',' << r.n << ',' << r.queries
            << ',' << r.topk << ',' << r.shards << ',' << r.ingest_sec << ',' << r.ingest_qps << ',' << r.query_qps
            << ',' << r.p50_us << ',' << r.p90_us << ',' << r.p95_us << ',' << r.p99_us << ',' << r.p999_us
            << ',' << r.recall1 << ',' << r.recall10 << ',' << r.recall100
            << ',' << r.routed_shards_avg << ',' << r.routed_shards_p95
            << ',' << r.routed_probe_avg << ',' << r.routed_probe_p95
            << ',' << r.routed_buckets_avg << ',' << r.routed_buckets_p95
            << ',' << r.rss_open_kb << ',' << r.rss_ingest_kb << ',' << r.rss_query_kb << ',' << r.peak_rss_kb
            << ',' << r.user_cpu_sec << ',' << r.sys_cpu_sec << ',' << r.verdict << ',' << '"' << r.error << '"' << "\n";
    }
}

void WriteJson(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    out << "{\n  \"bench\": \"bench_cbrs\",\n  \"rows\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        out << "    {\n"
            << "      \"scenario\": \"" << r.scenario << "\",\n"
            << "      \"routing\": \"" << r.routing << "\",\n"
            << "      \"dataset\": \"" << r.dataset << "\",\n"
            << "      \"dim\": " << r.dim << ",\n"
            << "      \"n\": " << r.n << ",\n"
            << "      \"queries\": " << r.queries << ",\n"
            << "      \"topk\": " << r.topk << ",\n"
            << "      \"shards\": " << r.shards << ",\n"
            << "      \"ingest_sec\": " << r.ingest_sec << ",\n"
            << "      \"ingest_qps\": " << r.ingest_qps << ",\n"
            << "      \"query_qps\": " << r.query_qps << ",\n"
            << "      \"latency_us\": {\"p50\": " << r.p50_us << ", \"p90\": " << r.p90_us << ", \"p95\": " << r.p95_us
            << ", \"p99\": " << r.p99_us << ", \"p999\": " << r.p999_us << "},\n"
            << "      \"recall\": {\"r1\": " << r.recall1 << ", \"r10\": " << r.recall10 << ", \"r100\": " << r.recall100 << "},\n"
            << "      \"routed\": {\"shards_avg\": " << r.routed_shards_avg << ", \"shards_p95\": " << r.routed_shards_p95
            << ", \"probe_avg\": " << r.routed_probe_avg << ", \"probe_p95\": " << r.routed_probe_p95
            << ", \"buckets_avg\": " << r.routed_buckets_avg << ", \"buckets_p95\": " << r.routed_buckets_p95 << "},\n"
            << "      \"memory_kb\": {\"open\": " << r.rss_open_kb << ", \"ingest\": " << r.rss_ingest_kb << ", \"query\": " << r.rss_query_kb
            << ", \"peak\": " << r.peak_rss_kb << "},\n"
            << "      \"cpu_sec\": {\"user\": " << r.user_cpu_sec << ", \"sys\": " << r.sys_cpu_sec << "},\n"
            << "      \"verdict\": \"" << r.verdict << "\",\n"
            << "      \"error\": \"" << r.error << "\"\n"
            << "    }" << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
}

Row RunScenario(const ScenarioConfig& sc) {
    const auto& cfg = sc.cfg;
    Row row;
    row.scenario = sc.name;
    row.routing = cfg.routing;
    row.dataset = cfg.dataset;
    row.dim = cfg.dim;
    row.n = cfg.n;
    row.queries = cfg.queries;
    row.topk = cfg.topk;
    row.shards = cfg.shards;

    fs::remove_all(cfg.path);
    fs::create_directories(cfg.path);

    auto data = GenerateDataset(cfg, sc.epoch_drift);

    pomai::DBOptions opt;
    opt.path = cfg.path;
    opt.dim = cfg.dim;
    opt.shard_count = cfg.shards;
    opt.fsync = ParseFsync(cfg.fsync);
    opt.routing_enabled = cfg.routing != "fanout";
    opt.routing_k = cfg.k_global;
    opt.routing_probe = cfg.probe;
    opt.routing_warmup_mult = 1;
    opt.routing_keep_prev = cfg.routing == "cbrs_no_dual" ? 0u : 1u;

    struct rusage ru0{};
    getrusage(RUSAGE_SELF, &ru0);

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        row.success = false;
        row.error = st.message();
        row.verdict = "FAIL";
        return row;
    }
    auto ropen = ReadResources();
    row.rss_open_kb = ropen.rss_kb;

    auto t0 = Clock::now();
    for (uint32_t i = 0; i < cfg.n; ++i) {
        st = db->Put(i, data.base[i]);
        if (!st.ok()) {
            row.success = false;
            row.error = st.message();
            row.verdict = "FAIL";
            return row;
        }
    }
    if (sc.epoch_drift) {
        db->Close();
        // Force publish new epoch with optional prev retention.
        const uint32_t rk = std::max(1u, cfg.k_global == 0 ? 2u * cfg.shards : cfg.k_global);
        std::vector<float> drift_flat;
        drift_flat.reserve(static_cast<size_t>(data.drift_half.size()) * cfg.dim);
        for (const auto& v : data.drift_half) {
            for (float x : v) drift_flat.push_back(-x);
        }
        auto tab = pomai::core::routing::BuildInitialTable(std::span<const float>(drift_flat.data(), drift_flat.size()),
                                                     static_cast<uint32_t>(data.drift_half.size()), cfg.dim,
                                                     rk, cfg.shards, 5, static_cast<uint32_t>(cfg.seed));
        (void)pomai::core::routing::SaveRoutingTableAtomic(cfg.path + "/membranes/default", tab, cfg.routing != "cbrs_no_dual");

        st = pomai::DB::Open(opt, &db);
        if (!st.ok()) {
            row.success = false;
            row.error = st.message();
            row.verdict = "FAIL";
            return row;
        }
        for (uint32_t i = 0; i < data.drift_half.size(); ++i) {
            st = db->Put(cfg.n + i, data.drift_half[i]);
            if (!st.ok()) {
                row.success = false;
                row.error = st.message();
                row.verdict = "FAIL";
                return row;
            }
        }
    }
    auto t1 = Clock::now();
    row.ingest_sec = std::chrono::duration<double>(t1 - t0).count();
    row.ingest_qps = static_cast<double>(cfg.n) / std::max(1e-9, row.ingest_sec);
    auto ring = ReadResources();
    row.rss_ingest_kb = ring.rss_kb;
    row.peak_rss_kb = ring.peak_rss_kb;

    const uint32_t warmup = std::min(100u, cfg.queries);
    pomai::SearchResult sres;
    pomai::SearchOptions sopt;
    if (cfg.routing == "fanout") sopt.force_fanout = true;
    if (cfg.probe > 0) sopt.routing_probe_override = cfg.probe;

    for (uint32_t i = 0; i < warmup; ++i) {
        (void)db->Search(data.queries[i], cfg.topk, sopt, &sres);
    }

    std::vector<double> lats;
    lats.reserve(cfg.queries);
    std::vector<uint32_t> routed_shards, routed_probe;
    double r1 = 0.0, r10 = 0.0, r100 = 0.0;

    std::vector<std::vector<float>> oracle_data = data.base;
    if (sc.epoch_drift) {
        oracle_data.insert(oracle_data.end(), data.drift_half.begin(), data.drift_half.end());
    }

    auto q0 = Clock::now();
    for (uint32_t i = 0; i < cfg.queries; ++i) {
        auto qs = Clock::now();
        st = db->Search(data.queries[i], cfg.topk, sopt, &sres);
        auto qe = Clock::now();
        if (!st.ok()) {
            row.success = false;
            row.error = st.message();
            row.verdict = "FAIL";
            return row;
        }
        lats.push_back(std::chrono::duration<double, std::micro>(qe - qs).count());
        routed_shards.push_back(sres.routed_shards_count);
        routed_probe.push_back(sres.routing_probe_centroids);

        const uint32_t gt_k = std::max<uint32_t>(100, cfg.topk);
        auto gt = BruteForceTopK(oracle_data, data.queries[i], gt_k);
        r1 += RecallAtK(sres.hits, gt, 1);
        r10 += RecallAtK(sres.hits, gt, 10);
        r100 += RecallAtK(sres.hits, gt, 100);
    }
    auto q1 = Clock::now();

    row.query_qps = static_cast<double>(cfg.queries) /
        std::max(1e-9, std::chrono::duration<double>(q1 - q0).count());
    row.p50_us = Percentile(lats, 0.50);
    row.p90_us = Percentile(lats, 0.90);
    row.p95_us = Percentile(lats, 0.95);
    row.p99_us = Percentile(lats, 0.99);
    row.p999_us = Percentile(lats, 0.999);
    row.recall1 = r1 / cfg.queries;
    row.recall10 = r10 / cfg.queries;
    row.recall100 = r100 / cfg.queries;

    std::vector<double> rs(routed_shards.begin(), routed_shards.end());
    std::vector<double> rp(routed_probe.begin(), routed_probe.end());
    row.routed_shards_avg = routed_shards.empty() ? 0.0 : std::accumulate(rs.begin(), rs.end(), 0.0) / rs.size();
    row.routed_shards_p95 = Percentile(rs, 0.95);
    row.routed_probe_avg = routed_probe.empty() ? 0.0 : std::accumulate(rp.begin(), rp.end(), 0.0) / rp.size();
    row.routed_probe_p95 = Percentile(rp, 0.95);

    auto rquery = ReadResources();
    row.rss_query_kb = rquery.rss_kb;
    row.peak_rss_kb = std::max(row.peak_rss_kb, rquery.peak_rss_kb);

    struct rusage ru1{};
    getrusage(RUSAGE_SELF, &ru1);
    row.user_cpu_sec = ToSec(ru1.ru_utime) - ToSec(ru0.ru_utime);
    row.sys_cpu_sec = ToSec(ru1.ru_stime) - ToSec(ru0.ru_stime);

    db->Close();
    return row;
}

std::vector<ScenarioConfig> BuildMatrix(const CliConfig& cli) {
    std::vector<ScenarioConfig> out;
    auto mk = [&](std::string name, std::string dataset, std::string routing, uint32_t n, uint32_t d,
                  uint32_t shards, uint32_t q, uint32_t topk, bool epoch = false) {
        ScenarioConfig s;
        s.name = std::move(name);
        s.cfg = cli;
        s.cfg.dataset = dataset;
        s.cfg.routing = routing;
        s.cfg.n = n;
        s.cfg.dim = d;
        s.cfg.shards = shards;
        s.cfg.queries = q;
        s.cfg.topk = topk;
        s.cfg.path = cli.path + "/" + s.name;
        s.epoch_drift = epoch;
        out.push_back(std::move(s));
    };

    mk("small_fanout", "uniform", "fanout", 10000, 128, 4, 1000, 10);
    mk("small_cbrs", "uniform", "cbrs", 10000, 128, 4, 1000, 10);
    mk("small_cbrs_no_dual", "uniform", "cbrs_no_dual", 10000, 128, 4, 1000, 10);

    mk("medium_1shard", "clustered", "cbrs", 100000, 256, 1, 600, 10);
    mk("medium_2shard", "clustered", "cbrs", 100000, 256, 2, 600, 10);
    mk("medium_4shard", "clustered", "cbrs", 100000, 256, 4, 600, 10);
    mk("large_8shard", "clustered", "cbrs", 500000, 256, 8, 200, 10);

    mk("highdim_top1", "uniform", "cbrs", 200000, 512, 4, 300, 1);
    mk("highdim_top100", "uniform", "cbrs", 200000, 512, 4, 300, 100);

    mk("overlap_fanout", "overlap", "fanout", 100000, 256, 4, 500, 10);
    mk("overlap_cbrs", "overlap", "cbrs", 100000, 256, 4, 500, 10);

    mk("skew_fanout", "skew", "fanout", 100000, 128, 8, 500, 10);
    mk("skew_cbrs", "skew", "cbrs", 100000, 128, 8, 500, 10);

    mk("epoch_drift_dual_on", "clustered", "cbrs", 100000, 256, 4, 600, 10, true);
    out.back().cfg.probe = 1;
    mk("epoch_drift_dual_off", "clustered", "cbrs_no_dual", 100000, 256, 4, 600, 10, true);
    out.back().cfg.probe = 1;
    return out;
}

void ParseArgs(int argc, char** argv, CliConfig* c) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--path") c->path = next();
        else if (a == "--seed") c->seed = std::stoull(next());
        else if (a == "--shards") c->shards = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--dim") c->dim = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--n") c->n = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--queries") c->queries = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--topk") c->topk = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--dataset") c->dataset = next();
        else if (a == "--clusters") c->clusters = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--routing") c->routing = next();
        else if (a == "--probe") c->probe = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--k_global") c->k_global = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--fsync") c->fsync = next();
        else if (a == "--threads") c->threads = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--report_json") c->report_json = next();
        else if (a == "--report_csv") c->report_csv = next();
        else if (a == "--matrix") c->matrix = next();
    }
}

} // namespace

int main(int argc, char** argv) {
    CliConfig cli;
    ParseArgs(argc, argv, &cli);

    const auto now = std::chrono::system_clock::now();
    const auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    fs::create_directories("out");
    if (cli.report_json.empty()) cli.report_json = "out/bench_cbrs_" + std::to_string(epoch) + ".json";
    if (cli.report_csv.empty()) cli.report_csv = "out/bench_cbrs_" + std::to_string(epoch) + ".csv";

    std::vector<ScenarioConfig> scenarios;
    if (cli.matrix == "full") {
        scenarios = BuildMatrix(cli);
    } else {
        ScenarioConfig one;
        one.name = "single";
        one.cfg = cli;
        one.cfg.path = cli.path;
        scenarios.push_back(std::move(one));
    }

    std::vector<Row> rows;
    rows.reserve(scenarios.size());

    for (const auto& s : scenarios) {
        std::printf("\n=== Scenario: %s ===\n", s.name.c_str());
        auto row = RunScenario(s);
        rows.push_back(row);
        std::printf("ingest_qps=%.1f query_qps=%.1f p99=%.1fus recall@10=%.4f routed_shards_avg=%.2f\n",
                    row.ingest_qps, row.query_qps, row.p99_us, row.recall10, row.routed_shards_avg);
    }

    // verdict vs fanout baseline by dataset+shape
    for (auto& r : rows) {
        const Row* baseline = nullptr;
        for (const auto& b : rows) {
            if (b.routing == "fanout" && b.dataset == r.dataset && b.dim == r.dim && b.n == r.n && b.topk == r.topk) {
                baseline = &b;
                break;
            }
        }
        if (!r.success) {
            r.verdict = "FAIL";
        } else if (r.topk >= 10 && r.recall10 < 0.94) {
            r.verdict = "FAIL";
        } else if (r.topk < 10 && r.recall1 < 0.94) {
            r.verdict = "FAIL";
        } else if (baseline && r.routing != "fanout") {
            const double latency_improve = (baseline->p99_us - r.p99_us) / std::max(1e-9, baseline->p99_us);
            if (latency_improve >= 0.05 || r.routed_shards_avg <= baseline->shards * 0.5) r.verdict = "PASS";
            else r.verdict = "WARN";
        } else {
            r.verdict = "PASS";
        }
    }

    WriteJson(cli.report_json, rows);
    WriteCsv(cli.report_csv, rows);

    std::printf("\n%-22s %-9s %-8s %-8s %-8s %-8s %-8s %-8s\n",
                "scenario", "routing", "rec@10", "p99us", "qps", "ing_qps", "r_sh_avg", "verdict");
    for (const auto& r : rows) {
        std::printf("%-22s %-9s %-8.3f %-8.1f %-8.1f %-8.1f %-8.2f %-8s\n",
                    r.scenario.c_str(), r.routing.c_str(), r.recall10, r.p99_us, r.query_qps,
                    r.ingest_qps, r.routed_shards_avg, r.verdict.c_str());
    }
    std::printf("\nJSON: %s\nCSV: %s\n", cli.report_json.c_str(), cli.report_csv.c_str());
    return 0;
}
