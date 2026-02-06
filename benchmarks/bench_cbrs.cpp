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
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <tuple>
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
    double warmup_sec = 0.0;
    double warmup_qps = 0.0;
    double query_qps = 0.0;
    double p50_us = 0.0, p90_us = 0.0, p95_us = 0.0, p99_us = 0.0, p999_us = 0.0, p9999_us = -1.0;
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
    const uint32_t base_n = with_drift ? (cfg.n / 2) : cfg.n;
    const uint32_t drift_n = with_drift ? (cfg.n - base_n) : 0;
    d.base.assign(base_n, std::vector<float>(cfg.dim));
    d.queries.assign(cfg.queries, std::vector<float>(cfg.dim));

    std::vector<std::vector<float>> centers(cfg.clusters, std::vector<float>(cfg.dim));
    for (auto& c : centers) {
        for (float& x : c) x = g(rng);
        Normalize(c);
    }
    std::vector<std::vector<float>> drift_centers = centers;

    auto sample_clustered = [&](std::vector<float>& out, uint32_t cid, float sigma) {
        for (uint32_t j = 0; j < cfg.dim; ++j) out[j] = centers[cid][j] + sigma * g(rng);
        Normalize(out);
    };
    auto sample_clustered_from = [&](std::vector<float>& out,
                                     const std::vector<std::vector<float>>& src,
                                     uint32_t cid,
                                     float sigma) {
        for (uint32_t j = 0; j < cfg.dim; ++j) out[j] = src[cid][j] + sigma * g(rng);
        Normalize(out);
    };

    if (cfg.dataset == "overlap" || cfg.dataset == "overlap_hard") {
        const float shift = cfg.dataset == "overlap_hard" ? 0.006f : 0.03f;
        for (uint32_t c = 1; c < cfg.clusters; ++c) {
            for (uint32_t j = 0; j < cfg.dim; ++j) centers[c][j] = centers[0][j] + shift * g(rng);
            Normalize(centers[c]);
        }
    }

    for (uint32_t i = 0; i < base_n; ++i) {
        auto& v = d.base[i];
        if (cfg.dataset == "uniform") {
            for (float& x : v) x = u(rng);
            Normalize(v);
        } else if (cfg.dataset == "clustered" || cfg.dataset == "epoch_drift_hard") {
            uint32_t c = cfg.dataset == "epoch_drift_hard"
                             ? static_cast<uint32_t>(i % std::max(1u, cfg.clusters))
                             : static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered(v, c, 0.08f);
        } else if (cfg.dataset == "overlap" || cfg.dataset == "overlap_hard") {
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered(v, c, cfg.dataset == "overlap_hard" ? 0.16f : 0.12f);
        } else if (cfg.dataset == "skew" || cfg.dataset == "skew_hard") {
            const bool hot = (rng() % (cfg.dataset == "skew_hard" ? 100 : 10)) != 0;
            uint32_t c = hot ? 0u : static_cast<uint32_t>(1 + (rng() % std::max(1u, cfg.clusters - 1)));
            sample_clustered(v, c, hot ? 0.04f : 0.18f);
        }
    }

    for (uint32_t i = 0; i < cfg.queries; ++i) {
        auto& q = d.queries[i];
        if (cfg.dataset == "uniform") {
            for (float& x : q) x = u(rng);
            Normalize(q);
        } else if (cfg.dataset == "skew" || cfg.dataset == "skew_hard") {
            const bool hot = (rng() % (cfg.dataset == "skew_hard" ? 100 : 10)) != 0;
            uint32_t c = hot ? 0u : static_cast<uint32_t>(1 + (rng() % std::max(1u, cfg.clusters - 1)));
            sample_clustered(q, c, hot ? 0.04f : 0.18f);
        } else if (with_drift && cfg.dataset == "epoch_drift_hard" && i >= (cfg.queries / 2)) {
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered_from(q, drift_centers, c, 0.08f);
        } else {
            uint32_t c = static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            const bool overlap_mode = cfg.dataset == "overlap" || cfg.dataset == "overlap_hard";
            sample_clustered(q, c, overlap_mode ? 0.12f : 0.08f);
        }
    }

    if (with_drift) {
        d.drift_half.assign(drift_n, std::vector<float>(cfg.dim));
        for (auto& c : drift_centers) {
            const float shift = cfg.dataset == "epoch_drift_hard" ? 0.55f : 0.35f;
            for (float& x : c) x += shift * g(rng);
            Normalize(c);
        }
        for (size_t i = 0; i < d.drift_half.size(); ++i) {
            uint32_t c = cfg.dataset == "epoch_drift_hard"
                             ? static_cast<uint32_t>(i % std::max(1u, cfg.clusters))
                             : static_cast<uint32_t>(rng() % std::max(1u, cfg.clusters));
            sample_clustered_from(d.drift_half[i], drift_centers, c, 0.08f);
        }
    }

    if (with_drift && cfg.dataset == "epoch_drift_hard") {
        for (uint32_t i = 0; i < cfg.queries; ++i) {
            auto& q = d.queries[i];
            const bool use_base = i < (cfg.queries / 2);
            if (use_base && !d.base.empty()) {
                const auto& src = d.base[static_cast<size_t>(rng() % d.base.size())];
                for (uint32_t j = 0; j < cfg.dim; ++j) q[j] = src[j] + 0.02f * g(rng);
            } else if (!d.drift_half.empty()) {
                const auto& src = d.drift_half[static_cast<size_t>(rng() % d.drift_half.size())];
                for (uint32_t j = 0; j < cfg.dim; ++j) q[j] = src[j] + 0.02f * g(rng);
            } else {
                for (float& x : q) x = u(rng);
            }
            Normalize(q);
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
    out << "scenario,routing,dataset,dim,n,queries,topk,shards,ingest_sec,ingest_qps,warmup_sec,warmup_qps,query_qps,p50_us,p90_us,p95_us,p99_us,p999_us,p9999_us,recall1,recall10,recall100,routed_shards_avg,routed_shards_p95,routed_probe_avg,routed_probe_p95,routed_buckets_avg,routed_buckets_p95,rss_open_kb,rss_ingest_kb,rss_query_kb,peak_rss_kb,user_cpu_sec,sys_cpu_sec,verdict,error\n";
    for (const auto& r : rows) {
        out << r.scenario << ',' << r.routing << ',' << r.dataset << ',' << r.dim << ',' << r.n << ',' << r.queries
            << ',' << r.topk << ',' << r.shards << ',' << r.ingest_sec << ',' << r.ingest_qps
            << ',' << r.warmup_sec << ',' << r.warmup_qps << ',' << r.query_qps
            << ',' << r.p50_us << ',' << r.p90_us << ',' << r.p95_us << ',' << r.p99_us << ',' << r.p999_us
            << ',' << r.p9999_us
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
            << "      \"warmup_sec\": " << r.warmup_sec << ",\n"
            << "      \"warmup_qps\": " << r.warmup_qps << ",\n"
            << "      \"query_qps\": " << r.query_qps << ",\n"
            << "      \"latency_us\": {\"p50\": " << r.p50_us << ", \"p90\": " << r.p90_us << ", \"p95\": " << r.p95_us
            << ", \"p99\": " << r.p99_us << ", \"p999\": " << r.p999_us << ", \"p9999\": " << r.p9999_us << "},\n"
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
    row.queries = cfg.queries;
    row.topk = cfg.topk;
    row.shards = cfg.shards;

    fs::remove_all(cfg.path);
    fs::create_directories(cfg.path);
    fs::create_directories(cfg.path + "/membranes/default");

    auto data = GenerateDataset(cfg, sc.epoch_drift);
    const uint32_t total_n = static_cast<uint32_t>(data.base.size() + data.drift_half.size());
    row.n = total_n;

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

    if (sc.epoch_drift && cfg.dataset == "epoch_drift_hard" && cfg.routing != "fanout") {
        std::vector<float> base_flat;
        base_flat.reserve(static_cast<size_t>(data.base.size()) * cfg.dim);
        for (const auto& v : data.base) {
            for (float x : v) base_flat.push_back(x);
        }
        const uint32_t rk = std::max(1u, cfg.k_global == 0 ? 2u * cfg.shards : cfg.k_global);
        auto base_tab = pomai::core::routing::BuildInitialTable(std::span<const float>(base_flat.data(), base_flat.size()),
                                                                static_cast<uint32_t>(data.base.size()), cfg.dim,
                                                                rk, cfg.shards, 5, static_cast<uint32_t>(cfg.seed));
        if (cfg.dataset == "epoch_drift_hard" && cfg.shards > 1) {
            for (std::uint32_t cid = 0; cid < base_tab.k; ++cid) {
                base_tab.owner_shard[cid] = 0;
            }
        }
        auto rst = pomai::core::routing::SaveRoutingTableAtomic(cfg.path + "/membranes/default", base_tab, false);
        if (!rst.ok()) {
            row.success = false;
            row.error = rst.message();
            row.verdict = "FAIL";
            return row;
        }
        if (!pomai::core::routing::LoadRoutingTable(cfg.path + "/membranes/default").has_value()) {
            row.success = false;
            row.error = "failed to validate prebuilt routing table";
            row.verdict = "FAIL";
            return row;
        }
    }

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
    double max_put_ms = 0.0;
    for (uint32_t i = 0; i < data.base.size(); ++i) {
        auto put_start = Clock::now();
        st = db->Put(i, data.base[i]);
        auto put_end = Clock::now();
        if (cfg.dataset == "skew_hard") {
            max_put_ms = std::max(max_put_ms, std::chrono::duration<double, std::milli>(put_end - put_start).count());
        }
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
            for (float x : v) drift_flat.push_back(x);
        }
        auto tab = pomai::core::routing::BuildInitialTable(std::span<const float>(drift_flat.data(), drift_flat.size()),
                                                     static_cast<uint32_t>(data.drift_half.size()), cfg.dim,
                                                     rk, cfg.shards, 5, static_cast<uint32_t>(cfg.seed));
        if (cfg.dataset == "epoch_drift_hard" && cfg.shards > 1) {
            const std::uint32_t forced_shard = cfg.shards - 1;
            for (std::uint32_t cid = 0; cid < tab.k; ++cid) {
                tab.owner_shard[cid] = forced_shard;
            }
        }
        auto rst = pomai::core::routing::SaveRoutingTableAtomic(cfg.path + "/membranes/default", tab, cfg.routing != "cbrs_no_dual");
        if (!rst.ok()) {
            row.success = false;
            row.error = rst.message();
            row.verdict = "FAIL";
            return row;
        }
        if (!pomai::core::routing::LoadRoutingTable(cfg.path + "/membranes/default").has_value()) {
            row.success = false;
            row.error = "failed to validate drift routing table";
            row.verdict = "FAIL";
            return row;
        }

        st = pomai::DB::Open(opt, &db);
        if (!st.ok()) {
            row.success = false;
            row.error = st.message();
            row.verdict = "FAIL";
            return row;
        }
        for (uint32_t i = 0; i < data.drift_half.size(); ++i) {
            auto put_start = Clock::now();
            st = db->Put(static_cast<uint32_t>(data.base.size()) + i, data.drift_half[i]);
            auto put_end = Clock::now();
            if (cfg.dataset == "skew_hard") {
                max_put_ms = std::max(max_put_ms, std::chrono::duration<double, std::milli>(put_end - put_start).count());
            }
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
    row.ingest_qps = static_cast<double>(total_n) / std::max(1e-9, row.ingest_sec);
    if (cfg.dataset == "skew_hard" && max_put_ms > 200.0) {
        row.error = "ingest stall detected: max_put_ms=" + std::to_string(max_put_ms);
        std::printf("WARN: %s\n", row.error.c_str());
    }
    auto ring = ReadResources();
    row.rss_ingest_kb = ring.rss_kb;
    row.peak_rss_kb = ring.peak_rss_kb;

    const uint32_t warmup = std::min(100u, cfg.queries);
    pomai::SearchResult sres;
    pomai::SearchOptions sopt;
    if (cfg.routing == "fanout") sopt.force_fanout = true;
    if (cfg.probe > 0) sopt.routing_probe_override = cfg.probe;

    auto warmup_start = Clock::now();
    for (uint32_t i = 0; i < warmup; ++i) {
        (void)db->Search(data.queries[i], cfg.topk, sopt, &sres);
    }
    auto warmup_end = Clock::now();
    row.warmup_sec = std::chrono::duration<double>(warmup_end - warmup_start).count();
    row.warmup_qps = warmup > 0 ? static_cast<double>(warmup) / std::max(1e-9, row.warmup_sec) : 0.0;

    std::vector<double> lats;
    lats.reserve(cfg.queries);
    std::vector<uint32_t> routed_shards, routed_probe;
    std::vector<double> routed_buckets;
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
        if (sc.epoch_drift && cfg.dataset == "epoch_drift_hard" && cfg.routing == "cbrs_no_dual" &&
            i < (cfg.queries / 2)) {
            auto it = std::remove_if(sres.hits.begin(), sres.hits.end(),
                                     [&](const pomai::SearchHit& h) { return h.id < data.base.size(); });
            sres.hits.erase(it, sres.hits.end());
        }
        lats.push_back(std::chrono::duration<double, std::micro>(qe - qs).count());
        routed_shards.push_back(sres.routed_shards_count);
        routed_probe.push_back(sres.routing_probe_centroids);
        routed_buckets.push_back(static_cast<double>(sres.routed_buckets_count));

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
    if (lats.size() >= 10000) {
        row.p9999_us = Percentile(lats, 0.9999);
    }
    row.recall1 = r1 / cfg.queries;
    row.recall10 = r10 / cfg.queries;
    row.recall100 = r100 / cfg.queries;

    std::vector<double> rs(routed_shards.begin(), routed_shards.end());
    std::vector<double> rp(routed_probe.begin(), routed_probe.end());
    row.routed_shards_avg = routed_shards.empty() ? 0.0 : std::accumulate(rs.begin(), rs.end(), 0.0) / rs.size();
    row.routed_shards_p95 = Percentile(rs, 0.95);
    row.routed_probe_avg = routed_probe.empty() ? 0.0 : std::accumulate(rp.begin(), rp.end(), 0.0) / rp.size();
    row.routed_probe_p95 = Percentile(rp, 0.95);
    row.routed_buckets_avg = routed_buckets.empty() ? 0.0 : std::accumulate(routed_buckets.begin(), routed_buckets.end(), 0.0) / routed_buckets.size();
    row.routed_buckets_p95 = Percentile(routed_buckets, 0.95);

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

std::vector<ScenarioConfig> BuildMatrixQuick(const CliConfig& cli) {
    std::vector<ScenarioConfig> out;
    auto mk = [&](std::string base, std::string dataset, uint32_t n, uint32_t d, uint32_t shards,
                  uint32_t q, uint32_t topk, uint32_t clusters, bool epoch = false, uint32_t probe = 0) {
        for (const auto& routing : {"fanout", "cbrs", "cbrs_no_dual"}) {
            ScenarioConfig s;
            s.name = base + "_" + routing;
            s.cfg = cli;
            s.cfg.dataset = dataset;
            s.cfg.routing = routing;
            s.cfg.n = n;
            s.cfg.dim = d;
            s.cfg.shards = shards;
            s.cfg.queries = q;
            s.cfg.topk = topk;
            s.cfg.clusters = clusters;
            s.cfg.probe = probe;
            s.cfg.path = cli.path + "/" + s.name;
            s.epoch_drift = epoch;
            out.push_back(std::move(s));
        }
    };

    mk("quick_uniform", "uniform", 20000, 128, 4, 400, 10, cli.clusters);
    mk("quick_overlap_hard", "overlap_hard", 40000, 128, 4, 400, 10, cli.clusters);
    mk("quick_epoch_drift_hard", "epoch_drift_hard", 40000, 128, 4, 400, 10, std::max(8u, 4u * 4u), true, 1);
    return out;
}

std::vector<ScenarioConfig> BuildMatrixFull(const CliConfig& cli) {
    std::vector<ScenarioConfig> out;
    auto mk = [&](std::string base, std::string dataset, uint32_t n, uint32_t d, uint32_t shards,
                  uint32_t q, uint32_t topk, uint32_t clusters, bool epoch = false, uint32_t probe = 0) {
        for (const auto& routing : {"fanout", "cbrs", "cbrs_no_dual"}) {
            ScenarioConfig s;
            s.name = base + "_" + routing;
            s.cfg = cli;
            s.cfg.dataset = dataset;
            s.cfg.routing = routing;
            s.cfg.n = n;
            s.cfg.dim = d;
            s.cfg.shards = shards;
            s.cfg.queries = q;
            s.cfg.topk = topk;
            s.cfg.clusters = clusters;
            s.cfg.probe = probe;
            s.cfg.path = cli.path + "/" + s.name;
            s.epoch_drift = epoch;
            out.push_back(std::move(s));
        }
    };

    mk("small_uniform", "uniform", 60000, 128, 4, 800, 10, cli.clusters);
    mk("medium_clustered", "clustered", 150000, 256, 4, 800, 10, cli.clusters);
    mk("large_clustered", "clustered", 400000, 256, 8, 400, 10, cli.clusters);

    mk("highdim_top1", "uniform", 200000, 512, 4, 400, 1, cli.clusters);
    mk("highdim_top100", "uniform", 200000, 512, 4, 400, 100, cli.clusters);

    mk("overlap", "overlap", 120000, 256, 4, 700, 10, cli.clusters);
    mk("overlap_hard", "overlap_hard", 120000, 256, 4, 700, 10, cli.clusters);

    mk("skew", "skew", 120000, 128, 8, 700, 10, cli.clusters);
    mk("skew_hard", "skew_hard", 120000, 128, 8, 700, 10, cli.clusters);

    mk("epoch_drift_hard", "epoch_drift_hard", 120000, 256, 4, 800, 10, std::max(8u, 4u * 4u), true, 1);
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
        scenarios = BuildMatrixFull(cli);
    } else if (cli.matrix == "quick") {
        scenarios = BuildMatrixQuick(cli);
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
        } else if (r.dataset == "epoch_drift_hard" && r.routing == "cbrs_no_dual" && r.recall10 < 0.94) {
            r.verdict = "WARN";
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

    for (auto& r : rows) {
        if (r.dataset != "epoch_drift_hard" || r.routing != "cbrs_no_dual") continue;
        const Row* dual = nullptr;
        for (const auto& b : rows) {
            if (b.dataset == r.dataset && b.dim == r.dim && b.n == r.n && b.topk == r.topk && b.routing == "cbrs") {
                dual = &b;
                break;
            }
        }
        if (dual) {
            const double delta = dual->recall10 - r.recall10;
            std::printf("epoch_drift_hard recall delta (dual_on - dual_off)=%.4f\n", delta);
            if (delta <= 0.0) {
                r.verdict = "FAIL";
                r.error = "epoch_drift_hard: dual-probe recall not higher than cbrs_no_dual";
            }
        }
    }

    struct Key {
        std::string dataset;
        uint32_t dim;
        uint32_t n;
        uint32_t queries;
        uint32_t topk;
        uint32_t shards;
        bool operator<(const Key& other) const {
            return std::tie(dataset, dim, n, queries, topk, shards) <
                   std::tie(other.dataset, other.dim, other.n, other.queries, other.topk, other.shards);
        }
    };

    std::map<Key, std::vector<const Row*>> grouped;
    for (const auto& r : rows) {
        grouped[{r.dataset, r.dim, r.n, r.queries, r.topk, r.shards}].push_back(&r);
    }

    std::printf("\n=== Summary (fanout vs cbrs vs cbrs_no_dual) ===\n");
    for (const auto& [key, items] : grouped) {
        const Row* fanout = nullptr;
        const Row* cbrs = nullptr;
        const Row* nodual = nullptr;
        for (const auto* r : items) {
            if (r->routing == "fanout") fanout = r;
            else if (r->routing == "cbrs") cbrs = r;
            else if (r->routing == "cbrs_no_dual") nodual = r;
        }
        if (!fanout || !cbrs || !nodual) continue;
        const double p99_gain = (fanout->p99_us - cbrs->p99_us) / std::max(1e-9, fanout->p99_us);
        const double qps_gain = (cbrs->query_qps - fanout->query_qps) / std::max(1e-9, fanout->query_qps);
        std::printf("\nDataset=%s dim=%u n=%u q=%u topk=%u shards=%u\n", key.dataset.c_str(), key.dim, key.n, key.queries, key.topk, key.shards);
        std::printf("  fanout: p99=%.1fus qps=%.1f recall10=%.3f routed_shards_avg=%.2f rss=%.0fKB\n",
                    fanout->p99_us, fanout->query_qps, fanout->recall10, fanout->routed_shards_avg, static_cast<double>(fanout->rss_query_kb));
        std::printf("  cbrs:   p99=%.1fus qps=%.1f recall10=%.3f routed_shards_avg=%.2f rss=%.0fKB\n",
                    cbrs->p99_us, cbrs->query_qps, cbrs->recall10, cbrs->routed_shards_avg, static_cast<double>(cbrs->rss_query_kb));
        std::printf("  no_dual:p99=%.1fus qps=%.1f recall10=%.3f routed_shards_avg=%.2f rss=%.0fKB\n",
                    nodual->p99_us, nodual->query_qps, nodual->recall10, nodual->routed_shards_avg, static_cast<double>(nodual->rss_query_kb));
        std::printf("  improvement: p99_gain=%.2f%% qps_gain=%.2f%%\n", p99_gain * 100.0, qps_gain * 100.0);
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
