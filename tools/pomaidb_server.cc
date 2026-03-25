// Standalone process: open PomaiDB and run the Edge Gateway (HTTP + line ingest).
// Intended for Docker / VM / “server mode” while the library remains embeddable in-process.
//
// Environment (all optional except sensible defaults):
//   POMAID_DATA_PATH     database directory (default: /data/pomaidb)
//   POMAID_DIM           vector dimension (default: 128)
//   POMAID_SHARD_COUNT   shards (default: 1)
//   POMAID_PORT          single TCP port for HTTP + ingest (default: 2406)
//   POMAID_HTTP_PORT     HTTP only when different from ingest (default: 2406)
//   POMAID_INGEST_PORT   line protocol only when different from HTTP (default: 2406)
//   POMAID_EDGE_PROFILE  low_ram | balanced | throughput — applies DBOptions presets
//   POMAID_GATEWAY_TOKEN if set, enables StartEdgeGatewaySecure with this token
//
// When POMAID_HTTP_PORT == POMAID_INGEST_PORT (including defaults), one listener serves
// both REST/health/metrics and MQTT|… / WS|… lines.

#include "pomai/pomai.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <signal.h>
#endif

namespace {

std::atomic_bool g_run{true};

void on_signal(int) { g_run = false; }

unsigned getenv_u(const char* key, unsigned def) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def;
    char* end = nullptr;
    unsigned long x = std::strtoul(v, &end, 10);
    if (end == v) return def;
    return static_cast<unsigned>(x);
}

std::uint16_t getenv_port(const char* key, std::uint16_t def) {
    unsigned u = getenv_u(key, def);
    if (u == 0 || u > 65535u) return def;
    return static_cast<std::uint16_t>(u);
}

bool apply_edge_profile_from_env(pomai::DBOptions* opt) {
    const char* v = std::getenv("POMAID_EDGE_PROFILE");
    if (!v || !*v) return false;
    if (std::strcmp(v, "low_ram") == 0) {
        opt->edge_profile = pomai::EdgeProfile::kLowRam;
    } else if (std::strcmp(v, "balanced") == 0) {
        opt->edge_profile = pomai::EdgeProfile::kBalanced;
    } else if (std::strcmp(v, "throughput") == 0) {
        opt->edge_profile = pomai::EdgeProfile::kThroughput;
    } else {
        std::cerr << "pomaidb_server: unknown POMAID_EDGE_PROFILE=" << v << " (ignored)\n";
        return false;
    }
    opt->ApplyEdgeProfile();
    return true;
}

void print_usage() {
    std::cerr
        << "pomaidb_server — PomaiDB Edge Gateway (HTTP + ingest TCP)\n\n"
        << "Environment:\n"
        << "  POMAID_DATA_PATH      DB path (default /data/pomaidb)\n"
        << "  POMAID_DIM            vector dimension (default 128)\n"
        << "  POMAID_SHARD_COUNT    shards (default 1)\n"
        << "  POMAID_PORT           unified gateway port (default 2406)\n"
        << "  POMAID_HTTP_PORT      HTTP when split (default 2406)\n"
        << "  POMAID_INGEST_PORT    line ingest when split (default 2406)\n"
        << "  POMAID_EDGE_PROFILE   low_ram | balanced | throughput\n"
        << "  POMAID_GATEWAY_TOKEN  optional Bearer token\n";
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

#if !defined(_WIN32)
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#endif

    pomai::DBOptions opt;
    if (const char* p = std::getenv("POMAID_DATA_PATH")) {
        opt.path = p;
    } else {
        opt.path = "/data/pomaidb";
    }
    opt.dim = getenv_u("POMAID_DIM", 128);
    opt.shard_count = getenv_u("POMAID_SHARD_COUNT", 1);
    (void)apply_edge_profile_from_env(&opt);

    std::uint16_t http_port;
    std::uint16_t ingest_port;
    if (const char* pp = std::getenv("POMAID_PORT"); pp != nullptr && pp[0] != '\0') {
        const std::uint16_t p = getenv_port("POMAID_PORT", 2406);
        http_port = p;
        ingest_port = p;
    } else {
        http_port = getenv_port("POMAID_HTTP_PORT", 2406);
        ingest_port = getenv_port("POMAID_INGEST_PORT", 2406);
    }

    std::unique_ptr<pomai::DB> db;
    pomai::Status st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "pomaidb_server: Open failed: " << st.message() << '\n';
        return 1;
    }

    const char* tok = std::getenv("POMAID_GATEWAY_TOKEN");
    if (tok != nullptr && tok[0] != '\0') {
        st = db->StartEdgeGatewaySecure(http_port, ingest_port, tok);
    } else {
        st = db->StartEdgeGateway(http_port, ingest_port);
    }
    if (!st.ok()) {
        std::cerr << "pomaidb_server: gateway start failed: " << st.message() << '\n';
        (void)db->Close();
        return 1;
    }

    if (http_port == ingest_port) {
        std::cerr << "pomaidb_server: listening unified port=" << http_port << " path=" << opt.path << '\n';
    } else {
        std::cerr << "pomaidb_server: listening http=" << http_port << " ingest=" << ingest_port << " path=" << opt.path
                  << '\n';
    }

    while (g_run) {
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }

    (void)db->StopEdgeGateway();
    (void)db->Close();
    return 0;
}
