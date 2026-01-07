// Minimal PWP micro-benchmark: in-process server + threaded clients (SET/GET).
// Place this file at tools/bench/pwp_bench.cc

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#include "core/config.h"
#include "memory/arena.h"
#include "core/map.h"
#include "facade/server.h" // include path may be "server.h" in your repo layout

// Re-declare wire header layout locally (same as server)
struct alignas(16) PomaiHeaderWire
{
    uint8_t magic;
    uint8_t op;
    uint16_t status;
    uint32_t klen; // network order
    uint32_t vlen; // network order
    uint32_t reserved;
};

// Opcodes
static constexpr uint8_t MAGIC_P = 0x50; // 'P'

// Simple helper to send a frame (blocking)
static bool send_frame_and_recv_resp(int fd, uint8_t op, const std::string &key, const std::string &val, uint32_t &out_vlen)
{
    PomaiHeaderWire wh{};
    wh.magic = MAGIC_P;
    wh.op = op;
    wh.status = 0;
    wh.klen = htonl(static_cast<uint32_t>(key.size()));
    wh.vlen = htonl(static_cast<uint32_t>(val.size()));
    wh.reserved = 0;

    // build iovec-like single buffer
    std::vector<char> buf;
    buf.resize(sizeof(wh) + key.size() + val.size());
    memcpy(buf.data(), &wh, sizeof(wh));
    memcpy(buf.data() + sizeof(wh), key.data(), key.size());
    if (!val.empty())
        memcpy(buf.data() + sizeof(wh) + key.size(), val.data(), val.size());

    ssize_t w = write(fd, buf.data(), buf.size());
    if (w != (ssize_t)buf.size())
        return false;

    // read response header
    PomaiHeaderWire resp;
    ssize_t r = read(fd, &resp, sizeof(resp));
    if (r != (ssize_t)sizeof(resp))
        return false;
    if (resp.magic != MAGIC_P)
        return false;

    uint32_t resp_vlen = ntohl(resp.vlen);
    out_vlen = resp_vlen;

    // if value present, drain it (we don't use it for latency measurement)
    uint32_t to_read = resp_vlen;
    while (to_read > 0)
    {
        char tmp[4096];
        ssize_t n = read(fd, tmp, std::min<size_t>(sizeof(tmp), to_read));
        if (n <= 0)
            return false;
        to_read -= static_cast<uint32_t>(n);
    }
    return true;
}

struct Stats
{
    std::vector<double> latencies_ms;
    std::atomic<uint64_t> ops{0};
};

int main(int argc, char **argv)
{
    // Simple CLI
    int clients = 4;
    int iters = 10000;
    int val_size = 64;
    int port = static_cast<int>(pomai::config::runtime.default_port);

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--clients" && i + 1 < argc)
            clients = std::stoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = std::stoi(argv[++i]);
        else if (a == "--valsize" && i + 1 < argc)
            val_size = std::stoi(argv[++i]);
        else if (a == "--port" && i + 1 < argc)
            port = std::stoi(argv[++i]);
    }

    std::cout << "PWP benchmark: clients=" << clients << " iters=" << iters << " val_size=" << val_size << " port=" << port << "\n";

    // Start server in background thread using small arena
    uint64_t arena_mb = 64; // bench-friendly small arena
    PomaiArena arena = PomaiArena::FromMB(arena_mb);
    if (!arena.is_valid())
    {
        std::cerr << "arena alloc failed\n";
        return 1;
    }
    uint64_t max_seeds = arena.get_capacity_bytes() / sizeof(Seed);
    uint64_t slots = 1;
    while (slots < max_seeds)
        slots <<= 1;
    if (slots == 0)
        slots = 1;
    PomaiMap map(&arena, slots);

    PomaiServer server(&map, port);
    std::thread srv_thread([&]
                           { server.run(); });

    // small sleep to let server bind
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Stats per client
    std::vector<std::unique_ptr<Stats>> stats(clients);
    for (int i = 0; i < clients; ++i)
        stats[i].reset(new Stats);

    std::atomic<uint64_t> started{0};
    std::vector<std::thread> cthreads;
    cthreads.reserve(clients);

    // prepare value bytes (random)
    std::string valbuf(val_size, 'x');
    std::mt19937 rng(12345);
    std::uniform_int_distribution<unsigned> dist(0, 61);
    for (int i = 0; i < val_size; ++i)
        valbuf[i] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[dist(rng) % 62];

    auto client_fn = [&](int id)
    {
        // connect
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
        {
            perror("socket");
            return;
        }
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0)
        {
            perror("connect");
            close(fd);
            return;
        }

        // per-thread stats vector
        auto &S = *stats[id];
        S.latencies_ms.reserve(static_cast<size_t>(iters * 2));

        // each iteration: SET then GET for a unique key
        for (int it = 0; it < iters; ++it)
        {
            // unique key per client/iter
            std::string key = "k" + std::to_string(id) + "-" + std::to_string(it);

            // SET
            auto t0 = std::chrono::steady_clock::now();
            uint32_t rlen = 0;
            if (!send_frame_and_recv_resp(fd, OP_SET, key, valbuf, rlen))
            {
                break;
            }
            auto t1 = std::chrono::steady_clock::now();
            double lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            S.latencies_ms.push_back(lat_ms);
            S.ops.fetch_add(1, std::memory_order_relaxed);

            // GET
            t0 = std::chrono::steady_clock::now();
            if (!send_frame_and_recv_resp(fd, OP_GET, key, std::string(), rlen))
            {
                break;
            }
            t1 = std::chrono::steady_clock::now();
            lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            S.latencies_ms.push_back(lat_ms);
            S.ops.fetch_add(1, std::memory_order_relaxed);
        }

        close(fd);
    };

    // spawn clients
    for (int i = 0; i < clients; ++i)
        cthreads.emplace_back(client_fn, i);

    // wait for clients
    for (auto &t : cthreads)
        if (t.joinable())
            t.join();

    // aggregate stats
    uint64_t total_ops = 0;
    std::vector<double> all_lats;
    for (int i = 0; i < clients; ++i)
    {
        total_ops += stats[i]->ops.load();
        all_lats.insert(all_lats.end(), stats[i]->latencies_ms.begin(), stats[i]->latencies_ms.end());
    }

    // compute metrics
    std::sort(all_lats.begin(), all_lats.end());
    double sum = 0;
    for (double v : all_lats)
        sum += v;
    double avg = all_lats.empty() ? 0.0 : (sum / all_lats.size());
    double median = all_lats.empty() ? 0.0 : all_lats[all_lats.size() / 2];
    double p95 = all_lats.empty() ? 0.0 : all_lats[std::min<size_t>(all_lats.size() - 1, (size_t)((all_lats.size() * 95) / 100))];

    std::cout << "\nBenchmark results:\n";
    std::cout << "  clients: " << clients << "\n";
    std::cout << "  per-client iters: " << iters << "\n";
    std::cout << "  total ops (SET+GET): " << total_ops << "\n";
    std::cout << "  avg latency (ms): " << avg << "\n";
    std::cout << "  median (ms): " << median << "\n";
    std::cout << "  p95 (ms): " << p95 << "\n";

    // stop server
    server.stop();
    if (srv_thread.joinable())
        srv_thread.join();

    return 0;
}