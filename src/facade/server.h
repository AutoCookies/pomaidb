/*
 * src/facade/server.h
 * Updated for Hardcore Mode: Uses GlobalOrchestrator instead of PPSM.
 *
 * Changes:
 * - Removed PPSM/ShardManager ownership.
 * - Added GlobalOrchestrator* pointer (passed from main).
 * - VSET now hashes string keys to uint64 labels for the lock-free engine.
 */

#pragma once

#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/eventfd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/resource.h>
#include <errno.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstring>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <thread>
#include <functional>
#include <cstdint>

#include "src/core/map.h"
#include "src/core/metrics.h"
#include "src/core/config.h"
#include "src/core/orchestrator.h"

class PomaiServer
{
public:
    // Updated Constructor: Receives the Orchestrator
    PomaiServer(PomaiMap *kv_map, pomai::core::GlobalOrchestrator *orch, int port)
        : kv_map_(kv_map), orch_(orch), port_(port)
    {
        start();
    }

    ~PomaiServer()
    {
        stop();
    }

    void stop()
    {
        running_ = false;
        uint64_t u = 1;
        if (event_fd_ != -1) {
            ssize_t w = write(event_fd_, &u, sizeof(u));
            (void)w;
        }
        if (worker_.joinable()) worker_.join();
        if (listen_fd_ != -1) { close(listen_fd_); listen_fd_ = -1; }
        if (epoll_fd_ != -1) { close(epoll_fd_); epoll_fd_ = -1; }
        if (event_fd_ != -1) { close(event_fd_); event_fd_ = -1; }
    }

private:
    PomaiMap *kv_map_;
    pomai::core::GlobalOrchestrator *orch_; // [NEW] Reference to the engine
    int port_;
    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    int event_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_;

    struct Conn
    {
        int fd;
        std::vector<char> rbuf;
        std::vector<char> wbuf;
    };

    void start()
    {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) throw std::runtime_error("socket failed");

        int opt = 1;
        (void)setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, static_cast<socklen_t>(sizeof(opt)));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
            throw std::runtime_error("bind failed");

        if (listen(listen_fd_, 1024) < 0)
            throw std::runtime_error("listen failed");

        set_nonblocking(listen_fd_);

        epoll_fd_ = epoll_create1(0);
        if (epoll_fd_ < 0) throw std::runtime_error("epoll_create1 failed");

        event_fd_ = static_cast<int>(eventfd(0, EFD_NONBLOCK));
        if (event_fd_ < 0) throw std::runtime_error("eventfd failed");

        add_event(listen_fd_, EPOLLIN);
        add_event(event_fd_, EPOLLIN);

        running_ = true;
        worker_ = std::thread(&PomaiServer::loop, this);
    }

    void set_nonblocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0)
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    void add_event(int fd, uint32_t events)
    {
        struct epoll_event ev{};
        ev.events = events;
        ev.data.fd = fd;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) != 0) {
            throw std::runtime_error(std::string("epoll_ctl add failed: ") + std::strerror(errno));
        }
    }

    // FNV-1a 64-bit for hashing keys to uint64 labels (self-contained)
    static uint64_t fnv1a_hash(const void* data, size_t len) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i) {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Helper: Fast string hash for VSET keys
    static uint64_t hash_key(const std::string& key) {
        return fnv1a_hash(key.data(), key.size());
    }

    size_t parse_resp(const std::vector<char>& buf, std::vector<std::string>& args)
    {
        if (buf.empty()) return 0;
        const char* ptr = buf.data();
        const char* end = ptr + buf.size();

        if (*ptr != '*') return 0;
        const char* crlf = static_cast<const char*>(memchr(ptr, '\n', end - ptr));
        if (!crlf) return 0;

        int count = 0;
        try {
            std::string num_str(ptr + 1, crlf - (ptr + 1));
            if (!num_str.empty() && num_str.back() == '\r') num_str.pop_back();
            count = std::stoi(num_str);
        } catch (...) { return 0; }

        const char* curr = crlf + 1;
        args.clear();
        args.reserve(count);

        for (int i = 0; i < count; ++i)
        {
            if (curr >= end) return 0;
            if (*curr != '$') return 0;
            const char* len_end = static_cast<const char*>(memchr(curr, '\n', end - curr));
            if (!len_end) return 0;

            int len = 0;
            try {
                std::string len_str(curr + 1, len_end - (curr + 1));
                if (!len_str.empty() && len_str.back() == '\r') len_str.pop_back();
                len = std::stoi(len_str);
            } catch (...) { return 0; }

            const char* data_start = len_end + 1;
            const char* data_end = data_start + len;
            const char* next_item = data_end + 2;
            if (next_item > end) return 0;

            args.emplace_back(data_start, len);
            curr = next_item;
        }
        return static_cast<size_t>(curr - ptr);
    }

    void handle_request(Conn &c)
    {
        while (true)
        {
            std::vector<std::string> args;
            size_t consumed = parse_resp(c.rbuf, args);

            if (consumed == 0) return;

            if (consumed < c.rbuf.size()) {
                std::vector<char> remaining(c.rbuf.begin() + consumed, c.rbuf.end());
                c.rbuf = std::move(remaining);
            } else {
                c.rbuf.clear();
            }

            if (args.empty()) continue;
            std::string cmd = args[0];
            std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);

            std::string response;

            // --- Basic KV Commands ---
            if (cmd == "PING") {
                response = "+PONG\r\n";
            }
            else if (cmd == "SET" && args.size() == 3) {
                bool ok = kv_map_->put(args[1].data(), args[1].size(), args[2].data(), args[2].size());
                response = ok ? "+OK\r\n" : "-ERR set failed\r\n";
            }
            else if (cmd == "GET" && args.size() == 2) {
                uint32_t vlen = 0;
                const char *val = kv_map_->get(args[1].data(), args[1].size(), &vlen);
                if (val) {
                    response = "$" + std::to_string(vlen) + "\r\n";
                    response.append(val, vlen);
                    response += "\r\n";
                } else {
                    response = "$-1\r\n";
                }
            }
            // --- Hardcore Vector Commands ---
            else if (cmd == "VSET" && args.size() == 3) {
                const std::string& key = args[1];
                const std::string& vdata = args[2];
                size_t dim = vdata.size() / sizeof(float);

                if (vdata.size() % sizeof(float) != 0 || dim == 0) {
                    response = "-ERR invalid vector data size\r\n";
                } else if (!orch_) {
                    response = "-ERR orchestrator not initialized\r\n";
                } else {
                    const float* vec = reinterpret_cast<const float*>(vdata.data());

                    // 1. Hash Key -> Label (for Sharding & ID)
                    uint64_t label = hash_key(key);

                    // 2. Dispatch to Lock-free Engine
                    bool ok = orch_->insert(vec, label);
                    response = ok ? "+OK\r\n" : "-ERR insert failed\r\n";
                }
            }
            else if (cmd == "VGET" && args.size() == 2) {
                // VGET <label_or_key>
                const std::string& key = args[1];
                uint64_t label = 0;
                // try parse numeric label first
                try {
                    label = std::stoull(key);
                } catch (...) {
                    // fallback: key string -> hash
                    label = hash_key(key);
                }
                std::vector<float> vec;
                if (!orch_) {
                    response = "-ERR orchestrator not initialized\r\n";
                } else if (!orch_->get(label, vec)) {
                    response = "$-1\r\n";
                } else {
                    std::string body(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
                    response = "$" + std::to_string(body.size()) + "\r\n" + body + "\r\n";
                }
            }
            else if (cmd == "VDEL" && args.size() == 2) {
                const std::string& key = args[1];
                uint64_t label = 0;
                try {
                    label = std::stoull(key);
                } catch (...) {
                    label = hash_key(key);
                }
                if (!orch_) {
                    response = "-ERR orchestrator not initialized\r\n";
                } else {
                    bool ok = orch_->remove(label);
                    response = ok ? "+OK\r\n" : "-ERR remove failed\r\n";
                }
            }
            else if (cmd == "VSEARCH" && args.size() >= 3) {
                const std::string& qdata = args[1];
                size_t topk = 10;
                try { topk = std::stoull(args[2]); } catch(...) {}

                size_t dim = qdata.size() / sizeof(float);
                if (!orch_) {
                    response = "-ERR orchestrator not initialized\r\n";
                } else {
                    const float* query = reinterpret_cast<const float*>(qdata.data());

                    // Scatter-Gather Search
                    auto results = orch_->search(query, topk);

                    std::ostringstream ss;
                    ss << "*" << results.size() << "\r\n";
                    for (const auto& p : results) {
                        uint64_t label = p.first;
                        float dist = p.second;

                        // Return Label as Key (Since Orchestrator is pure vector store now)
                        // In a full system, you would reverse-lookup Label->Key here.
                        std::string key_repr = std::to_string(label);

                        ss << "*2\r\n";
                        ss << "$" << key_repr.size() << "\r\n" << key_repr << "\r\n";
                        std::string score = std::to_string(dist);
                        ss << "$" << score.size() << "\r\n" << score << "\r\n";
                    }
                    response = ss.str();
                }
            }
            else if (cmd == "MEMORY") {
                // Simplified memory stats
                response = "$22\r\nManaged by ShardArena\r\n";
            }
            else {
                response = "-ERR unknown command\r\n";
            }

            size_t sent = 0;
            while (sent < response.size()) {
                ssize_t n = write(c.fd, response.data() + sent, response.size() - sent);
                if (n < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                    break;
                }
                sent += static_cast<size_t>(n);
            }
        }
    }

    void loop()
    {
        std::vector<struct epoll_event> events(1024);
        std::unordered_map<int, Conn> conns;

        while (running_)
        {
            int n = epoll_wait(epoll_fd_, events.data(), static_cast<int>(events.size()), 100);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            for (int i = 0; i < n; ++i)
            {
                int fd = events[i].data.fd;
                if (fd == event_fd_) {
                    uint64_t u;
                    if (read(fd, &u, sizeof(u)) < 0) { /* ignore */ }
                }
                else if (fd == listen_fd_) {
                    int conn_fd = accept(listen_fd_, nullptr, nullptr);
                    if (conn_fd >= 0) {
                        set_nonblocking(conn_fd);
                        add_event(conn_fd, EPOLLIN);
                        conns.emplace(conn_fd, Conn{conn_fd, {}, {}});
                    }
                }
                else {
                    if (events[i].events & EPOLLIN) {
                        char buf[4096];
                        ssize_t rb = read(fd, buf, sizeof(buf));
                        if (rb > 0) {
                            Conn& c = conns[fd];
                            c.rbuf.insert(c.rbuf.end(), buf, buf + rb);
                            handle_request(c);
                        }
                        else if (rb == 0 || (rb < 0 && errno != EAGAIN)) {
                            close(fd);
                            conns.erase(fd);
                        }
                    }
                }
            }
        }
        for (auto &p : conns) close(p.first);
    }
};