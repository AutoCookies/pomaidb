/*
 * src/facade/server.h
 *
 * Pomai Wire Protocol (PWP) Server - Orbit Edition.
 *
 * Architecture:
 * - Network Layer: Epoll-based, non-blocking I/O.
 * - Logic Layer: Dispatches commands to:
 * 1. PomaiMap (Key-Value operations)
 * 2. PPSM (Vector operations - Powered by Pomai Orbit)
 *
 * Dependencies: Cleaned up. No legacy VectorStore/HNSW includes.
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

#include "src/core/map.h"
#include "src/core/metrics.h"
#include "src/core/config.h"
#include "src/core/seed.h"
#include "src/core/pps_manager.h" // Only PPSM is needed for Vectors
#include "src/core/shard_manager.h"

class PomaiServer
{
public:
    PomaiServer(PomaiMap *kv_map, int port)
        : kv_map_(kv_map), port_(port)
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
        // wake up epoll
        if (event_fd_ != -1)
        {
            // Cast to void to silence warn_unused_result
            (void)write(event_fd_, &u, sizeof(u));
        }
        if (worker_.joinable())
            worker_.join();
    }

private:
    PomaiMap *kv_map_;
    int port_;
    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    int event_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_;

    // Vector Engine (Lazy initialized on first vector command)
    std::unique_ptr<pomai::core::ShardManager> shard_mgr_;
    std::unique_ptr<pomai::core::PPSM> pps_manager_;
    std::mutex engine_mu_;

    struct Conn
    {
        int fd;
        std::vector<uint8_t> rbuf;
        std::vector<uint8_t> wbuf;
    };

    void start()
    {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
            throw std::runtime_error("socket failed");

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

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
        event_fd_ = eventfd(0, EFD_NONBLOCK);

        add_event(listen_fd_, EPOLLIN);
        add_event(event_fd_, EPOLLIN);

        running_ = true;
        worker_ = std::thread(&PomaiServer::loop, this);
    }

    void set_nonblocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    void add_event(int fd, uint32_t events)
    {
        struct epoll_event ev{};
        ev.events = events;
        ev.data.fd = fd;
        epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev);
    }

    // --- Core Logic: Ensure Vector Engine is Ready ---
    // PPSM/Orbit is initialized lazily based on the dimension of the first vector seen.
    bool ensure_vector_engine(size_t dim)
    {
        std::lock_guard<std::mutex> lk(engine_mu_);
        if (pps_manager_)
            return true;

        try
        {
            std::cout << "[Server] Initializing Orbit Engine with dim=" << dim << "...\n";

            // 1. Create Shard Manager
            uint32_t shards = pomai::config::runtime.shard_count;
            shard_mgr_ = std::make_unique<pomai::core::ShardManager>(shards);

            // 2. Create PPSM (which creates Orbit instances)
            size_t max_total = pomai::config::runtime.max_elements_total;
            pps_manager_ = std::make_unique<pomai::core::PPSM>(
                shard_mgr_.get(),
                dim,
                max_total,
                /*async_ack=*/true);
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Server] Engine Init Failed: " << e.what() << "\n";
            return false;
        }
    }

    // --- Command Handling ---

    void handle_request(Conn &c)
    {
        // Simple RESP-like parser (simplified for brevity)
        // [NOTE]: This is a placeholder. Real implementation should buffer c.rbuf
        // until a full command is available.

        std::vector<std::string> args;
        // Parsing logic here...

        if (args.empty())
            return;

        std::string cmd = args[0];
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::toupper);

        std::string response;

        if (cmd == "PING")
        {
            response = "+PONG\r\n";
        }
        else if (cmd == "SET" && args.size() == 3)
        {
            // KV Set
            bool ok = kv_map_->put(args[1].data(), args[1].size(), args[2].data(), args[2].size());
            response = ok ? "+OK\r\n" : "-ERR set failed\r\n";
        }
        else if (cmd == "GET" && args.size() == 2)
        {
            // KV Get
            uint32_t vlen = 0;
            const char *val = kv_map_->get(args[1].data(), args[1].size(), &vlen);
            if (val)
            {
                response = "$" + std::to_string(vlen) + "\r\n";
                response.append(val, vlen);
                response += "\r\n";
            }
            else
            {
                response = "$-1\r\n";
            }
        }
        else if (cmd == "VSET" && args.size() == 3)
        {
            // VSET key <binary_vector_bytes>
            const std::string &key = args[1];
            const std::string &vdata = args[2];

            size_t dim = vdata.size() / sizeof(float);
            if (vdata.size() % sizeof(float) != 0 || dim == 0)
            {
                response = "-ERR invalid vector data size\r\n";
            }
            else
            {
                if (ensure_vector_engine(dim))
                {
                    const float *vec = reinterpret_cast<const float *>(vdata.data());
                    bool ok = pps_manager_->addVec(key.data(), key.size(), vec);
                    response = ok ? "+OK\r\n" : "-ERR insert failed\r\n";
                }
                else
                {
                    response = "-ERR engine init failed\r\n";
                }
            }
        }
        else if (cmd == "VSEARCH" && args.size() >= 3)
        {
            // VSEARCH <binary_query> <topk>
            const std::string &qdata = args[1];
            size_t topk = std::stoull(args[2]);

            size_t dim = qdata.size() / sizeof(float);
            if (!pps_manager_)
            {
                response = "-ERR engine not ready (insert data first)\r\n";
            }
            else
            {
                const float *query = reinterpret_cast<const float *>(qdata.data());
                auto results = pps_manager_->search(query, dim, topk);

                std::ostringstream ss;
                ss << "*" << results.size() << "\r\n";
                for (const auto &p : results)
                {
                    ss << "*2\r\n";
                    ss << "$" << p.first.size() << "\r\n"
                       << p.first << "\r\n"; // Key
                    std::string score_str = std::to_string(p.second);
                    ss << "$" << score_str.size() << "\r\n"
                       << score_str << "\r\n";
                }
                response = ss.str();
            }
        }
        else if (cmd == "MEMORY")
        {
            if (pps_manager_)
            {
                auto mem = pps_manager_->memoryUsage();
                std::ostringstream ss;
                ss << "Payload: " << (mem.payload_bytes / 1024 / 1024) << " MB, ";
                ss << "Overhead: " << (mem.index_overhead_bytes / 1024 / 1024) << " MB";
                std::string s = ss.str();
                response = "$" + std::to_string(s.size()) + "\r\n" + s + "\r\n";
            }
            else
            {
                response = "+No Vector Engine Active\r\n";
            }
        }
        else
        {
            response = "-ERR unknown command\r\n";
        }

        // Send response
        size_t sent = 0;
        while (sent < response.size())
        {
            ssize_t n = write(c.fd, response.data() + sent, response.size() - sent);
            if (n < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                break;
            }
            sent += n;
        }
    }

    void loop()
    {
        std::vector<struct epoll_event> events(1024);
        std::unordered_map<int, Conn> conns;

        while (running_)
        {
            int n = epoll_wait(epoll_fd_, events.data(), 1024, 100);
            for (int i = 0; i < n; ++i)
            {
                int fd = events[i].data.fd;
                if (fd == event_fd_)
                {
                    uint64_t u;
                    // Cast to void to silence warn_unused_result
                    (void)read(fd, &u, sizeof(u));
                }
                else if (fd == listen_fd_)
                {
                    int conn_fd = accept(listen_fd_, nullptr, nullptr);
                    if (conn_fd >= 0)
                    {
                        set_nonblocking(conn_fd);
                        add_event(conn_fd, EPOLLIN);
                        conns[conn_fd] = Conn{conn_fd, {}, {}};
                    }
                }
                else
                {
                    if (events[i].events & EPOLLIN)
                    {
                        char buf[4096];
                        ssize_t rb = read(fd, buf, sizeof(buf));
                        if (rb > 0)
                        {
                            Conn &c = conns[fd];
                            // Mock: Trigger handler directly
                            handle_request(c);
                        }
                        else if (rb == 0 || (rb < 0 && errno != EAGAIN))
                        {
                            close(fd);
                            conns.erase(fd);
                        }
                    }
                }
            }
        }
        for (auto &p : conns)
            close(p.first);
    }
};