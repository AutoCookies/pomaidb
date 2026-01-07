#pragma once
// facade/server.h - Pomai Wire Protocol (PWP) v1 implementation (enhanced)
// ... (header comments unchanged) ...

#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h> // writev
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

#include "core/map.h"
#include "core/metrics.h"
#include "core/config.h"
#include "ai/vector_index.h" // <- new

struct alignas(16) PomaiHeader
{
    uint8_t magic;     // 'P'
    uint8_t op;        // opcode
    uint16_t status;   // status (response)
    uint32_t klen;     // key length (network order on wire)
    uint32_t vlen;     // value length
    uint32_t reserved; // reserved/checksum
};

enum OpCode : uint8_t
{
    OP_GET = 1,
    OP_SET = 2,
    OP_DEL = 3,
    OP_PING = 99,
    OP_VSET = 10,
    OP_VSEARCH = 11
};
enum StatusCode : uint16_t
{
    STATUS_OK = 0,
    STATUS_MISS = 1,
    STATUS_FULL = 2,
    STATUS_ERR = 3
};

class PomaiServer
{
private:
    int server_fd{-1};
    int epoll_fd{-1};
    int event_fd{-1}; // for graceful shutdown signalling
    PomaiMap *engine_map;
    std::unique_ptr<VectorIndex> vec_index_; // <- new
    std::atomic<bool> running{true};

    struct Connection
    {
        int fd;
        std::vector<char> inbuf;
        std::vector<char> outbuf; // pending bytes to send
        Connection(int fd_) : fd(fd_)
        {
            inbuf.reserve(pomai::config::PWP_INBUF_RESERVE);
            outbuf.reserve(pomai::config::PWP_OUTBUF_RESERVE);
        }
    };

    std::unordered_map<int, std::unique_ptr<Connection>> conns;

    static void set_nonblocking_fd(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags == -1)
            flags = 0;
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    static void set_socket_options(int fd)
    {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
    }

    // Modify epoll events for this fd to add/remove EPOLLOUT depending on whether conn->outbuf is empty.
    void update_epoll_out(int fd, bool want_out)
    {
        struct epoll_event ev{};
        ev.data.fd = fd;
        // Ensure both branches of the conditional have the same non-enum type to silence -Wextra
        ev.events = EPOLLIN | EPOLLET | (want_out ? static_cast<uint32_t>(EPOLLOUT) : static_cast<uint32_t>(0));
        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &ev) != 0)
        {
            if (errno == ENOENT)
            {
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev);
            }
        }
    }

    void queue_response(Connection *conn, const PomaiHeader &hdr_host, const char *val_ptr, uint32_t val_len)
    {
        PomaiHeader wire{};
        wire.magic = pomai::config::PWP_MAGIC;
        wire.op = hdr_host.op;
        wire.status = htons(hdr_host.status);
        wire.klen = htonl(hdr_host.klen);
        wire.vlen = htonl(hdr_host.vlen);
        wire.reserved = htonl(hdr_host.reserved);

        size_t old_out = conn->outbuf.size();
        conn->outbuf.resize(old_out + sizeof(PomaiHeader) + val_len);
        memcpy(conn->outbuf.data() + old_out, &wire, sizeof(PomaiHeader));
        if (val_len > 0 && val_ptr)
            memcpy(conn->outbuf.data() + old_out + sizeof(PomaiHeader), val_ptr, val_len);

        update_epoll_out(conn->fd, true);
    }

    bool flush_out(Connection *conn)
    {
        while (!conn->outbuf.empty())
        {
            ssize_t n = send(conn->fd, conn->outbuf.data(), conn->outbuf.size(), 0);
            if (n > 0)
            {
                conn->outbuf.erase(conn->outbuf.begin(), conn->outbuf.begin() + static_cast<ptrdiff_t>(n));
                continue;
            }
            else if (n == -1)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        update_epoll_out(conn->fd, false);
        return true;
    }

    void send_response_buffered(Connection *conn, uint8_t op, uint16_t status, const char *val_ptr, uint32_t val_len)
    {
        PomaiHeader hdr{};
        hdr.op = op;
        hdr.status = status;
        hdr.klen = 0;
        hdr.vlen = val_len;
        hdr.reserved = 0;
        queue_response(conn, hdr, val_ptr, val_len);
    }

    // parse and execute; extended to handle VSET and VSEARCH
    void execute_command(Connection *conn, const PomaiHeader &hdr_host, const char *body_ptr)
    {
        if (hdr_host.klen > pomai::config::SERVER_MAX_PART_BYTES || hdr_host.vlen > pomai::config::SERVER_MAX_PART_BYTES)
        {
            send_response_buffered(conn, hdr_host.op, STATUS_ERR, nullptr, 0);
            return;
        }
        uint64_t total = static_cast<uint64_t>(hdr_host.klen) + static_cast<uint64_t>(hdr_host.vlen);
        if (total > pomai::config::PWP_MAX_PACKET_SIZE)
        {
            send_response_buffered(conn, hdr_host.op, STATUS_ERR, nullptr, 0);
            return;
        }

        if (hdr_host.op == OP_SET)
        {
            const char *kptr = body_ptr;
            const char *vptr = body_ptr + hdr_host.klen;
            bool ok = engine_map->put(kptr, hdr_host.klen, vptr, hdr_host.vlen);
            if (ok)
            {
                PomaiMetrics::puts.fetch_add(1, std::memory_order_relaxed);
                send_response_buffered(conn, OP_SET, STATUS_OK, nullptr, 0);
            }
            else
            {
                PomaiMetrics::arena_alloc_fails.fetch_add(1, std::memory_order_relaxed);
                send_response_buffered(conn, OP_SET, STATUS_FULL, nullptr, 0);
            }
        }
        else if (hdr_host.op == OP_GET)
        {
            uint32_t out_len = 0;
            const char *res = engine_map->get(body_ptr, hdr_host.klen, &out_len);
            if (res && out_len > 0)
            {
                PomaiMetrics::hits.fetch_add(1, std::memory_order_relaxed);
                send_response_buffered(conn, OP_GET, STATUS_OK, res, out_len);
            }
            else
            {
                PomaiMetrics::misses.fetch_add(1, std::memory_order_relaxed);
                send_response_buffered(conn, OP_GET, STATUS_MISS, nullptr, 0);
            }
        }
        else if (hdr_host.op == OP_DEL)
        {
            std::string key(body_ptr, hdr_host.klen);
            bool removed = engine_map->erase(key.c_str());
            send_response_buffered(conn, OP_DEL, removed ? STATUS_OK : STATUS_MISS, nullptr, 0);
        }
        else if (hdr_host.op == OP_VSET)
        {
            // VSET: body = [key bytes][vector bytes (float32 array)]
            const char *kptr = body_ptr;
            const char *vptr = body_ptr + hdr_host.klen;
            bool ok = engine_map->put(kptr, hdr_host.klen, vptr, hdr_host.vlen);
            if (ok)
            {
                // mark seed as vector
                Seed *s = engine_map->find_seed(kptr, hdr_host.klen);
                if (s)
                {
                    s->type = Seed::OBJ_VECTOR;
                }
                PomaiMetrics::puts.fetch_add(1, std::memory_order_relaxed);
                send_response_buffered(conn, OP_VSET, STATUS_OK, nullptr, 0);
            }
            else
            {
                PomaiMetrics::arena_alloc_fails.fetch_add(1, std::memory_order_relaxed);
                send_response_buffered(conn, OP_VSET, STATUS_FULL, nullptr, 0);
            }
        }
        else if (hdr_host.op == OP_VSEARCH)
        {
            // VSEARCH request body: [4B topk (network order)][query float32 bytes...]
            if (!vec_index_)
            {
                send_response_buffered(conn, OP_VSEARCH, STATUS_ERR, nullptr, 0);
                return;
            }
            if (hdr_host.vlen < 4)
            {
                send_response_buffered(conn, OP_VSEARCH, STATUS_ERR, nullptr, 0);
                return;
            }
            uint32_t topk_net = 0;
            memcpy(&topk_net, body_ptr, 4);
            uint32_t topk = ntohl(topk_net);
            if (topk == 0)
            {
                send_response_buffered(conn, OP_VSEARCH, STATUS_ERR, nullptr, 0);
                return;
            }
            size_t query_bytes = hdr_host.vlen - 4;
            if (query_bytes % sizeof(float) != 0)
            {
                send_response_buffered(conn, OP_VSEARCH, STATUS_ERR, nullptr, 0);
                return;
            }
            size_t dim = query_bytes / sizeof(float);
            const float *query = reinterpret_cast<const float *>(body_ptr + 4);

            // perform search
            std::vector<char> results = vec_index_->search(query, dim, topk);
            if (results.empty())
            {
                // send OK with empty body (no matches)
                send_response_buffered(conn, OP_VSEARCH, STATUS_OK, nullptr, 0);
            }
            else
            {
                send_response_buffered(conn, OP_VSEARCH, STATUS_OK, results.data(), static_cast<uint32_t>(results.size()));
            }
        }
        else if (hdr_host.op == OP_PING)
        {
            const char pong[] = "PONG";
            send_response_buffered(conn, OP_PING, STATUS_OK, pong, sizeof(pong) - 1);
        }
        else
        {
            send_response_buffered(conn, hdr_host.op, STATUS_ERR, nullptr, 0);
        }
    }

    size_t try_consume_one_frame(Connection *conn)
    {
        if (conn->inbuf.size() < sizeof(PomaiHeader))
            return 0;

        PomaiHeader wire_hdr;
        std::memcpy(&wire_hdr, conn->inbuf.data(), sizeof(PomaiHeader));
        if (wire_hdr.magic != pomai::config::PWP_MAGIC)
            return SIZE_MAX;

        PomaiHeader hdr_host{};
        hdr_host.magic = wire_hdr.magic;
        hdr_host.op = wire_hdr.op;
        hdr_host.status = ntohs(wire_hdr.status);
        hdr_host.klen = ntohl(wire_hdr.klen);
        hdr_host.vlen = ntohl(wire_hdr.vlen);
        hdr_host.reserved = ntohl(wire_hdr.reserved);

        if (hdr_host.klen > pomai::config::SERVER_MAX_PART_BYTES || hdr_host.vlen > pomai::config::SERVER_MAX_PART_BYTES)
            return SIZE_MAX;

        uint64_t body_len = static_cast<uint64_t>(hdr_host.klen) + static_cast<uint64_t>(hdr_host.vlen);
        if (body_len > pomai::config::PWP_MAX_PACKET_SIZE)
            return SIZE_MAX;

        size_t total_len = sizeof(PomaiHeader) + static_cast<size_t>(body_len);
        if (conn->inbuf.size() < total_len)
            return 0;

        const char *body_ptr = conn->inbuf.data() + sizeof(PomaiHeader);
        execute_command(conn, hdr_host, body_ptr);

        return total_len;
    }

public:
    PomaiServer(PomaiMap *map, int port) : engine_map(map)
    {
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == -1)
            throw std::runtime_error("socket failed");

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
        set_nonblocking_fd(server_fd);

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (bind(server_fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0)
            throw std::runtime_error("bind failed");
        if (listen(server_fd, pomai::config::PWP_LISTEN_BACKLOG) != 0)
            throw std::runtime_error("listen failed");

        epoll_fd = epoll_create1(0);
        if (epoll_fd == -1)
            throw std::runtime_error("epoll_create1 failed");

        event_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
        if (event_fd == -1)
            throw std::runtime_error("eventfd failed");

        struct epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.fd = server_fd;
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev) != 0)
            throw std::runtime_error("epoll_ctl ADD server_fd");

        ev.events = EPOLLIN;
        ev.data.fd = event_fd;
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, event_fd, &ev) != 0)
            throw std::runtime_error("epoll_ctl ADD event_fd");

        // initialize vector index (new)
        vec_index_.reset(new VectorIndex(engine_map));
    }

    ~PomaiServer()
    {
        running.store(false);
        if (server_fd != -1)
            close(server_fd);
        if (event_fd != -1)
            close(event_fd);
        if (epoll_fd != -1)
            close(epoll_fd);
    }

    void stop()
    {
        uint64_t one = 1;
        running.store(false);
        if (event_fd != -1)
        {
            ssize_t rr = write(event_fd, &one, sizeof(one));
            (void)rr; // use the return value (silence warn_unused_result) but don't treat as fatal
        }
    }

    void run()
    {
        std::vector<struct epoll_event> events(pomai::config::SERVER_MAX_EVENTS);
        std::cerr << "[Pomai] server starting; pid=" << getpid() << "\n";

        while (running.load())
        {
            int nfds = epoll_wait(epoll_fd, events.data(), static_cast<int>(events.size()), -1);
            if (nfds < 0)
            {
                if (errno == EINTR)
                    continue;
                perror("epoll_wait");
                break;
            }

            for (int i = 0; i < nfds; ++i)
            {
                int fd = events[i].data.fd;
                uint32_t ev = events[i].events;

                if (fd == server_fd)
                {
                    while (true)
                    {
                        struct sockaddr_in peer{};
                        socklen_t plen = sizeof(peer);
                        int conn_sock = accept(server_fd, reinterpret_cast<struct sockaddr *>(&peer), &plen);
                        if (conn_sock == -1)
                        {
                            if (errno == EAGAIN || errno == EWOULDBLOCK)
                                break;
                            perror("accept");
                            break;
                        }
                        set_nonblocking_fd(conn_sock);
                        set_socket_options(conn_sock);

                        struct epoll_event client_ev{};
                        client_ev.events = EPOLLIN | EPOLLET;
                        client_ev.data.fd = conn_sock;
                        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, conn_sock, &client_ev) != 0)
                        {
                            perror("epoll_ctl ADD conn");
                            close(conn_sock);
                            continue;
                        }
                        conns.emplace(conn_sock, std::make_unique<Connection>(conn_sock));
                    }
                }
                else if (fd == event_fd)
                {
                    uint64_t v = 0;
                    ssize_t rr = read(event_fd, &v, sizeof(v));
                    (void)rr; // consume return to silence warn_unused_result
                    running.store(false);
                    break;
                }
                else
                {
                    auto it = conns.find(fd);
                    if (it == conns.end())
                    {
                        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                        close(fd);
                        continue;
                    }
                    Connection *conn = it->second.get();

                    if (ev & EPOLLIN)
                    {
                        bool close_conn = false;
                        while (true)
                        {
                            char tmp[4096];
                            ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
                            if (n > 0)
                            {
                                conn->inbuf.insert(conn->inbuf.end(), tmp, tmp + n);
                                if (conn->inbuf.size() > pomai::config::SERVER_MAX_COMMAND_BYTES)
                                {
                                    close_conn = true;
                                    break;
                                }
                                while (true)
                                {
                                    size_t consumed = try_consume_one_frame(conn);
                                    if (consumed == 0)
                                        break;
                                    if (consumed == SIZE_MAX)
                                    {
                                        close_conn = true;
                                        break;
                                    }
                                    conn->inbuf.erase(conn->inbuf.begin(), conn->inbuf.begin() + static_cast<ptrdiff_t>(consumed));
                                }
                                if (close_conn)
                                    break;
                                continue;
                            }
                            else if (n == 0)
                            {
                                close_conn = true;
                                break;
                            }
                            else
                            {
                                if (errno == EAGAIN || errno == EWOULDBLOCK)
                                    break;
                                close_conn = true;
                                break;
                            }
                        }
                        if (close_conn)
                        {
                            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                            close(fd);
                            conns.erase(it);
                            continue;
                        }
                    }

                    if (ev & EPOLLOUT)
                    {
                        if (!flush_out(conn))
                        {
                            epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                            close(fd);
                            conns.erase(it);
                            continue;
                        }
                    }

                    if (ev & (EPOLLHUP | EPOLLERR))
                    {
                        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
                        close(fd);
                        conns.erase(it);
                        continue;
                    }
                }
            }
        }

        for (auto &p : conns)
            close(p.first);
        conns.clear();
        std::cerr << "[Pomai] server shutting down\n";
    }
};