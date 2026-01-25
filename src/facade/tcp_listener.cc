// src/facade/tcp_listener.cc
//
// Implementation of TcpListener: minimal, production-minded, predictable behavior.
//
// Design notes:
// - Accept loop uses poll() with short timeout so stop() can interrupt quickly.
// - Each client handled in its own thread (detached-joinable via vector) to keep logic simple
//   and avoid complex epoll per-connection state while preserving isolation.
// - Per-connection thread reads from socket into fixed buffer and dispatches newline-terminated
//   commands to handler. Response is sent back as-is (handler output).
// - Uses blocking read on client FD (no busy-loop) — threads are cheap for connections in typical DB use.
// - All resources are closed cleanly on stop().

#include "src/facade/tcp_listener.h"
#include "src/facade/net_io.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#include <cstring>
#include <iostream>
#include <mutex>

              namespace pomai::server::net
{

    static inline void set_socket_reuseaddr(int fd)
    {
        int v = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &v, sizeof(v));
#ifdef SO_REUSEPORT
        setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &v, sizeof(v));
#endif
    }

    TcpListener::TcpListener(uint16_t port, CmdHandler handler) noexcept
        : port_(port), handler_(std::move(handler)), listen_fd_(-1)
    {
    }

    TcpListener::~TcpListener()
    {
        stop();
    }

    bool TcpListener::start()
    {
        if (running_.load())
            return true;

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
        {
            std::cerr << "[TcpListener] socket() failed: " << strerror(errno) << "\n";
            return false;
        }

        set_socket_reuseaddr(listen_fd_);

        // Bind to 0.0.0.0:port
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port_);

        if (bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0)
        {
            std::cerr << "[TcpListener] bind() failed: " << strerror(errno) << "\n";
            ::close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }

        if (listen(listen_fd_, 128) != 0)
        {
            std::cerr << "[TcpListener] listen() failed: " << strerror(errno) << "\n";
            ::close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }

        // Set non-blocking accept so poll can be used.
        int flags = fcntl(listen_fd_, F_GETFL, 0);
        if (flags >= 0)
            fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);

        running_.store(true, std::memory_order_release);
        accept_thread_ = std::thread(&TcpListener::accept_loop, this);
        return true;
    }

    void TcpListener::stop()
    {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false))
        {
            // already stopped
            if (listen_fd_ >= 0)
            {
                ::close(listen_fd_);
                listen_fd_ = -1;
            }
            return;
        }

        // Close listening socket to wake up poll/accept
        if (listen_fd_ >= 0)
        {
            ::close(listen_fd_);
            listen_fd_ = -1;
        }

        // join accept thread
        if (accept_thread_.joinable())
            accept_thread_.join();

        // join client threads
        {
            std::lock_guard<std::mutex> lk(clients_mu_);
            for (auto &t : client_threads_)
            {
                if (t.joinable())
                    t.join();
            }
            client_threads_.clear();
        }
    }

    void TcpListener::accept_loop()
    {
        const int POLL_TIMEOUT_MS = 250; // short, responsive shutdown
        while (running_.load(std::memory_order_acquire))
        {
            struct pollfd pfd;
            pfd.fd = listen_fd_;
            pfd.events = POLLIN;

            int ret = poll(&pfd, 1, POLL_TIMEOUT_MS);
            if (ret < 0)
            {
                if (errno == EINTR)
                    continue;
                std::cerr << "[TcpListener] poll() failed: " << strerror(errno) << "\n";
                break;
            }
            if (ret == 0)
                continue; // timeout

            if (pfd.revents & POLLIN)
            {
                struct sockaddr_in peer;
                socklen_t plen = sizeof(peer);
                int client_fd = ::accept(listen_fd_, reinterpret_cast<struct sockaddr *>(&peer), &plen);
                if (client_fd < 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                        continue;
                    std::cerr << "[TcpListener] accept() failed: " << strerror(errno) << "\n";
                    continue;
                }

                // Make client socket blocking (simpler per-thread model)
                int cflags = fcntl(client_fd, F_GETFL, 0);
                if (cflags >= 0)
                    fcntl(client_fd, F_SETFL, cflags & ~O_NONBLOCK);

                // Spawn client worker thread
                std::lock_guard<std::mutex> lk(clients_mu_);
                client_threads_.emplace_back(&TcpListener::client_worker, this, client_fd);
            }
        }
    }

    void TcpListener::client_worker(int client_fd)
    {
        // Ensure socket closed at end
        int fd = client_fd;
        constexpr size_t BUF_SZ = 8192;
        std::string inbuf;
        inbuf.reserve(4096);
        char tmp[BUF_SZ];

        while (true)
        {
            ssize_t n = ::recv(fd, tmp, BUF_SZ, 0);
            if (n == 0)
            {
                // connection closed by peer
                break;
            }
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    // no data; continue
                    continue;
                }
                // fatal error
                break;
            }

            inbuf.append(tmp, static_cast<size_t>(n));

            // Process newline-terminated commands
            size_t pos = 0;
            while (true)
            {
                size_t nl = inbuf.find('\n', pos);
                if (nl == std::string::npos)
                {
                    // keep remainder; break to read more
                    if (pos > 0)
                        inbuf.erase(0, pos);
                    break;
                }

                std::string line = inbuf.substr(0, nl);
                // remove optional '\r'
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();

                // Dispatch handler (safely)
                std::string resp;
                try
                {
                    resp = handler_(line);
                }
                catch (const std::exception &e)
                {
                    resp = std::string("ERR: ") + e.what() + "\n";
                }
                catch (...)
                {
                    resp = std::string("ERR: unknown\n");
                }

                // Ensure trailing newline in response
                if (resp.empty() || resp.back() != '\n')
                    resp.push_back('\n');

                // Best-effort send (blocking). Use send_all helper to handle partial writes.
                ssize_t sent = send_all(fd, resp.data(), resp.size());
                (void)sent;

                // erase processed part including newline
                inbuf.erase(0, nl + 1);
                pos = 0;
            }
        }

        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
    }

} // namespace pomai::server::net