#include "src/facade/tcp_listener.h"
#include "src/facade/net_io.h"
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include <poll.h>

namespace pomai::server::net
{
    static bool send_all_safe(int fd, const void *data, size_t len)
    {
        const char *ptr = static_cast<const char *>(data);
        size_t remaining = len;
        while (remaining > 0)
        {
            ssize_t n = ::send(fd, ptr, remaining, MSG_NOSIGNAL);
            if (n <= 0)
            {
                if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
                {
                    continue;
                }
                return false;
            }
            ptr += n;
            remaining -= n;
        }
        return true;
    }

    static void prepare_socket(int fd)
    {
        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
        setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    TcpListener::TcpListener(uint16_t port, StreamHandler handler) noexcept
        : port_(port), handler_(std::move(handler)) {}

    TcpListener::~TcpListener() { stop(); }

    bool TcpListener::start()
    {
        if (running_.load())
            return true;

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
            return false;

        prepare_socket(listen_fd_);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port_);

        if (::bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) != 0)
        {
            ::close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }

        if (::listen(listen_fd_, 4096) != 0)
        {
            ::close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }

        epoll_fd_ = epoll_create1(0);
        if (epoll_fd_ < 0)
        {
            ::close(listen_fd_);
            listen_fd_ = -1;
            return false;
        }

        struct epoll_event ev{};
        ev.events = EPOLLIN | EPOLLEXCLUSIVE;
        ev.data.fd = listen_fd_;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0)
        {
            ::close(epoll_fd_);
            ::close(listen_fd_);
            epoll_fd_ = -1;
            listen_fd_ = -1;
            return false;
        }

        running_.store(true);
        try
        {
            accept_thread_ = std::thread(&TcpListener::accept_loop, this);
        }
        catch (...)
        {
            running_.store(false);
            ::close(epoll_fd_);
            ::close(listen_fd_);
            epoll_fd_ = -1;
            listen_fd_ = -1;
            return false;
        }
        return true;
    }

    void TcpListener::stop()
    {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false))
            return;

        try
        {
            if (epoll_fd_ >= 0)
            {
                ::close(epoll_fd_);
                epoll_fd_ = -1;
            }

            if (listen_fd_ >= 0)
            {
                ::shutdown(listen_fd_, SHUT_RDWR);
                ::close(listen_fd_);
                listen_fd_ = -1;
            }

            if (accept_thread_.joinable())
                accept_thread_.join();

            while (active_clients_.load() > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        catch (...)
        {
        }
    }

    void TcpListener::accept_loop()
    {
        struct epoll_event events[16];
        while (running_.load())
        {
            int nfds = epoll_wait(epoll_fd_, events, 16, 500);
            if (nfds < 0)
            {
                if (errno == EINTR)
                    continue;
                break;
            }
            for (int i = 0; i < nfds; ++i)
            {
                if (events[i].data.fd == listen_fd_)
                {
                    while (running_.load())
                    {
                        sockaddr_in client_addr;
                        socklen_t client_len = sizeof(client_addr);
                        int client_fd = ::accept4(listen_fd_, (struct sockaddr *)&client_addr, &client_len, SOCK_NONBLOCK | SOCK_CLOEXEC);

                        if (client_fd < 0)
                        {
                            if (errno == EAGAIN || errno == EWOULDBLOCK)
                                break;
                            if (errno == EINTR)
                                continue;
                            break;
                        }

                        int flag = 1;
                        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));

                        try
                        {
                            active_clients_.fetch_add(1);
                            std::thread(&TcpListener::client_worker, this, client_fd).detach();
                        }
                        catch (...)
                        {
                            ::close(client_fd);
                            active_clients_.fetch_sub(1);
                        }
                    }
                }
            }
        }
    }

    void TcpListener::client_worker(int fd)
    {
        const size_t CAP = 8 * 1024 * 1024;
        std::vector<char> buf(CAP);
        size_t rpos = 0;
        size_t wpos = 0;

        try
        {
            while (running_.load())
            {
                if (rpos > 0)
                {
                    if (rpos >= wpos)
                    {
                        rpos = wpos = 0;
                    }
                    else if (CAP - wpos < 4096)
                    {
                        size_t len = wpos - rpos;
                        std::memmove(buf.data(), buf.data() + rpos, len);
                        rpos = 0;
                        wpos = len;
                    }
                }

                if (wpos == CAP && rpos == 0)
                    break;

                struct pollfd pfd{fd, POLLIN, 0};
                int ret = poll(&pfd, 1, 1000);
                if (ret < 0)
                {
                    if (errno == EINTR)
                        continue;
                    break;
                }
                if (ret == 0)
                    continue;

                ssize_t n = ::recv(fd, buf.data() + wpos, CAP - wpos, 0);
                if (n <= 0)
                    break;
                wpos += n;

                while (rpos < wpos)
                {
                    std::string_view view(buf.data() + rpos, wpos - rpos);
                    size_t consumed = 0;
                    std::string resp;
                    try
                    {
                        auto out = handler_(view);
                        consumed = out.first;
                        resp = std::move(out.second);
                    }
                    catch (...)
                    {
                        consumed = wpos - rpos;
                        resp = "";
                    }

                    if (consumed == 0)
                        break;

                    if (!resp.empty())
                    {
                        if (!send_all_safe(fd, resp.data(), resp.size()))
                            goto cleanup;
                    }
                    rpos += consumed;
                }
            }
        }
        catch (...)
        {
        }

    cleanup:
        try
        {
            ::shutdown(fd, SHUT_RDWR);
        }
        catch (...)
        {
        }
        ::close(fd);
        active_clients_.fetch_sub(1);
    }
}