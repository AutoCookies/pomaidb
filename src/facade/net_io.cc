#include "src/facade/net_io.h"
#include <fcntl.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <iostream>
#include <poll.h>

namespace pomai::server::net
{

    void set_nonblocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0)
        {
            std::cerr << "[net_io] set_nonblocking: F_GETFL failed: " << std::strerror(errno) << "\n";
            return;
        }
        if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0)
        {
            std::cerr << "[net_io] set_nonblocking: F_SETFL failed: " << std::strerror(errno) << "\n";
        }
        else
        {
            std::clog << "[net_io] fd=" << fd << " set to non-blocking\n";
        }
    }

    void add_event(int epoll_fd, int fd, uint32_t events)
    {
        struct epoll_event ev{};
        ev.events = events;
        ev.data.fd = fd;
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) != 0)
        {
            std::string err = std::strerror(errno);
            throw std::runtime_error(std::string("epoll_ctl add failed: ") + err);
        }
        std::clog << "[net_io] epoll ADD fd=" << fd << " events=" << events << "\n";
    }

    void mod_event(int epoll_fd, int fd, uint32_t events)
    {
        struct epoll_event ev{};
        ev.events = events;
        ev.data.fd = fd;
        if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &ev) != 0)
        {
            // Log nhẹ để không làm sập server
            std::cerr << "[net_io] epoll MOD failed for fd=" << fd << ": " << std::strerror(errno) << "\n";
        }
        else
        {
            std::clog << "[net_io] epoll MOD fd=" << fd << " events=" << events << "\n";
        }
    }

    void del_event(int epoll_fd, int fd)
    {
        epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
        std::clog << "[net_io] epoll DEL fd=" << fd << "\n";
    }

    ssize_t send_all(int fd, const char *data, size_t len)
    {
        size_t written = 0;
        while (written < len)
        {
            ssize_t n = ::write(fd, data + written, len - written);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    // FIX: Chờ sự kiện POLLOUT thay vì ngủ
                    struct pollfd pfd;
                    pfd.fd = fd;
                    pfd.events = POLLOUT;
                    // Timeout 1000ms để tránh treo vĩnh viễn nếu client chết
                    int ret = poll(&pfd, 1, 1000);

                    if (ret > 0)
                        continue; // Socket đã sẵn sàng, write tiếp
                    if (ret == 0)
                        return -1; // Timeout -> Client quá chậm, cắt kết nối
                    return -1;     // Error poll
                }
                return -1; // Lỗi fatal khác
            }
            written += n;
        }
        return written;
    }

} // namespace pomai::server::net