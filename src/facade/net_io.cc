#include "src/facade/net_io.h"
#include <fcntl.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdexcept>

namespace pomai::server::net {

void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0)
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void add_event(int epoll_fd, int fd, uint32_t events) {
    struct epoll_event ev{};
    ev.events = events;
    ev.data.fd = fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) != 0) {
        throw std::runtime_error(std::string("epoll_ctl add failed: ") + std::strerror(errno));
    }
}

void mod_event(int epoll_fd, int fd, uint32_t events) {
    struct epoll_event ev{};
    ev.events = events;
    ev.data.fd = fd;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &ev) != 0) {
        // Log lỗi nhẹ để không làm sập server
    }
}

void del_event(int epoll_fd, int fd) {
    epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
}

ssize_t send_all(int fd, const char* data, size_t len) {
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, data + written, len - written);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue; // Thử lại ngay lập tức cho hiệu suất tối đa
            }
            return -1; // Lỗi nghiêm trọng
        }
        written += static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(written);
}

} // namespace pomai::server::net