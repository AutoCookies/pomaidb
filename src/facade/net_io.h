#pragma once
#include <string>
#include <cstdint>
#include <sys/types.h>

namespace pomai::server::net {

    // Thiết lập file descriptor sang chế độ non-blocking
    void set_nonblocking(int fd);

    // Quản lý sự kiện Epoll (Add, Mod, Del)
    void add_event(int epoll_fd, int fd, uint32_t events);
    void mod_event(int epoll_fd, int fd, uint32_t events);
    void del_event(int epoll_fd, int fd);

    // Gửi toàn bộ dữ liệu qua socket, xử lý các lỗi EAGAIN/EWOULDBLOCK
    ssize_t send_all(int fd, const char* data, size_t len);

} // namespace pomai::server::net