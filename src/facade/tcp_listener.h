#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <cstdint>

namespace pomai::server::net
{
    using StreamHandler = std::function<std::pair<size_t, std::string>(std::string_view)>;

    class TcpListener
    {
    public:
        TcpListener(uint16_t port, StreamHandler handler) noexcept;
        ~TcpListener();

        bool start();
        void stop();
        bool running() const noexcept { return running_.load(); }

    private:
        void accept_loop();
        void client_worker(int client_fd);

        uint16_t port_;
        StreamHandler handler_;

        int listen_fd_ = -1;
        int epoll_fd_ = -1;
        std::thread accept_thread_;
        std::atomic<bool> running_{false};

        std::mutex clients_mu_;
        std::atomic<size_t> active_clients_{0};
    };
}