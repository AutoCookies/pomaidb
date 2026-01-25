#pragma once
// src/facade/tcp_listener.h
//
// Simple, robust TCP listener for Pomai server.
// - Accepts incoming TCP connections on configured port.
// - Spawns a short-lived thread per connection to service text-line protocol.
// - Each received newline-terminated command is passed to the provided handler.
// - Uses non-blocking accept loop with poll() so stop() can shut down promptly.

#include <functional>
#include <string>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <mutex>
#include <cstdint>

namespace pomai::server::net
{
    using CmdHandler = std::function<std::string(const std::string &)>;

    class TcpListener
    {
    public:
        // port: TCP port to bind
        // handler: called for each received newline-terminated command; return value is written back to client
        TcpListener(uint16_t port, CmdHandler handler) noexcept;
        ~TcpListener();

        // Start listening (spawns accept thread). Returns true on success.
        bool start();

        // Stop listening and join background threads. Safe to call multiple times.
        void stop();

        bool running() const noexcept { return running_.load(); }
        uint16_t port() const noexcept { return port_; }

    private:
        void accept_loop();
        void client_worker(int client_fd);

        uint16_t port_;
        CmdHandler handler_;

        int listen_fd_ = -1;
        std::thread accept_thread_;
        std::atomic<bool> running_{false};

        // track active client threads so we can join them on stop
        std::mutex clients_mu_;
        std::vector<std::thread> client_threads_;
    };

} // namespace pomai::server::net