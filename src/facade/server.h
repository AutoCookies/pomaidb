/*
 * src/facade/server.h
 * PomaiServer (V3) - Turbo Parser + Batch INSERT support + WhisperGrain integration
 *
 * Upgrades:
 *  - Batch INSERT: support "INSERT INTO <name> VALUES (<label>, [v...]),(...)" and fast parsing
 *  - WhisperGrain controller integrated: server computes per-query budget and reports latencies/CPU
 *  - CPU sampler thread updates WhisperGrain periodically using /proc/stat
 *
 * Notes:
 *  - This file is a self-contained server header. It depends on pomai_db and orbit APIs
 *    including budget-aware search methods (search_with_budget / search_filtered_with_budget).
 *  - Added support for: GET MEMBRANCE INFO <name>;  (returns dim, estimated num_vectors, disk size)
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
#include <mutex>
#include <cctype>
#include <exception>
#include <string_view>
#include <cstdlib> // strtof
#include <string>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <iomanip>

#include "src/core/map.h"
#include "src/core/metrics.h"
#include "src/core/config.h"
#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h" // for Tag
#include "src/ai/whispergrain.h"
#include "src/core/config.h"
#include "src/core/metrics.h"
#include "src/facade/server_utils.h"
#include "src/facade/data_supplier.h"
#include "src/facade/net_io.h"
#include "src/facade/sql_executor.h"

namespace utils = pomai::server::utils;
namespace net = pomai::server::net;

class PomaiServer
{
public:
    PomaiServer(pomai::core::PomaiMap *kv_map, pomai::core::PomaiDB *pomai_db, const pomai::config::PomaiConfig &config)
        : kv_map_(kv_map),
          pomai_db_(pomai_db),
          config_(config),
          port_(config.net.port),
          whisper_(config.whisper),
          cpu_sampler_running_(false)
    {
        start();
    }

    ~PomaiServer()
    {
        stop();
    }

    void stop()
    {
        // stop CPU sampler first
        cpu_sampler_running_.store(false);
        if (cpu_sampler_thread_.joinable())
            cpu_sampler_thread_.join();

        running_ = false;
        uint64_t u = 1;
        if (event_fd_ != -1)
        {
            ssize_t w = write(event_fd_, &u, sizeof(u));
            (void)w;
        }
        if (worker_.joinable())
            worker_.join();
        if (listen_fd_ != -1)
        {
            close(listen_fd_);
            listen_fd_ = -1;
        }
        if (epoll_fd_ != -1)
        {
            close(epoll_fd_);
            epoll_fd_ = -1;
        }
        if (event_fd_ != -1)
        {
            close(event_fd_);
            event_fd_ = -1;
        }

        std::lock_guard<std::mutex> lk(conns_mu_);
        for (auto &kv : conns_)
            close(kv.first);
        conns_.clear();

        // Best-effort snapshot on server stop (safe to call twice; main also does it).
        try
        {
            if (pomai_db_)
            {
                bool ok_schemas = pomai_db_->save_all_membrances();
                bool ok_manifest = pomai_db_->save_manifest();
                if (!ok_schemas || !ok_manifest)
                {
                    std::cerr << "[PomaiServer] Warning: snapshot on stop completed with warnings. Schemas: "
                              << (ok_schemas ? "ok" : "fail") << ", manifest: " << (ok_manifest ? "ok" : "fail") << "\n";
                }
                else
                {
                    std::clog << "[PomaiServer] Snapshot on stop completed.\n";
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[PomaiServer] Exception while snapshotting DB on stop: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[PomaiServer] Unknown exception while snapshotting DB on stop\n";
        }
    }

private:
    pomai::core::PomaiMap *kv_map_;
    pomai::core::PomaiDB *pomai_db_;
    pomai::config::PomaiConfig config_;
    int port_;
    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    int event_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mutex conns_mu_;
    pomai::server::SqlExecutor sql_executor_;

    struct Conn
    {
        int fd;
        std::vector<char> rbuf;
        std::vector<char> wbuf;
        std::string sql_current_membr;
    };

    std::unordered_map<int, Conn> conns_; // protected by conns_mu_

    // WhisperGrain controller (single instance for server)
    pomai::ai::WhisperGrain whisper_;
    // Simple frequency map for hot-query detection (string -> count)
    std::unordered_map<std::string, uint32_t> query_freq_;
    mutable std::mutex query_freq_mu_;

    // CPU sampler thread
    std::atomic<bool> cpu_sampler_running_;
    std::thread cpu_sampler_thread_;
    // previous cpu snapshot for /proc/stat sampling
    uint64_t prev_total_{0}, prev_idle_{0};

    void cpu_sampler_loop()
    {
        // initialize prev snapshot
        uint64_t idle, total;
        if (!utils::read_proc_stat(prev_idle_, prev_total_))
        {
            // cannot read procstat: set cpu to 0 and return
            whisper_.set_cpu_load(0.0f);
            return;
        }

        const std::chrono::milliseconds sample_interval(config_.server.cpu_sample_interval_ms);
        while (cpu_sampler_running_.load())
        {
            std::this_thread::sleep_for(sample_interval);
            if (!utils::read_proc_stat(idle, total))
                continue;
            uint64_t delta_idle = idle - prev_idle_;
            uint64_t delta_total = total - prev_total_;
            prev_idle_ = idle;
            prev_total_ = total;
            float cpu_percent = 0.0f;
            if (delta_total > 0)
            {
                cpu_percent = 100.0f * (1.0f - (static_cast<double>(delta_idle) / static_cast<double>(delta_total)));
                if (cpu_percent < 0.0f)
                    cpu_percent = 0.0f;
                if (cpu_percent > 100.0f)
                    cpu_percent = 100.0f;
            }
            whisper_.set_cpu_load(cpu_percent);
        }
    }

    // ---------- SQL command execution ----------
    std::string exec_sql_command(Conn &c, const std::string &raw_cmd)
    {
        // Build client state from connection
        pomai::server::ClientState state;
        state.current_membrance = c.sql_current_membr;

        // Delegate to SqlExecutor (it will trim/remove trailing ';' itself)
        std::string resp = sql_executor_.execute(pomai_db_, whisper_, state, raw_cmd);

        // Persist any changes to current membrance back into Conn
        c.sql_current_membr = state.current_membrance;
        return resp;
    }

    // send response + "<END>\n" marker
    void send_response(Conn &c, const std::string &resp)
    {
        std::string out = resp;
        utils::append_end_marker(out);
        size_t written = 0;
        const char *ptr = out.data();
        size_t left = out.size();
        while (left > 0)
        {
            ssize_t n = write(c.fd, ptr + written, left);
            if (n < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                break;
            }
            written += static_cast<size_t>(n);
            left -= static_cast<size_t>(n);
        }
    }

    void process_sql_buffer(Conn &c)
    {
        while (!c.rbuf.empty())
        {
            // 1. Kiểm tra Binary Magic Fast-Path ('PMIB' = 0x504D4942)
            if (c.rbuf.size() >= 4)
            {
                uint32_t magic;
                std::memcpy(&magic, c.rbuf.data(), 4);
                if (magic == 0x504D4942)
                {
                    // Cần ít nhất 9 bytes để đọc Header (Magic + Op + PayloadLen)
                    if (c.rbuf.size() < 9)
                        break;

                    uint32_t payload_len;
                    std::memcpy(&payload_len, c.rbuf.data() + 5, 4);

                    // Đợi cho đến khi nhận đủ toàn bộ gói tin nhị phân
                    if (c.rbuf.size() < 9 + payload_len)
                        break;

                    // Thực thi Fast-path không qua Parser văn bản
                    std::string resp = sql_executor_.execute_binary_insert(pomai_db_, c.rbuf.data() + 9, payload_len);
                    send_response(c, resp);

                    // Xóa gói tin đã xử lý khỏi buffer
                    c.rbuf.erase(c.rbuf.begin(), c.rbuf.begin() + 9 + payload_len);
                    continue;
                }
            }

            // 2. Fallback: Giao thức văn bản (Dựa trên dấu chấm phẩy)
            auto it = std::find(c.rbuf.begin(), c.rbuf.end(), ';');
            if (it != c.rbuf.end())
            {
                size_t pos = std::distance(c.rbuf.begin(), it);
                std::string cmd(c.rbuf.begin(), c.rbuf.begin() + pos + 1);
                std::string resp = exec_sql_command(c, cmd);
                send_response(c, resp);

                c.rbuf.erase(c.rbuf.begin(), c.rbuf.begin() + pos + 1);
                continue;
            }

            // Không tìm thấy lệnh hoàn chỉnh nào, thoát vòng lặp để đợi thêm dữ liệu
            break;
        }
    }

    void start()
    {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
            throw std::runtime_error("socket failed");

        int opt = 1;
        (void)setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, static_cast<socklen_t>(sizeof(opt)));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
            throw std::runtime_error("bind failed");

        if (listen(listen_fd_, config_.server.backlog) < 0)
            throw std::runtime_error("listen failed");

        net::set_nonblocking(listen_fd_);

        epoll_fd_ = epoll_create1(0);
        if (epoll_fd_ < 0)
            throw std::runtime_error("epoll_create1 failed");

        event_fd_ = static_cast<int>(eventfd(0, EFD_NONBLOCK));
        if (event_fd_ < 0)
            throw std::runtime_error("eventfd failed");

        net::add_event(epoll_fd_, listen_fd_, EPOLLIN);
        net::add_event(epoll_fd_, event_fd_, EPOLLIN);

        // Start CPU sampler thread
        cpu_sampler_running_.store(true);
        cpu_sampler_thread_ = std::thread([this]()
                                          { cpu_sampler_loop(); });

        running_ = true;
        worker_ = std::thread(&PomaiServer::loop, this);
    }

    void loop()
    {
        std::vector<struct epoll_event> events(config_.server.max_events);
        while (running_)
        {
            int n = epoll_wait(epoll_fd_, events.data(), static_cast<int>(events.size()), config_.server.epoll_timeout_ms);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                break;
            }

            for (int i = 0; i < n; ++i)
            {
                int fd = events[i].data.fd;
                if (fd == event_fd_)
                { /* ignore */
                }
                else if (fd == listen_fd_)
                {
                    int conn_fd = accept(listen_fd_, nullptr, nullptr);
                    if (conn_fd >= 0)
                    {
                        // [FIXED]: Sử dụng NetIO đã tách
                        net::set_nonblocking(conn_fd);
                        net::add_event(epoll_fd_, conn_fd, EPOLLIN); // Cần epoll_fd_ ở đây

                        std::lock_guard<std::mutex> lk(conns_mu_);
                        conns_.emplace(conn_fd, Conn{conn_fd, {}, {}, ""});
                    }
                }
                else
                {
                    if (events[i].events & EPOLLIN)
                    {
                        char buf[pomai::config::SERVER_READ_BUFFER];
                        ssize_t rb = read(fd, buf, sizeof(buf));
                        if (rb > 0)
                        {
                            std::lock_guard<std::mutex> lk(conns_mu_);
                            auto it = conns_.find(fd);
                            if (it != conns_.end())
                            {
                                it->second.rbuf.insert(it->second.rbuf.end(), buf, buf + rb);
                                process_sql_buffer(it->second);
                            }
                        }
                        else if (rb == 0 || (rb < 0 && errno != EAGAIN))
                        {
                            std::lock_guard<std::mutex> lk(conns_mu_);
                            net::del_event(epoll_fd_, fd);
                            close(fd);
                            conns_.erase(fd);
                        }
                    }
                }
            }
        }
    }
};