#include "src/ai/network_cortex.h"

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>
#include <poll.h>
#include <random>
#include <thread>
#include <cassert>
#include <algorithm>
#include <functional>
#include <mutex>

#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define htobe64(x) OSSwapHostToBigInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#elif defined(__linux__)
#include <endian.h>
#else
#include <byteswap.h>
#define htobe64(x) __builtin_bswap64(x)
#define be64toh(x) __builtin_bswap64(x)
#endif

namespace pomai::ai::orbit
{
    static inline uint64_t now_ms()
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    }

    NetworkCortex::NetworkCortex(const pomai::config::NetworkCortexConfig &cfg)
        : cfg_(cfg)
    {
        node_id_ = make_instance_node_id();
    }

    NetworkCortex::~NetworkCortex()
    {
        stop();
    }

    bool NetworkCortex::start()
    {
        if (running_)
            return true;

        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0)
        {
            return false;
        }

        int opt = 1;
        setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
        setsockopt(sockfd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
        setsockopt(sockfd_, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(cfg_.udp_port);

        if (bind(sockfd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        {
            close(sockfd_);
            sockfd_ = -1;
            return false;
        }

        running_ = true;
        listener_thread_ = std::thread(&NetworkCortex::listen_loop, this);
        pulse_thread_ = std::thread(&NetworkCortex::pulse_loop, this);

        return true;
    }

    void NetworkCortex::stop()
    {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false))
        {
            return;
        }

        if (sockfd_ != -1)
        {
            close(sockfd_);
            sockfd_ = -1;
        }

        if (listener_thread_.joinable())
            listener_thread_.join();
        if (pulse_thread_.joinable())
            pulse_thread_.join();
    }

    void NetworkCortex::listen_loop()
    {
        uint8_t buf[NetConsts::RECV_BUF_SZ];

        while (running_)
        {
            if (sockfd_ == -1)
                break;

            struct pollfd pfd;
            pfd.fd = sockfd_;
            pfd.events = POLLIN;

            int ret = poll(&pfd, 1, 500);
            if (ret > 0 && (pfd.revents & POLLIN))
            {
                sockaddr_in sender;
                socklen_t slen = sizeof(sender);
                ssize_t n = recvfrom(sockfd_, buf, sizeof(buf), 0, (struct sockaddr *)&sender, &slen);
                if (n > 0)
                {
                    process_packet(sender, buf, static_cast<size_t>(n));
                }
            }
        }
    }

    void NetworkCortex::pulse_loop()
    {
        while (running_)
        {
            char dummy[1] = {0};
            emit_pheromone(PheromoneType::IAM_HERE, dummy, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.pulse_interval_ms));
        }
    }

    void NetworkCortex::emit_pheromone(PheromoneType type, const void *data, size_t len)
    {
        if (sockfd_ == -1)
            return;

        size_t payload_sz = std::min(len, NetConsts::SAFE_UDP_PAYLOAD);
        size_t total_sz = sizeof(PheromonePacket) + payload_sz;

        if (total_sz > NetConsts::RECV_BUF_SZ)
            return;

        uint8_t packet_buf[NetConsts::RECV_BUF_SZ];
        auto *hdr = reinterpret_cast<PheromonePacket *>(packet_buf);

        hdr->magic = htonl(PheromonePacket::MAGIC_VAL);
        hdr->type = static_cast<uint8_t>(type);
        hdr->sender_port = htons(cfg_.udp_port);
        hdr->node_id = htobe64(node_id_);
        hdr->payload_len = htonl(static_cast<uint32_t>(payload_sz));

        if (payload_sz > 0 && data)
        {
            std::memcpy(packet_buf + sizeof(PheromonePacket), data, payload_sz);
        }

        sockaddr_in bcast{};
        bcast.sin_family = AF_INET;
        bcast.sin_addr.s_addr = htonl(INADDR_BROADCAST);
        bcast.sin_port = htons(cfg_.udp_port);

        sendto(sockfd_, packet_buf, static_cast<ssize_t>(total_sz), 0, (struct sockaddr *)&bcast, sizeof(bcast));
    }

    void NetworkCortex::process_packet(const sockaddr_in &sender, const uint8_t *buf, size_t len)
    {
        if (len < sizeof(PheromonePacket))
            return;

        auto *hdr = reinterpret_cast<const PheromonePacket *>(buf);
        if (ntohl(hdr->magic) != PheromonePacket::MAGIC_VAL)
            return;

        uint64_t remote_id = be64toh(hdr->node_id);
        if (remote_id == node_id_)
            return;

        if (hdr->type == static_cast<uint8_t>(PheromoneType::IAM_HERE))
        {
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &(sender.sin_addr), ip_str, INET_ADDRSTRLEN);

            NeighborInfo info;
            info.ip = std::string(ip_str);
            info.port = ntohs(hdr->sender_port);
            info.last_seen = now_ms();
            info.load_factor = 0.0f;

            std::unique_lock<std::shared_mutex> lk(neighbors_mu_);
            neighbors_[remote_id] = info;
        }
    }

    std::vector<NeighborInfo> NetworkCortex::get_neighbors()
    {
        std::shared_lock<std::shared_mutex> lk(neighbors_mu_);
        std::vector<NeighborInfo> out;
        uint64_t ts = now_ms();
        out.reserve(neighbors_.size());

        // Note: We cannot erase while holding a shared_lock (read-only).
        // For strict cleanup, we would need a unique_lock or a separate cleanup cycle.
        // For performance, we filter on read and return valid ones.
        // The cleanup should ideally happen in pulse_loop or a separate maintenance task.

        for (const auto &kv : neighbors_)
        {
            if (ts - kv.second.last_seen <= cfg_.neighbor_ttl_ms)
            {
                out.push_back(kv.second);
            }
        }
        return out;
    }

    uint64_t NetworkCortex::make_instance_node_id() const noexcept
    {
        uint64_t pid = static_cast<uint64_t>(::getpid());
        uint64_t t = now_ms();
        std::random_device rd;
        std::hash<std::thread::id> hasher;
        uint64_t tid = hasher(std::this_thread::get_id());

        uint64_t r = (static_cast<uint64_t>(rd()) << 32) | rd();
        return r ^ t ^ (pid << 16) ^ (tid << 48);
    }
}