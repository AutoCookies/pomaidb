/*
 * src/ai/network_cortex.cc
 *
 * Simplified, robust UDP-based NetworkCortex for local node discovery.
 *
 * Changes:
 *  - Ensure network/host byte-order conversions for portable on-wire format.
 *  - Add neighbor eviction (timeout) to avoid unbounded growth of neighbors_.
 *  - Safer parsing of load_factor payload (validate, bounds-check).
 */

#include "src/ai/network_cortex.h"

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>
#include <poll.h>
#include <random>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <thread>
#include <cassert>
#include <errno.h>
#include <cerrno>
#include <algorithm>
#include <climits>

#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#ifndef htobe64
#define htobe64(x) OSSwapHostToBigInt64(x)
#endif
#ifndef be64toh
#define be64toh(x) OSSwapBigToHostInt64(x)
#endif
#else
#include <endian.h> // htobe64 / be64toh on Linux
#endif

namespace pomai::ai::orbit
{

    namespace {
        static inline uint64_t now_ms() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
        }

        // Neighbor TTL (milliseconds)
        static constexpr uint64_t NEIGHBOR_TTL_MS = 15 * 1000; // 15 seconds

        // Max payload length accepted for ASCII load_factor parsing
        static constexpr size_t MAX_LOAD_PAYLOAD = 32;
    }

    NetworkCortex::NetworkCortex(uint16_t udp_port)
        : port_(udp_port), running_(false)
    {
        node_id_ = make_instance_node_id();

        // create UDP socket
        sockfd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            std::cerr << "[Cortex] socket() failed: " << std::strerror(errno) << "\n";
            sockfd_ = -1;
            return;
        }

        // Allow broadcast
        int on = 1;
        if (setsockopt(sockfd_, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on)) != 0) {
            std::clog << "[Cortex] warning: setsockopt(SO_BROADCAST) failed: " << std::strerror(errno) << "\n";
        }

        // Reuse addr
        if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) != 0) {
            std::clog << "[Cortex] warning: setsockopt(SO_REUSEADDR) failed: " << std::strerror(errno) << "\n";
        }
#ifdef SO_REUSEPORT
        if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on)) != 0) {
            std::clog << "[Cortex] warning: setsockopt(SO_REUSEPORT) failed: " << std::strerror(errno) << "\n";
        }
#endif

        // Bind to port on INADDR_ANY
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sockfd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            std::cerr << "[Cortex] bind(" << port_ << ") failed: " << std::strerror(errno) << "\n";
            ::close(sockfd_);
            sockfd_ = -1;
            return;
        }

        // Good: socket ready
        std::clog << "[Cortex] socket created and bound on UDP port " << port_ << "\n";
    }

    NetworkCortex::~NetworkCortex()
    {
        stop();
    }

    void NetworkCortex::start()
    {
        if (sockfd_ < 0) {
            std::cerr << "[Cortex] cannot start: socket invalid\n";
            return;
        }
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true))
            return; // already running

        std::clog << "[Cortex] Awakening... Listening on UDP " << port_ << "\n";

        listener_thread_ = std::thread([this]{ listen_loop(); });
        pulse_thread_ = std::thread([this]{ pulse_loop(); });
    }

    void NetworkCortex::stop()
    {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) {
            // not running
        }

        // join threads
        if (listener_thread_.joinable())
            listener_thread_.join();
        if (pulse_thread_.joinable())
            pulse_thread_.join();

        if (sockfd_ >= 0) {
            ::close(sockfd_);
            sockfd_ = -1;
        }

        std::clog << "[Cortex] stopped\n";
    }

    void NetworkCortex::emit_pheromone(PheromoneType type, const void* data, size_t len)
    {
        if (sockfd_ < 0)
            return;

        // Limit payload to avoid IP fragmentation
        if (len > SAFE_UDP_PAYLOAD) {
            std::clog << "[Cortex] emit_pheromone: payload too large (" << len << "), truncated to " << SAFE_UDP_PAYLOAD << "\n";
            len = SAFE_UDP_PAYLOAD;
        }

        // Build packet
        PheromonePacket pkt{};
        // convert to network byte order
        pkt.magic = htonl(0x504F4D41u);
        pkt.type = static_cast<uint8_t>(type);
        pkt.sender_port = htons(port_);
        pkt.node_id = htobe64(node_id_);
        pkt.payload_len = htonl(static_cast<uint32_t>(len));

        size_t pkt_size = sizeof(pkt) + len;
        std::vector<uint8_t> buf;
        buf.resize(pkt_size);

        // copy header then payload
        std::memcpy(buf.data(), &pkt, sizeof(pkt));
        if (data && len > 0)
            std::memcpy(buf.data() + sizeof(pkt), data, len);

        sockaddr_in dest{};
        dest.sin_family = AF_INET;
        dest.sin_port = htons(port_);
        dest.sin_addr.s_addr = INADDR_BROADCAST;

        ssize_t sent = ::sendto(sockfd_, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
        if (sent < 0) {
            std::clog << "[Cortex] sendto failed: " << std::strerror(errno) << "\n";
        }
    }

    void NetworkCortex::listen_loop()
    {
        // Local receive buffer (stack-backed via std::vector)
        std::vector<uint8_t> recv_buf(RECV_BUF_SZ);

        struct pollfd pfd{};
        pfd.fd = sockfd_;
        pfd.events = POLLIN;

        while (running_.load(std::memory_order_acquire)) {
            int ret = ::poll(&pfd, 1, POLL_TIMEOUT_MS);
            if (ret < 0) {
                if (errno == EINTR) continue;
                std::clog << "[Cortex] poll error: " << std::strerror(errno) << "\n";
                break;
            }
            if (ret == 0) {
                // timeout - recheck running_
                continue;
            }
            if ((pfd.revents & POLLIN) == 0) continue;

            sockaddr_in sender{};
            socklen_t slen = sizeof(sender);
            ssize_t n = ::recvfrom(sockfd_, recv_buf.data(), static_cast<int>(recv_buf.size()), 0,
                                   reinterpret_cast<sockaddr*>(&sender), &slen);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                std::clog << "[Cortex] recvfrom error: " << std::strerror(errno) << "\n";
                continue;
            }
            if (static_cast<size_t>(n) < sizeof(PheromonePacket)) {
                // too small
                continue;
            }

            // process in-place (no extra copy)
            process_packet(sender, recv_buf.data(), static_cast<size_t>(n));
        }
    }

    void NetworkCortex::pulse_loop()
    {
        const char status[] = "ALIVE";
        while (running_.load(std::memory_order_acquire)) {
            emit_pheromone(PheromoneType::IAM_HERE, status, sizeof(status) - 1);
            // sleep with periodic check
            for (int i = 0; i < 30 && running_.load(std::memory_order_acquire); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void NetworkCortex::process_packet(const sockaddr_in& sender, const uint8_t* buf, size_t len)
    {
        if (len < sizeof(PheromonePacket)) return;

        // Copy header safely (avoid unaligned access)
        PheromonePacket header_net{};
        std::memcpy(&header_net, buf, sizeof(header_net));

        // Convert network-order fields to host-order for processing
        uint32_t magic = ntohl(header_net.magic);
        if (magic != 0x504F4D41u) return;

        uint16_t sender_port = ntohs(header_net.sender_port);
        uint64_t node_id = be64toh(header_net.node_id);
        uint32_t payload_len = ntohl(header_net.payload_len);

        // Validate payload length consistency
        if (payload_len > SAFE_UDP_PAYLOAD) {
            // suspicious or malicious packet -> ignore
            return;
        }
        size_t expected_total = sizeof(PheromonePacket) + static_cast<size_t>(payload_len);
        if (len < expected_total) {
            // truncated packet -> ignore
            return;
        }

        // Ignore our own packet (compare host-order node_id)
        if (node_id == node_id_) return;

        // build ip string
        char ip_str[INET_ADDRSTRLEN] = {0};
        if (!inet_ntop(AF_INET, &sender.sin_addr, ip_str, sizeof(ip_str))) {
            std::strncpy(ip_str, "0.0.0.0", sizeof(ip_str));
        }

        if (static_cast<PheromoneType>(header_net.type) == PheromoneType::IAM_HERE) {
            NeighborInfo info;
            info.ip = std::string(ip_str);
            info.port = sender_port;
            info.last_seen = now_ms();
            info.load_factor = 0.5f; // default

            // Safely parse optional ASCII load payload (bounded)
            if (payload_len >= 1) {
                size_t take = std::min<size_t>(payload_len, MAX_LOAD_PAYLOAD);
                // copy to stack-local buffer and null-terminate
                char smallbuf[MAX_LOAD_PAYLOAD + 1];
                std::memcpy(smallbuf, buf + sizeof(PheromonePacket), take);
                smallbuf[take] = '\0';

                // parse float safely
                char *endp = nullptr;
                errno = 0;
                float v = std::strtof(smallbuf, &endp);
                if (endp != smallbuf && errno == 0) {
                    // accept only 0.0 .. 1.0
                    if (v >= 0.0f && v <= 1.0f) info.load_factor = v;
                }
            }

            {
                std::lock_guard<std::mutex> lk(neighbors_mu_);
                neighbors_[node_id] = std::move(info);
            }
        }

        // other types can be extended here (SCENT_MAP, NECTAR_REQ, NECTAR_RES)
    }

    std::vector<NeighborInfo> NetworkCortex::get_neighbors()
    {
        std::lock_guard<std::mutex> lk(neighbors_mu_);
        uint64_t now = now_ms();

        // prune stale neighbors
        for (auto it = neighbors_.begin(); it != neighbors_.end(); ) {
            if (now > it->second.last_seen && (now - it->second.last_seen) > NEIGHBOR_TTL_MS) {
                it = neighbors_.erase(it);
            } else {
                ++it;
            }
        }

        std::vector<NeighborInfo> out;
        out.reserve(neighbors_.size());
        for (const auto &kv : neighbors_) out.push_back(kv.second);
        return out;
    }

    uint64_t NetworkCortex::make_instance_node_id() const noexcept
    {
        // Build a simple but fairly unique id using pid + time + random
        uint64_t pid = static_cast<uint64_t>(::getpid());
        uint64_t t = now_ms();
        std::random_device rd;
        uint64_t r = 0;
        // random_device may produce values smaller than 32 bits; mix safely
        for (int i = 0; i < 4; ++i)
            r = (r << 16) ^ static_cast<uint64_t>(rd() & 0xFFFF);

        // mix (64-bit mix)
        uint64_t v = pid;
        v = (v * 0x9ddfea08eb382d69ULL) ^ (t + 0x9e3779b97f4a7c15ULL);
        v ^= (r << 1) | 1;
        if (v == 0) v = 1; // avoid reserved zero
        return v;
    }

} // namespace pomai::ai::orbit