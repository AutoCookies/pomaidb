/*
 * src/ai/network_cortex.cc
 *
 * Simplified, robust UDP-based NetworkCortex for local node discovery.
 *
 * Features:
 * - Secure Packet Processing: Strict bounds checking to prevent Buffer Overread/Overflow.
 * - Cross-Platform Endianness: Correct serialization for mixed-arch clusters.
 * - Self-Healing: Automatic eviction of stale neighbors (TTL).
 * - Zero-Allocation Hot Path: Uses stack buffers for packet processing.
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

// Platform-independent Endianness Macros
#if defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define htobe64(x) OSSwapHostToBigInt64(x)
#define be64toh(x) OSSwapBigToHostInt64(x)
#elif defined(__linux__)
#include <endian.h> 
#else
// Fallback for others (assuming little-endian host for x86/ARM)
#include <byteswap.h>
#define htobe64(x) __builtin_bswap64(x)
#define be64toh(x) __builtin_bswap64(x)
#endif

namespace pomai::ai::orbit
{

    namespace {
        // --- Constants & Config ---
        static constexpr size_t RECV_BUF_SZ = 4096;
        static constexpr size_t SAFE_UDP_PAYLOAD = 1400; // MTU safe limit
        static constexpr int POLL_TIMEOUT_MS = 1000;
        static constexpr uint64_t NEIGHBOR_TTL_MS = 15 * 1000; // 15 seconds
        static constexpr size_t MAX_LOAD_PAYLOAD = 32;         // Small buffer for float parsing

        static inline uint64_t now_ms() {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
        }
    }

    NetworkCortex::NetworkCortex(uint16_t udp_port)
        : port_(udp_port), running_(false), sockfd_(-1)
    {
        node_id_ = make_instance_node_id();

        // 1. Create UDP socket
        sockfd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            std::cerr << "[Cortex] socket() failed: " << std::strerror(errno) << "\n";
            return;
        }

        // 2. Configure Socket
        int on = 1;
        // Allow broadcast
        if (setsockopt(sockfd_, SOL_SOCKET, SO_BROADCAST, &on, sizeof(on)) != 0) {
            std::clog << "[Cortex] warning: setsockopt(SO_BROADCAST) failed\n";
        }
        // Allow address reuse (quick restart)
        if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) != 0) {
            std::clog << "[Cortex] warning: setsockopt(SO_REUSEADDR) failed\n";
        }
#ifdef SO_REUSEPORT
        if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on)) != 0) {
            std::clog << "[Cortex] warning: setsockopt(SO_REUSEPORT) failed\n";
        }
#endif

        // 3. Bind
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY; // Listen on 0.0.0.0

        if (::bind(sockfd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            std::cerr << "[Cortex] bind(" << port_ << ") failed: " << std::strerror(errno) << "\n";
            ::close(sockfd_);
            sockfd_ = -1;
            return;
        }

        std::clog << "[Cortex] Node initialized (ID: " << std::hex << node_id_ << std::dec << ") on UDP port " << port_ << "\n";
    }

    NetworkCortex::~NetworkCortex()
    {
        stop();
    }

    void NetworkCortex::start()
    {
        if (sockfd_ < 0) return;
        
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) return;

        std::clog << "[Cortex] Service starting...\n";
        listener_thread_ = std::thread([this]{ listen_loop(); });
        pulse_thread_ = std::thread([this]{ pulse_loop(); });
    }

    void NetworkCortex::stop()
    {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) return;

        if (listener_thread_.joinable()) listener_thread_.join();
        if (pulse_thread_.joinable()) pulse_thread_.join();

        if (sockfd_ >= 0) {
            ::close(sockfd_);
            sockfd_ = -1;
        }
        std::clog << "[Cortex] Service stopped.\n";
    }

    void NetworkCortex::emit_pheromone(PheromoneType type, const void* data, size_t len)
    {
        if (sockfd_ < 0) return;

        // Security: Truncate oversized payloads
        if (len > SAFE_UDP_PAYLOAD) {
            len = SAFE_UDP_PAYLOAD;
        }

        // Build Packet
        PheromonePacket pkt{};
        pkt.magic = htonl(0x504F4D41u); // "POMA"
        pkt.type = static_cast<uint8_t>(type);
        pkt.sender_port = htons(port_);
        pkt.node_id = htobe64(node_id_);
        pkt.payload_len = htonl(static_cast<uint32_t>(len));

        // Serialize to buffer
        // Note: Using std::vector here is safe but allocation-heavy. 
        // For 10/10 perf, we could use a thread_local stack buffer, 
        // but this function is not called in a hot loop (only 1/sec), so clarity > micro-opt.
        std::vector<uint8_t> buf(sizeof(pkt) + len);
        std::memcpy(buf.data(), &pkt, sizeof(pkt));
        if (data && len > 0) {
            std::memcpy(buf.data() + sizeof(pkt), data, len);
        }

        // Broadcast
        sockaddr_in dest{};
        dest.sin_family = AF_INET;
        dest.sin_port = htons(port_);
        dest.sin_addr.s_addr = INADDR_BROADCAST;

        ::sendto(sockfd_, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&dest), sizeof(dest));
    }

    void NetworkCortex::listen_loop()
    {
        // Zero-allocation receive buffer
        std::vector<uint8_t> recv_buf(RECV_BUF_SZ);
        struct pollfd pfd{};
        pfd.fd = sockfd_;
        pfd.events = POLLIN;

        while (running_.load(std::memory_order_relaxed)) {
            int ret = ::poll(&pfd, 1, POLL_TIMEOUT_MS);
            if (ret <= 0) continue; // Timeout or Error (ignore)

            if (pfd.revents & POLLIN) {
                sockaddr_in sender{};
                socklen_t slen = sizeof(sender);
                ssize_t n = ::recvfrom(sockfd_, recv_buf.data(), recv_buf.size(), 0,
                                       reinterpret_cast<sockaddr*>(&sender), &slen);
                
                if (n > 0) {
                    process_packet(sender, recv_buf.data(), static_cast<size_t>(n));
                }
            }
        }
    }

    void NetworkCortex::pulse_loop()
    {
        const char status[] = "ALIVE";
        while (running_.load(std::memory_order_relaxed)) {
            emit_pheromone(PheromoneType::IAM_HERE, status, sizeof(status) - 1);
            
            // Sleep 1s, check shutdown every 100ms
            for (int i = 0; i < 10 && running_.load(std::memory_order_relaxed); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void NetworkCortex::process_packet(const sockaddr_in& sender, const uint8_t* buf, size_t len)
    {
        // 1. Basic Header Check
        if (len < sizeof(PheromonePacket)) return;

        PheromonePacket header_net;
        std::memcpy(&header_net, buf, sizeof(header_net));

        // 2. Validate Magic & ID
        if (ntohl(header_net.magic) != 0x504F4D41u) return;
        
        uint64_t remote_id = be64toh(header_net.node_id);
        if (remote_id == node_id_) return; // Ignore self

        // 3. [SECURITY] Validate Payload Length
        uint32_t payload_len = ntohl(header_net.payload_len);
        
        // Check 1: Logical limit
        if (payload_len > SAFE_UDP_PAYLOAD) return; 
        
        // Check 2: Physical buffer boundary (Anti-Buffer-Overread)
        if (len < sizeof(PheromonePacket) + payload_len) return; 

        // 4. Process Logic
        if (static_cast<PheromoneType>(header_net.type) == PheromoneType::IAM_HERE) {
            char ip_str[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &sender.sin_addr, ip_str, sizeof(ip_str));

            NeighborInfo info;
            info.ip = ip_str;
            info.port = ntohs(header_net.sender_port);
            info.last_seen = now_ms();
            info.load_factor = 0.5f;

            // Safe Payload Parsing (Stack buffer with hard limit)
            if (payload_len > 0) {
                char smallbuf[MAX_LOAD_PAYLOAD + 1];
                size_t copy_len = std::min<size_t>(payload_len, MAX_LOAD_PAYLOAD);
                std::memcpy(smallbuf, buf + sizeof(PheromonePacket), copy_len);
                smallbuf[copy_len] = '\0'; // Ensure null-termination

                // Try parse float (load factor)
                try {
                    float lf = std::stof(smallbuf);
                    if (lf >= 0.0f && lf <= 1.0f) info.load_factor = lf;
                } catch (...) {}
            }

            // Update Neighbor Map
            std::lock_guard<std::mutex> lk(neighbors_mu_);
            neighbors_[remote_id] = info;
        }
    }

    std::vector<NeighborInfo> NetworkCortex::get_neighbors()
    {
        std::lock_guard<std::mutex> lk(neighbors_mu_);
        std::vector<NeighborInfo> out;
        uint64_t now = now_ms();

        // Iterate and Prune (Lazy Eviction)
        for (auto it = neighbors_.begin(); it != neighbors_.end(); ) {
            if (now - it->second.last_seen > NEIGHBOR_TTL_MS) {
                it = neighbors_.erase(it); // Remove stale node
            } else {
                out.push_back(it->second);
                ++it;
            }
        }
        return out;
    }

    uint64_t NetworkCortex::make_instance_node_id() const noexcept
    {
        // Generate a reasonably unique 64-bit ID
        uint64_t pid = static_cast<uint64_t>(::getpid());
        uint64_t t = now_ms();
        
        std::random_device rd;
        uint64_t r = (static_cast<uint64_t>(rd()) << 32) | rd(); // Mix 2x 32-bit entropy

        // Hash mix
        uint64_t h = pid;
        h = (h ^ t) * 0x9ddfea08eb382d69ULL;
        h = (h ^ r) * 0x9ddfea08eb382d69ULL;
        
        return h ? h : 1; // Ensure non-zero
    }

} // namespace pomai::ai::orbit