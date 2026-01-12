/* src/ai/network_cortex.cc */
#include "src/ai/network_cortex.h"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>

namespace pomai::ai::orbit
{
    // Helper lấy thời gian hiện tại (ms)
    static uint64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    NetworkCortex::NetworkCortex(uint16_t udp_port) 
        : port_(udp_port), running_(false), sockfd_(-1) 
    {
        // Tạo socket UDP
        sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd_ < 0) {
            std::cerr << "[Cortex] Failed to create socket\n";
            return;
        }

        // Cho phép Broadcast và Reuse Port (để chạy nhiều node trên 1 máy test)
        int broadcast = 1;
        setsockopt(sockfd_, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
        
        int reuse = 1;
        setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        #ifdef SO_REUSEPORT
        setsockopt(sockfd_, SOL_SOCKET, SO_REUSEPORT, &reuse, sizeof(reuse));
        #endif

        sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[Cortex] Failed to bind port " << port_ << "\n";
            close(sockfd_);
            sockfd_ = -1;
        }
    }

    NetworkCortex::~NetworkCortex() { stop(); }

    void NetworkCortex::start() {
        if (sockfd_ < 0 || running_) return;
        running_ = true;
        std::cout << "[Cortex] Awakening... Listening on UDP " << port_ << "\n";

        // Thread 1: Lắng nghe tín hiệu
        listener_thread_ = std::thread(&NetworkCortex::listen_loop, this);
        
        // Thread 2: Nhịp tim (gửi IAM_HERE mỗi 3s)
        pulse_thread_ = std::thread(&NetworkCortex::pulse_loop, this);
    }

    void NetworkCortex::stop() {
        running_ = false;
        if (sockfd_ >= 0) { close(sockfd_); sockfd_ = -1; }
        if (listener_thread_.joinable()) listener_thread_.join();
        if (pulse_thread_.joinable()) pulse_thread_.join();
    }

    void NetworkCortex::emit_pheromone(PheromoneType type, const void* data, size_t len) {
        if (sockfd_ < 0) return;

        std::vector<uint8_t> buffer(sizeof(PheromonePacket) + len);
        PheromonePacket* pkt = reinterpret_cast<PheromonePacket*>(buffer.data());
        
        pkt->magic = 0x504F4D41;
        pkt->type = static_cast<uint8_t>(type);
        pkt->sender_port = port_; // Tạm thời dùng port UDP làm định danh
        pkt->node_id = 12345;     // TODO: Generate real ID
        pkt->payload_len = len;
        
        if (data && len > 0) {
            std::memcpy(buffer.data() + sizeof(PheromonePacket), data, len);
        }

        sockaddr_in dest_addr;
        std::memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(port_);
        dest_addr.sin_addr.s_addr = INADDR_BROADCAST; // Gửi cho toàn mạng LAN

        sendto(sockfd_, buffer.data(), buffer.size(), 0, 
               (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    }

    void NetworkCortex::listen_loop() {
        std::vector<uint8_t> buffer(4096); // MTU an toàn
        sockaddr_in sender;
        socklen_t sender_len = sizeof(sender);

        while (running_) {
            ssize_t n = recvfrom(sockfd_, buffer.data(), buffer.size(), 0,
                                 (struct sockaddr*)&sender, &sender_len);
            if (n > 0) {
                // Parse packet
                if (n < sizeof(PheromonePacket)) continue;
                process_packet(sender, std::vector<uint8_t>(buffer.begin(), buffer.begin() + n));
            }
        }
    }

    void NetworkCortex::pulse_loop() {
        while (running_) {
            // Gửi tín hiệu IAM_HERE
            const char* status = "ALIVE";
            emit_pheromone(PheromoneType::IAM_HERE, status, strlen(status));
            
            // Ngủ 3 giây
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }

    void NetworkCortex::process_packet(const sockaddr_in& sender, const std::vector<uint8_t>& buffer) {
        const PheromonePacket* pkt = reinterpret_cast<const PheromonePacket*>(buffer.data());
        if (pkt->magic != 0x504F4D41) return;

        // Bỏ qua gói tin của chính mình (TODO: check node_id thật)
        
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(sender.sin_addr), ip_str, INET_ADDRSTRLEN);

        if (static_cast<PheromoneType>(pkt->type) == PheromoneType::IAM_HERE) {
            // std::cout << "[Cortex] Sensed pheromone from " << ip_str << "\n";
            std::lock_guard<std::mutex> lock(neighbors_mu_);
            neighbors_[pkt->node_id] = NeighborInfo{
                std::string(ip_str),
                pkt->sender_port,
                now_ms(),
                0.5f // Mock load
            };
        }
    }
    
    std::vector<NeighborInfo> NetworkCortex::get_neighbors() {
        std::lock_guard<std::mutex> lock(neighbors_mu_);
        std::vector<NeighborInfo> res;
        for (const auto& kv : neighbors_) res.push_back(kv.second);
        return res;
    }
}