/* src/ai/network_cortex.h */
#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <netinet/in.h>

namespace pomai::ai::orbit
{
    // Các loại tín hiệu hóa học (Pheromone Types)
    enum class PheromoneType : uint8_t {
        IAM_HERE = 1,      // Nhịp tim: "Tôi đang sống ở đây"
        SCENT_MAP = 2,     // Bloom Filter: "Tôi có mùi dữ liệu này"
        NECTAR_REQ = 3,    // Yêu cầu: "Cho tôi xin dữ liệu này"
        NECTAR_RES = 4     // Phản hồi: "Dữ liệu đây"
    };

    // Cấu trúc gói tin siêu nhỏ (Packed struct để gửi qua mạng)
    #pragma pack(push, 1)
    struct PheromonePacket {
        uint32_t magic = 0x504F4D41; // 'POMA'
        uint8_t type;                // PheromoneType
        uint16_t sender_port;        // Port TCP để fetch dữ liệu (nếu cần)
        uint64_t node_id;            // Định danh Node (SimHash của IP + Time)
        uint32_t payload_len;        // Độ dài dữ liệu đi kèm
        // Payload sẽ nằm ngay sau struct này
    };
    #pragma pack(pop)

    struct NeighborInfo {
        std::string ip;
        uint16_t port;
        uint64_t last_seen;
        float load_factor; // 0.0 - 1.0 (RAM usage)
    };

    class NetworkCortex
    {
    public:
        NetworkCortex(uint16_t udp_port = 7777);
        ~NetworkCortex();

        void start();
        void stop();

        // Phát tán mùi hương (Broadcast UDP)
        void emit_pheromone(PheromoneType type, const void* data, size_t len);

        // Lấy danh sách hàng xóm đang sống
        std::vector<NeighborInfo> get_neighbors();

    private:
        int sockfd_;
        uint16_t port_;
        std::atomic<bool> running_;
        std::thread listener_thread_;
        std::thread pulse_thread_; // Nhịp tim định kỳ

        std::mutex neighbors_mu_;
        std::unordered_map<uint64_t, NeighborInfo> neighbors_;

        void listen_loop();
        void pulse_loop();
        void process_packet(const sockaddr_in& sender, const std::vector<uint8_t>& buffer);
    };
}