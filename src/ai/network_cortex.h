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
        uint32_t magic;     // 'POMA' == 0x504F4D41
        uint8_t type;       // PheromoneType
        uint16_t sender_port;
        uint64_t node_id;
        uint32_t payload_len;
        // Payload nằm ngay sau struct này
    };
    #pragma pack(pop)

    struct NeighborInfo {
        std::string ip;
        uint16_t port;
        uint64_t last_seen;
        float load_factor; // 0.0 - 1.0 (RAM usage)
    };

    // Lightweight, robust UDP broadcast listener/emitter for local discovery.
    // - Non-blocking friendly (uses poll() with timeout inside listen loop).
    // - Thread-safe neighbours map.
    // - Minimal heap allocations on hot path.
    class NetworkCortex
    {
    public:
        explicit NetworkCortex(uint16_t udp_port = 7777);
        ~NetworkCortex();

        // start/stop lifecycle (start spawns listener+pulse background threads)
        void start();
        void stop();

        // Emit broadcast pheromone. Payload is truncated to a safe UDP payload if too large.
        void emit_pheromone(PheromoneType type, const void* data, size_t len);

        // Snapshot copy of discovered neighbors
        std::vector<NeighborInfo> get_neighbors();

    private:
        // internal helpers
        void listen_loop();
        void pulse_loop();
        void process_packet(const sockaddr_in& sender, const uint8_t* buf, size_t len);
        uint64_t make_instance_node_id() const noexcept;

        int sockfd_ = -1;
        uint16_t port_;
        std::atomic<bool> running_{false};
        std::thread listener_thread_;
        std::thread pulse_thread_;

        std::mutex neighbors_mu_;
        std::unordered_map<uint64_t, NeighborInfo> neighbors_;

        // stable instance node id (used to ignore our own broadcasts)
        uint64_t node_id_ = 0;

        // tuning constants
        static constexpr size_t RECV_BUF_SZ = 4096;
        static constexpr size_t SAFE_UDP_PAYLOAD = 1400; // avoid fragmentation
        static constexpr int POLL_TIMEOUT_MS = 500;
    };
}