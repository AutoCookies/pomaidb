#pragma once
/*
 * src/ai/network_cortex.h
 *
 * Simplified, robust UDP-based NetworkCortex for local node discovery.
 * Refactored to use centralized pomai::config::NetworkCortexConfig.
 */

#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <netinet/in.h>

#include "src/core/config.h"

namespace pomai::ai::orbit
{
    // Alias for constants
    using NetConsts = pomai::config::NetworkCortexConstants;

    enum class PheromoneType : uint8_t
    {
        IAM_HERE = 1,
        SCENT_MAP = 2,
        NECTAR_REQ = 3,
        NECTAR_RES = 4
    };

#pragma pack(push, 1)
    struct PheromonePacket
    {
        uint32_t magic; // 'POMA'
        uint8_t type;   // PheromoneType
        uint16_t sender_port;
        uint64_t node_id;
        uint32_t payload_len;
    };
#pragma pack(pop)

    struct NeighborInfo
    {
        std::string ip;
        uint16_t port;
        uint64_t last_seen;
        float load_factor;
    };

    class NetworkCortex
    {
    public:
        explicit NetworkCortex(const pomai::config::NetworkCortexConfig &cfg);
        ~NetworkCortex();

        void start();
        void stop();
        void emit_pheromone(PheromoneType type, const void *data, size_t len);
        std::vector<NeighborInfo> get_neighbors();

    private:
        void listen_loop();
        void pulse_loop();
        void process_packet(const sockaddr_in &sender, const uint8_t *buf, size_t len);
        uint64_t make_instance_node_id() const noexcept;

        pomai::config::NetworkCortexConfig cfg_;

        int sockfd_ = -1;
        std::atomic<bool> running_{false};
        std::thread listener_thread_;
        std::thread pulse_thread_;

        std::mutex neighbors_mu_;
        std::unordered_map<uint64_t, NeighborInfo> neighbors_;

        uint64_t node_id_ = 0;
    };
}