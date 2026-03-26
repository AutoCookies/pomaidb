#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <span>

namespace pomai::core {

    /** 
     * Identification for system-wide internal pods.
     */
    enum class PodId : uint8_t {
        kKernel = 0,
        kIngestion = 1,
        kIndex = 2,
        kQuery = 3,
        kRag = 4,
        kGraph = 5,
        kTimeSeries = 6,
        kEviction = 7,
        kCustom = 255
    };

    /** Standard Opcodes */
    namespace Op {
        // Vector/Index Opcodes (0x01 - 0x0F)
        constexpr uint32_t kPut    = 0x01;
        constexpr uint32_t kGet    = 0x02;
        constexpr uint32_t kSearch = 0x03;
        constexpr uint32_t kDelete = 0x04;
        constexpr uint32_t kFlush  = 0x05;
        constexpr uint32_t kFreeze = 0x06;
        constexpr uint32_t kSync   = 0x07;
        constexpr uint32_t kSearchMultiModal = 0x08;
        constexpr uint32_t kExists = 0x09;
        constexpr uint32_t kGetSnapshot      = 0x0A;
        constexpr uint32_t kNewIterator      = 0x0B;
        constexpr uint32_t kSearchLexical    = 0x0C;
        constexpr uint32_t kPutBatch         = 0x0D;

        // Graph Opcodes (0x10 - 0x1F)
        constexpr uint32_t kAddVertex           = 0x10;
        constexpr uint32_t kAddEdge             = 0x11;
        constexpr uint32_t kGetNeighbors       = 0x12;
        constexpr uint32_t kGetNeighborsWithType = 0x13;
    }

    struct TraceMetadata {
        uint64_t start_time_ns = 0;
        uint32_t hop_count = 0;
        bool enabled = false;
    };

    /**
     * Internal Pomegranate Message.
     * Lightweight and zero-copy where possible (payload is often a view).
     */
    struct Message {
        PodId sender;
        PodId target;
        uint32_t opcode;
        std::string_view membrane_id;
        
        // Payload can be raw bytes (span) or a small string
        std::span<const uint8_t> payload;
        
        // Synchronous result pointer (for single-threaded direct response)
        void* result_ptr = nullptr;

        // Optional LSN or Sequence for crash-recovery tracking
        uint64_t lsn = 0;

        // Tracing for Bigtech-scale observability
        TraceMetadata trace;

        // Factory helpers
        static Message Create(PodId to, uint32_t op, std::span<const uint8_t> pay = {}) {
            Message m;
            m.target = to;
            m.opcode = op;
            m.payload = pay;
            return m;
        }
    };

} // namespace pomai::core
