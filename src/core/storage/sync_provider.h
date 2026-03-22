#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/metadata.h"

namespace pomai::core {

/**
 * @brief Represents a single operation in the WAL.
 */
struct WalEntry {
    uint64_t lsn;
    uint8_t op; // 1=Put, 2=Del, 3=PutMeta, 4=RawKV, 5=BatchStart, 6=BatchEnd
    VectorId id;
    uint32_t dim;
    std::span<const float> vec;
    std::string raw_data; // For RawKV, metadata blobs, etc.
    Metadata meta;
};

/**
 * @brief Interface for receiving synced WAL entries.
 * This can be implemented as a gRPC client, HTTP pusher, etc.
 */
class SyncReceiver {
public:
    virtual ~SyncReceiver() = default;
    virtual Status Receive(const WalEntry& entry) = 0;
};

/**
 * @brief Mock receiver for local testing and demonstration.
 */
class MockSyncReceiver : public SyncReceiver {
public:
    Status Receive(const WalEntry& entry) override {
        // In a real implementation, this would send data over the network.
        // For now, we just acknowledge receipt.
        last_received_lsn = entry.lsn;
        return Status::Ok();
    }
    uint64_t last_received_lsn = 0;
};

/**
 * @brief Handles streaming of WAL entries from a specific LSN.
 */
class WalStreamer {
public:
    WalStreamer(const std::string& db_path, uint32_t shard_id);
    
    /**
     * @brief Pushes all entries since last_lsn to the receiver.
     * @return Number of entries pushed.
     */
    Status PushSince(uint64_t last_lsn, SyncReceiver* receiver, uint64_t* new_last_lsn);

private:
    std::string db_path_;
    uint32_t shard_id_;
    
    std::string SegmentPath(uint64_t gen) const;
};

} // namespace pomai::core
