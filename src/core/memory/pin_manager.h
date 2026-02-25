#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstdint>

#include "pomai/snapshot.h"

namespace pomai::core {

class MemoryPinManager {
public:
    static MemoryPinManager& Instance() {
        static MemoryPinManager instance;
        return instance;
    }

    uint64_t Pin(std::shared_ptr<pomai::Snapshot> snapshot) {
        if (!snapshot) return 0;
        uint64_t session_id = next_session_id_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mu_);
        pinned_.emplace(session_id, std::move(snapshot));
        return session_id;
    }

    void Unpin(uint64_t session_id) {
        if (session_id == 0) return;
        std::lock_guard<std::mutex> lock(mu_);
        pinned_.erase(session_id);
    }

private:
    MemoryPinManager() : next_session_id_(1) {}
    ~MemoryPinManager() = default;

    MemoryPinManager(const MemoryPinManager&) = delete;
    MemoryPinManager& operator=(const MemoryPinManager&) = delete;

    std::mutex mu_;
    std::unordered_map<uint64_t, std::shared_ptr<pomai::Snapshot>> pinned_;
    std::atomic<uint64_t> next_session_id_;
};

} // namespace pomai::core
