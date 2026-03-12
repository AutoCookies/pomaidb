#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "palloc_page_pool.h"

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
        uint64_t session_id = next_session_id_++;
        pinned_.emplace(session_id, std::move(snapshot));
        return session_id;
    }

    void Unpin(uint64_t session_id) {
        if (session_id == 0) return;
        pinned_.erase(session_id);
    }

private:
    MemoryPinManager() : next_session_id_(1) {}
    ~MemoryPinManager() = default;

    MemoryPinManager(const MemoryPinManager&) = delete;
    MemoryPinManager& operator=(const MemoryPinManager&) = delete;

    std::unordered_map<uint64_t, std::shared_ptr<pomai::Snapshot>> pinned_;
    uint64_t next_session_id_{1};
};

} // namespace pomai::core
