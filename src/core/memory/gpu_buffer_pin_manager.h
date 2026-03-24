#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace pomai::core {

/// Lightweight handle for keeping GPU bridge resources alive across sessions (single-threaded).
/// Mirrors MemoryPinManager semantics: Pin returns non-zero id; Unpin releases.
class GpuBufferPinRegistry {
public:
    static GpuBufferPinRegistry& Instance() {
        static GpuBufferPinRegistry instance;
        return instance;
    }

    template <typename T>
    uint64_t Pin(std::shared_ptr<T> resource) {
        if (!resource) {
            return 0;
        }
        const uint64_t session_id = next_session_id_++;
        pinned_.emplace(session_id, std::shared_ptr<void>(resource));
        return session_id;
    }

    void Unpin(uint64_t session_id) {
        if (session_id == 0) {
            return;
        }
        pinned_.erase(session_id);
    }

private:
    GpuBufferPinRegistry() : next_session_id_(1) {}
    ~GpuBufferPinRegistry() = default;

    GpuBufferPinRegistry(const GpuBufferPinRegistry&) = delete;
    GpuBufferPinRegistry& operator=(const GpuBufferPinRegistry&) = delete;

    std::unordered_map<uint64_t, std::shared_ptr<void>> pinned_;
    uint64_t next_session_id_{1};
};

}  // namespace pomai::core
