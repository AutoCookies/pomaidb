#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

#include "pomai/status.h"

namespace pomai::core {

class KeyValueEngine {
public:
    struct RetentionPolicy {
        uint32_t ttl_sec = 0;
        uint32_t max_count = 0;
        uint64_t max_bytes = 0;
    };

    KeyValueEngine(std::string path, std::size_t max_entries);
    KeyValueEngine(std::string path, std::size_t max_entries, RetentionPolicy retention);
    Status Open();
    Status Close();
    Status Put(std::string_view key, std::string_view value);
    Status Get(std::string_view key, std::string* out) const;
    Status Delete(std::string_view key);
    Status Compact();
    Status SetRetentionPolicy(RetentionPolicy retention);

    /** Stable order not guaranteed; snapshot of current map. */
    void ForEach(const std::function<void(std::string_view key, std::string_view value)>& fn) const;

private:
    struct Entry {
        std::string value;
        uint64_t created_at_sec = 0;
    };
    void EnforceRetention();
    bool IsExpired(const Entry& e, uint64_t now_sec) const;
    uint64_t ApproxBytesUsed() const;

    std::string path_;
    std::size_t max_entries_;
    RetentionPolicy retention_;
    std::unordered_map<std::string, Entry> kv_;
};

} // namespace pomai::core

