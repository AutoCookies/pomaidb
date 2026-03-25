#include "core/keyvalue/keyvalue_engine.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace pomai::core {

namespace {
uint64_t NowSec() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
}

KeyValueEngine::KeyValueEngine(std::string path, std::size_t max_entries)
    : KeyValueEngine(std::move(path), max_entries, RetentionPolicy{}) {}

KeyValueEngine::KeyValueEngine(std::string path, std::size_t max_entries, RetentionPolicy retention)
    : path_(std::move(path)), max_entries_(max_entries), retention_(retention) {}

Status KeyValueEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("kv create dir failed");
    const uint64_t now = NowSec();

    std::ifstream in(path_ + "/kv.log", std::ios::binary);
    if (in.good()) {
        while (true) {
            uint8_t op = 0;
            uint32_t ksz = 0, vsz = 0;
            if (!in.read(reinterpret_cast<char*>(&op), sizeof(op))) break;
            if (!in.read(reinterpret_cast<char*>(&ksz), sizeof(ksz))) break;
            if (!in.read(reinterpret_cast<char*>(&vsz), sizeof(vsz))) break;
            std::string k(ksz, '\0');
            if (!in.read(k.data(), static_cast<std::streamsize>(ksz))) break;
            std::string v(vsz, '\0');
            if (vsz > 0 && !in.read(v.data(), static_cast<std::streamsize>(vsz))) break;
            if (op == 1) kv_[std::move(k)] = Entry{std::move(v), now};
            else if (op == 2) kv_.erase(k);
            if (max_entries_ > 0 && kv_.size() > max_entries_) kv_.erase(kv_.begin());
        }
    }
    std::ifstream in2(path_ + "/kv_timed.log", std::ios::binary);
    if (in2.good()) {
        while (true) {
            uint8_t op = 0;
            uint32_t ksz = 0, vsz = 0;
            uint64_t ts = 0;
            if (!in2.read(reinterpret_cast<char*>(&op), sizeof(op))) break;
            if (!in2.read(reinterpret_cast<char*>(&ksz), sizeof(ksz))) break;
            if (!in2.read(reinterpret_cast<char*>(&vsz), sizeof(vsz))) break;
            if (!in2.read(reinterpret_cast<char*>(&ts), sizeof(ts))) break;
            std::string k(ksz, '\0');
            if (!in2.read(k.data(), static_cast<std::streamsize>(ksz))) break;
            std::string v(vsz, '\0');
            if (vsz > 0 && !in2.read(v.data(), static_cast<std::streamsize>(vsz))) break;
            if (op == 1) kv_[std::move(k)] = Entry{std::move(v), ts};
            else if (op == 2) kv_.erase(k);
        }
    }
    EnforceRetention();
    return Status::Ok();
}

Status KeyValueEngine::Close() { return Status::Ok(); }

Status KeyValueEngine::Put(std::string_view key, std::string_view value) {
    if (key.empty()) return Status::InvalidArgument("kv key empty");
    if (max_entries_ > 0 && kv_.size() >= max_entries_ && kv_.find(std::string(key)) == kv_.end()) {
        kv_.erase(kv_.begin());
    }
    const uint64_t now = NowSec();
    kv_[std::string(key)] = Entry{std::string(value), now};
    EnforceRetention();
    std::ofstream out(path_ + "/kv_timed.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("kv append failed");
    uint8_t op = 1;
    uint32_t ksz = static_cast<uint32_t>(key.size());
    uint32_t vsz = static_cast<uint32_t>(value.size());
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&ksz), sizeof(ksz));
    out.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
    out.write(reinterpret_cast<const char*>(&now), sizeof(now));
    out.write(key.data(), static_cast<std::streamsize>(key.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
    return Status::Ok();
}

Status KeyValueEngine::Get(std::string_view key, std::string* out) const {
    if (!out) return Status::InvalidArgument("kv out null");
    auto it = kv_.find(std::string(key));
    if (it == kv_.end()) return Status::NotFound("kv not found");
    if (IsExpired(it->second, NowSec())) return Status::NotFound("kv expired");
    *out = it->second.value;
    return Status::Ok();
}

Status KeyValueEngine::Delete(std::string_view key) {
    kv_.erase(std::string(key));
    std::ofstream out(path_ + "/kv_timed.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("kv append failed");
    uint8_t op = 2;
    uint32_t ksz = static_cast<uint32_t>(key.size());
    uint32_t vsz = 0;
    uint64_t ts = NowSec();
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&ksz), sizeof(ksz));
    out.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
    out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
    out.write(key.data(), static_cast<std::streamsize>(key.size()));
    return Status::Ok();
}

Status KeyValueEngine::Compact() {
    EnforceRetention();
    std::ofstream out(path_ + "/kv_timed.log", std::ios::binary | std::ios::trunc);
    if (!out.good()) return Status::IOError("kv compact open failed");
    for (const auto& kv : kv_) {
        uint8_t op = 1;
        uint32_t ksz = static_cast<uint32_t>(kv.first.size());
        uint32_t vsz = static_cast<uint32_t>(kv.second.value.size());
        uint64_t ts = kv.second.created_at_sec;
        out.write(reinterpret_cast<const char*>(&op), sizeof(op));
        out.write(reinterpret_cast<const char*>(&ksz), sizeof(ksz));
        out.write(reinterpret_cast<const char*>(&vsz), sizeof(vsz));
        out.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
        out.write(kv.first.data(), static_cast<std::streamsize>(kv.first.size()));
        out.write(kv.second.value.data(), static_cast<std::streamsize>(kv.second.value.size()));
    }
    return Status::Ok();
}

Status KeyValueEngine::SetRetentionPolicy(RetentionPolicy retention) {
    retention_ = retention;
    EnforceRetention();
    return Compact();
}

bool KeyValueEngine::IsExpired(const Entry& e, uint64_t now_sec) const {
    return retention_.ttl_sec > 0 && e.created_at_sec > 0 && now_sec > e.created_at_sec &&
           (now_sec - e.created_at_sec) >= retention_.ttl_sec;
}

uint64_t KeyValueEngine::ApproxBytesUsed() const {
    uint64_t bytes = 0;
    for (const auto& kv : kv_) bytes += kv.first.size() + kv.second.value.size();
    return bytes;
}

void KeyValueEngine::EnforceRetention() {
    const uint64_t now = NowSec();
    if (retention_.ttl_sec > 0) {
        for (auto it = kv_.begin(); it != kv_.end();) {
            if (IsExpired(it->second, now)) it = kv_.erase(it);
            else ++it;
        }
    }
    auto prune_oldest = [this]() {
        auto oldest = kv_.end();
        for (auto it = kv_.begin(); it != kv_.end(); ++it) {
            if (oldest == kv_.end() || it->second.created_at_sec < oldest->second.created_at_sec) oldest = it;
        }
        if (oldest != kv_.end()) kv_.erase(oldest);
    };
    while (retention_.max_count > 0 && kv_.size() > retention_.max_count) prune_oldest();
    while (retention_.max_bytes > 0 && ApproxBytesUsed() > retention_.max_bytes && !kv_.empty()) prune_oldest();
}

void KeyValueEngine::ForEach(const std::function<void(std::string_view key, std::string_view value)>& fn) const {
    for (const auto& [k, e] : kv_) fn(k, e.value);
}

} // namespace pomai::core

