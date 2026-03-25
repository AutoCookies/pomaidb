#include "core/blob/blob_engine.h"

#include <filesystem>
#include <fstream>

namespace pomai::core {

BlobEngine::BlobEngine(std::string path, std::size_t max_blob_bytes)
    : path_(std::move(path)), max_blob_bytes_(max_blob_bytes) {}

Status BlobEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("blob create dir failed");
    std::ifstream in(path_ + "/blob.log", std::ios::binary);
    if (!in.good()) return Status::Ok();
    while (true) {
        uint8_t op = 0;
        uint64_t id = 0;
        uint32_t sz = 0;
        if (!in.read(reinterpret_cast<char*>(&op), sizeof(op))) break;
        if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
        if (!in.read(reinterpret_cast<char*>(&sz), sizeof(sz))) break;
        if (op == 1) {
            std::vector<uint8_t> b(sz);
            if (sz > 0 && !in.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(sz))) break;
            auto old = blobs_.find(id);
            if (old != blobs_.end()) current_bytes_ -= old->second.size();
            blobs_[id] = std::move(b);
            current_bytes_ += sz;
        } else if (op == 2) {
            auto it = blobs_.find(id);
            if (it != blobs_.end()) {
                current_bytes_ -= it->second.size();
                blobs_.erase(it);
            }
        } else {
            break;
        }
    }
    return Status::Ok();
}

Status BlobEngine::Close() { return Status::Ok(); }

Status BlobEngine::Put(uint64_t id, std::span<const uint8_t> data) {
    std::size_t incoming = data.size();
    std::size_t old = 0;
    auto it = blobs_.find(id);
    if (it != blobs_.end()) old = it->second.size();
    if (max_blob_bytes_ > 0 && current_bytes_ - old + incoming > max_blob_bytes_) {
        return Status::ResourceExhausted("blob memory cap exceeded");
    }
    std::vector<uint8_t> copy(data.begin(), data.end());
    blobs_[id] = copy;
    current_bytes_ = current_bytes_ - old + incoming;

    std::ofstream out(path_ + "/blob.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("blob append failed");
    uint8_t op = 1;
    uint32_t sz = static_cast<uint32_t>(copy.size());
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    if (!copy.empty()) out.write(reinterpret_cast<const char*>(copy.data()), static_cast<std::streamsize>(copy.size()));
    return Status::Ok();
}

Status BlobEngine::Get(uint64_t id, std::vector<uint8_t>* out) const {
    if (!out) return Status::InvalidArgument("blob out null");
    auto it = blobs_.find(id);
    if (it == blobs_.end()) return Status::NotFound("blob not found");
    *out = it->second;
    return Status::Ok();
}

Status BlobEngine::Delete(uint64_t id) {
    auto it = blobs_.find(id);
    if (it != blobs_.end()) {
        current_bytes_ -= it->second.size();
        blobs_.erase(it);
    }
    std::ofstream out(path_ + "/blob.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("blob append failed");
    uint8_t op = 2;
    uint32_t sz = 0;
    out.write(reinterpret_cast<const char*>(&op), sizeof(op));
    out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    return Status::Ok();
}

void BlobEngine::ForEach(const std::function<void(uint64_t id, std::size_t nbytes)>& fn) const {
    for (const auto& [id, data] : blobs_) fn(id, data.size());
}

} // namespace pomai::core

