#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

class BlobEngine {
public:
    BlobEngine(std::string path, std::size_t max_blob_bytes);
    Status Open();
    Status Close();
    Status Put(uint64_t id, std::span<const uint8_t> data);
    Status Get(uint64_t id, std::vector<uint8_t>* out) const;
    Status Delete(uint64_t id);
    void ForEach(const std::function<void(uint64_t id, std::size_t nbytes)>& fn) const;

private:
    std::string path_;
    std::size_t max_blob_bytes_;
    std::size_t current_bytes_ = 0;
    std::unordered_map<uint64_t, std::vector<uint8_t>> blobs_;
};

} // namespace pomai::core

