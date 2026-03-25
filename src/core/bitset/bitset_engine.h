#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <span>
#include <unordered_map>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

class BitsetEngine {
public:
    explicit BitsetEngine(std::size_t max_bytes) : max_bytes_(max_bytes) {}
    Status Put(std::uint64_t id, std::span<const std::uint8_t> bits);
    Status And(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out) const;
    Status Or(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out) const;
    Status Xor(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out) const;
    Status Hamming(std::uint64_t a, std::uint64_t b, double* out) const;
    Status Jaccard(std::uint64_t a, std::uint64_t b, double* out) const;
    void ForEach(const std::function<void(std::uint64_t id, std::size_t nbytes)>& fn) const;

private:
    Status BinaryOp(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out, std::uint8_t op) const;
    std::size_t max_bytes_ = 0;
    std::size_t cur_bytes_ = 0;
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> bitsets_;
};

} // namespace pomai::core

