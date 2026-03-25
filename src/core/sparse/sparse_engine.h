#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>

#include "pomai/pomai.h"
#include "pomai/status.h"

namespace pomai::core {

class SparseEngine {
public:
    explicit SparseEngine(std::size_t max_entries) : max_entries_(max_entries) {}
    Status Put(std::uint64_t id, const pomai::SparseEntry& entry);
    Status Dot(std::uint64_t a, std::uint64_t b, double* out) const;
    Status Intersect(std::uint64_t a, std::uint64_t b, std::uint32_t* out) const;
    Status Jaccard(std::uint64_t a, std::uint64_t b, double* out) const;
    void ForEach(const std::function<void(std::uint64_t id, std::size_t nnz)>& fn) const;

private:
    std::size_t max_entries_;
    std::unordered_map<std::uint64_t, pomai::SparseEntry> sparse_;
};

} // namespace pomai::core

