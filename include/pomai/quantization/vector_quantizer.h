#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

// A highly extensible base template interface for compressing raw vectors 
// into lower-precision representations to minimize memory footprint and latency.
template <typename T>
class VectorQuantizer {
public:
    VectorQuantizer() = default;
    virtual ~VectorQuantizer() = default;

    // Strict RAII: delete copy semantics to prevent accidental costly copies
    VectorQuantizer(const VectorQuantizer&) = delete;
    VectorQuantizer& operator=(const VectorQuantizer&) = delete;

    // Support move semantics for efficient ownership transfer
    VectorQuantizer(VectorQuantizer&&) noexcept = default;
    VectorQuantizer& operator=(VectorQuantizer&&) noexcept = default;

    // Learns the distribution (e.g., min/max bounds) of the dataset.
    // Returns Status::Ok() on success, or an error status on failure.
    virtual pomai::Status Train(std::span<const T> data, size_t num_vectors) = 0;

    // Compresses a raw vector into a discrete bucket representation.
    virtual std::vector<uint8_t> Encode(std::span<const T> vector) const = 0;

    // Decompresses the discrete codes back into an approximate float representation.
    virtual std::vector<T> Decode(std::span<const uint8_t> codes) const = 0;

    // Computes the L2 or Inner Product distance directly between a raw float query 
    // and a compressed vector without fully decoding the vector.
    virtual float ComputeDistance(std::span<const T> query, std::span<const uint8_t> codes) const = 0;
};

} // namespace pomai::core
