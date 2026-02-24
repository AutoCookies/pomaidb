#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pomai/quantization/vector_quantizer.h"
#include "pomai/status.h"

namespace pomai::core {

// ScalarQuantizer8Bit compresses 32-bit floats into 8-bit unsigned integers.
// It relies on global min and max bounds across the entire dataset to maximize 
// memory efficiency (critical for Edge devices) and allow for the simplest, 
// auto-vectorizable decoding loop.
class ScalarQuantizer8Bit : public VectorQuantizer<float> {
public:
    explicit ScalarQuantizer8Bit(size_t dim);
    ~ScalarQuantizer8Bit() override = default;

    // Strict RAII: delete copy semantics
    ScalarQuantizer8Bit(const ScalarQuantizer8Bit&) = delete;
    ScalarQuantizer8Bit& operator=(const ScalarQuantizer8Bit&) = delete;

    // Support move semantics
    ScalarQuantizer8Bit(ScalarQuantizer8Bit&&) noexcept = default;
    ScalarQuantizer8Bit& operator=(ScalarQuantizer8Bit&&) noexcept = default;

    // Train determines the global min and max to bucket [min, max] into [0, 255].
    pomai::Status Train(std::span<const float> data, size_t num_vectors) override;

    // Encodes a float vector to uint8_t codes.
    std::vector<uint8_t> Encode(std::span<const float> vector) const override;

    // Decodes uint8_t codes back to float space for rough approximation.
    std::vector<float> Decode(std::span<const uint8_t> codes) const override;

    // Computes L2 distance natively between a raw float query and compressed codes.
    float ComputeDistance(std::span<const float> query, std::span<const uint8_t> codes) const override;

    // Serialization getters and setters for MMap instantiation
    float GetGlobalMin() const { return global_min_; }
    float GetGlobalInvScale() const { return global_inv_scale_; }
    void LoadState(float min_val, float inv_scale);

private:
    size_t dim_{0};
    bool is_trained_{false};

    // Storing a single global scale instead of per-dimension scale saves memory 
    // (O(1) vs O(D) usage) and drastically simplifies vectorization broadcasting 
    // in ComputeDistance because constants can reside in single scalar registers.
    float global_min_{0.0f};
    float global_scale_{0.0f};      // Used during Encode
    float global_inv_scale_{0.0f};  // Used during Decode / ComputeDistance
};

} // namespace pomai::core
