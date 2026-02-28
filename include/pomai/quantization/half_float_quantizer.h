#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pomai/quantization/vector_quantizer.h"
#include "pomai/status.h"

namespace pomai::core {

// HalfFloatQuantizer compresses 32-bit floats into 16-bit half-precision floats.
// It provides a high-accuracy alternative to 8-bit scalar quantization, 
// using half the memory of full 32-bit floats with minimal precision loss.
class HalfFloatQuantizer : public VectorQuantizer<float> {
public:
    explicit HalfFloatQuantizer(size_t dim);
    ~HalfFloatQuantizer() override = default;

    // Strict RAII: delete copy semantics
    HalfFloatQuantizer(const HalfFloatQuantizer&) = delete;
    HalfFloatQuantizer& operator=(const HalfFloatQuantizer&) = delete;

    // Support move semantics
    HalfFloatQuantizer(HalfFloatQuantizer&&) noexcept = default;
    HalfFloatQuantizer& operator=(HalfFloatQuantizer&&) noexcept = default;

    // Train is a no-op for FP16 as it doesn't require learning bounds.
    pomai::Status Train(std::span<const float> data, size_t num_vectors) override;

    // Encodes a float vector to uint16_t (as uint8_t codes, 2x bytes).
    std::vector<uint8_t> Encode(std::span<const float> vector) const override;

    // Decodes uint16_t codes (from uint8_t codes) back to float space.
    std::vector<float> Decode(std::span<const uint8_t> codes) const override;

    // Computes distance natively between raw float query and compressed FP16 codes.
    float ComputeDistance(std::span<const float> query, std::span<const uint8_t> codes) const override;

private:
    size_t dim_{0};
};

} // namespace pomai::core
