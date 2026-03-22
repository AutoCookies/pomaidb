#pragma once
#include "pomai/quantization/vector_quantizer.h"
#include "pomai/status.h"

namespace pomai::core {

/**
 * @brief 1-bit Binary Quantizer.
 * Signs of the components are encoded: 1 if > 0, 0 otherwise.
 * Memory reduction: 32x.
 */
class BitQuantizer : public VectorQuantizer<float> {
public:
    explicit BitQuantizer(size_t dim);
    ~BitQuantizer() override = default;

    BitQuantizer(const BitQuantizer&) = delete;
    BitQuantizer& operator=(const BitQuantizer&) = delete;

    pomai::Status Train(std::span<const float> data, size_t num_vectors) override;

    std::vector<uint8_t> Encode(std::span<const float> vector) const override;

    std::vector<float> Decode(std::span<const uint8_t> codes) const override;

    float ComputeDistance(std::span<const float> query, std::span<const uint8_t> codes) const override;

private:
    size_t dim_{0};
    bool is_trained_{false};
};

} // namespace pomai::core
