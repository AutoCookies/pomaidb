#include "pomai/quantization/half_float_quantizer.h"
#include "util/half_float.h"
#include "core/distance.h"

#include <algorithm>
#include <cstring>

namespace pomai::core {

HalfFloatQuantizer::HalfFloatQuantizer(size_t dim)
    : dim_(dim) {}

pomai::Status HalfFloatQuantizer::Train(std::span<const float> /*data*/, size_t num_vectors) {
    if (num_vectors == 0 || dim_ == 0) {
        return pomai::Status::InvalidArgument("Empty dimensions or vectors for training");
    }
    // FP16 quantization is a direct mapping and doesn't require learning dataset bounds.
    return pomai::Status::Ok();
}

std::vector<uint8_t> HalfFloatQuantizer::Encode(std::span<const float> vector) const {
    if (vector.size() != dim_) {
        return {};
    }

    // We store uint16_t but the interface expects uint8_t codes.
    // Length is dim * 2 bytes.
    std::vector<uint8_t> codes(dim_ * sizeof(uint16_t));
    uint16_t* h_ptr = reinterpret_cast<uint16_t*>(codes.data());

    for (size_t i = 0; i < dim_; ++i) {
        h_ptr[i] = pomai::util::float32_to_float16(vector[i]);
    }

    return codes;
}

std::vector<float> HalfFloatQuantizer::Decode(std::span<const uint8_t> codes) const {
    if (codes.size() != dim_ * sizeof(uint16_t)) {
        return {};
    }

    std::vector<float> decoded(dim_);
    const uint16_t* h_ptr = reinterpret_cast<const uint16_t*>(codes.data());

    for (size_t i = 0; i < dim_; ++i) {
        decoded[i] = pomai::util::float16_to_float32(h_ptr[i]);
    }

    return decoded;
}

float HalfFloatQuantizer::ComputeDistance(std::span<const float> query, std::span<const uint8_t> codes) const {
    if (query.size() != dim_ || codes.size() != dim_ * sizeof(uint16_t)) {
        return -1.0f;
    }

    const uint16_t* h_ptr = reinterpret_cast<const uint16_t*>(codes.data());
    std::span<const uint16_t> h_codes(h_ptr, dim_);

    // Dispatch to optimized distance kernel (Dot or L2 depending on default metric).
    // For simplicity, we assume Dot here or dispatch based on runtime context if available.
    // In PomaiDB, SegmentReader usually knows the metric.
    // Here we use DotFp16 as a primary example.
    return pomai::core::DotFp16(query, h_codes);
}

} // namespace pomai::core
