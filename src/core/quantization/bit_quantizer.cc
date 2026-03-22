#include "pomai/quantization/bit_quantizer.h"
#include "core/distance.h"
#include <cstring>

namespace pomai::core {

BitQuantizer::BitQuantizer(size_t dim) : dim_(dim) {}

pomai::Status BitQuantizer::Train(std::span<const float> data, size_t num_vectors) {
    is_trained_ = true;
    return pomai::Status::Ok();
}

std::vector<uint8_t> BitQuantizer::Encode(std::span<const float> vector) const {
    if (vector.size() != dim_) return {};
    std::vector<uint8_t> codes((dim_ + 7) / 8, 0);
    BitQuantize(vector, codes.data());
    return codes;
}

std::vector<float> BitQuantizer::Decode(std::span<const uint8_t> codes) const {
    if (codes.size() != (dim_ + 7) / 8) return {};
    std::vector<float> decoded(dim_);
    for (size_t i = 0; i < dim_; ++i) {
        bool bit = (codes[i / 8] >> (i % 8)) & 1;
        decoded[i] = bit ? 1.0f : -1.0f;
    }
    return decoded;
}

float BitQuantizer::ComputeDistance(std::span<const float> query, std::span<const uint8_t> codes) const {
    return DotBit(query, codes);
}

} // namespace pomai::core
