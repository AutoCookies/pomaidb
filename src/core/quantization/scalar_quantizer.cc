#include "pomai/quantization/scalar_quantizer.h"
#include "core/distance.h"

#include <algorithm>
#include <cmath>

namespace pomai::core {

ScalarQuantizer8Bit::ScalarQuantizer8Bit(size_t dim)
    : dim_(dim) {}

pomai::Status ScalarQuantizer8Bit::Train(std::span<const float> data, size_t num_vectors) {
    if (num_vectors == 0 || dim_ == 0) {
        return pomai::Status::InvalidArgument("Empty dimensions or vectors for training");
    }

    const size_t total_elements = num_vectors * dim_;
    if (data.size() < total_elements) {
        return pomai::Status::InvalidArgument("Data span is smaller than num_vectors * dim");
    }

    // Mathematical logic:
    // Single pass to find the dataset's global minimum and maximum bounds.
    // The loop is completely contiguous and unrolled by the compiler into FAST minps/maxps operations.
    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 1; i < total_elements; ++i) {
        const float val = data[i];
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }

    global_min_ = min_val;
    const float range = max_val - min_val;

    // Prevent division by zero if all dataset values are roughly identical
    if (range <= 1e-6f) {
        global_scale_ = 0.0f;
        global_inv_scale_ = 0.0f;
    } else {
        // We map the continuous float range [min_val, max_val] into [0, 255].
        global_scale_ = 255.0f / range;
        global_inv_scale_ = range / 255.0f;
    }

    is_trained_ = true;
    return pomai::Status::Ok();
}

std::vector<uint8_t> ScalarQuantizer8Bit::Encode(std::span<const float> vector) const {
    if (!is_trained_ || vector.size() != dim_) {
        // Return empty buffer on failure as no exception/status return is supported here
        return {};
    }

    std::vector<uint8_t> codes(dim_);
    
    // Mathematical logic:
    // f_norm = (x - min) * scale 
    // Clamp to [0, 255] strictly to avoid undefined behavior on float-to-uint cast 
    // for outliers not seen during training.
    const float min_val = global_min_;
    const float scale = global_scale_;

    for (size_t i = 0; i < dim_; ++i) {
        float f = (vector[i] - min_val) * scale;
        
        // Branchless min/max clamp. Maps seamlessly to CPU SIMD limits.
        // Rounding adjustment (+0.5f) reduces the mean squared quantization error.
        f = std::max(0.0f, std::min(255.0f, f + 0.5f));
        
        codes[i] = static_cast<uint8_t>(f);
    }

    return codes;
}

std::vector<float> ScalarQuantizer8Bit::Decode(std::span<const uint8_t> codes) const {
    if (!is_trained_ || codes.size() != dim_) {
        return {};
    }

    std::vector<float> decoded(dim_);
    const float min_val = global_min_;
    const float inv_scale = global_inv_scale_;

    // Mathematical logic:
    // x_approx = (code * range / 255.0) + min
    // Fully branchless array mapping capable of native vectorization.
    for (size_t i = 0; i < dim_; ++i) {
        decoded[i] = min_val + static_cast<float>(codes[i]) * inv_scale;
    }

    return decoded;
}

float ScalarQuantizer8Bit::ComputeDistance(std::span<const float> query, std::span<const uint8_t> codes) const {
    if (!is_trained_ || query.size() != dim_ || codes.size() != dim_) {
        return -1.0f; // Soft error state since exceptions are disallowed
    }

    // Call out to the optimized global dispatcher which dynamically employs AVX2 if supported
    return pomai::core::DotSq8(query, codes, global_min_, global_inv_scale_);
}

void ScalarQuantizer8Bit::LoadState(float min_val, float inv_scale) {
    global_min_ = min_val;
    global_inv_scale_ = inv_scale;
    // Scale is only used for encoding, which we won't do on historically loaded MMaps, 
    // but we can reconstruct loosely if needed, or simply leave uninitialized.
    if (inv_scale > 1e-6f) {
        global_scale_ = 1.0f / inv_scale;
    } else {
        global_scale_ = 0.0f;
    }
    is_trained_ = true;
}

} // namespace pomai::core
