#pragma once
#include <cstdint>
#include <span>

namespace pomai::core
{
    // Inner Product (Dot)
    float Dot(std::span<const float> a, std::span<const float> b);

    // L2 Squared
    float L2Sq(std::span<const float> a, std::span<const float> b);

    // Inner Product for SQ8 Quantized Codes
    float DotSq8(std::span<const float> query, std::span<const uint8_t> codes, float min_val, float inv_scale);

    // Setup/Init dispatch (optional, called automatically or via static block)
    void InitDistance();
}
