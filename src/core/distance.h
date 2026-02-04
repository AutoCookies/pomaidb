#pragma once
#include <cstdint>
#include <span>

namespace pomai::core
{
    // Inner Product (Dot)
    float Dot(std::span<const float> a, std::span<const float> b);

    // L2 Squared
    float L2Sq(std::span<const float> a, std::span<const float> b);

    // Setup/Init dispatch (optional, called automatically or via static block)
    void InitDistance();
}
