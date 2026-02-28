#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>

namespace pomai::util {

// Fast IEEE 754 float32 to float16 conversion (and vice versa)
// Distilled from high-performance implementations (like Maratyszcza's FP16 or FAISS)
// Optimized for throughput on modern CPUs.

inline uint16_t float32_to_float16(float f) {
    uint32_t f32;
    std::memcpy(&f32, &f, sizeof(uint32_t));
    
    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x007FFFFF;

    if (exponent <= -15) {
        if (exponent < -24) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x00800000;
        uint32_t shift = static_cast<uint32_t>(-14 - exponent);
        mantissa >>= shift;
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    } else if (exponent >= 16) {
        return static_cast<uint16_t>(sign | 0x7C00); // Infinity
    } else {
        return static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13));
    }
}

inline float float16_to_float32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF) << 13;

    if (exponent == 0) {
        if (mantissa == 0) {
            float f = 0;
            uint32_t f32 = sign;
            std::memcpy(&f, &f32, sizeof(float));
            return f;
        }
        // Subnormals
        while (!(mantissa & 0x00800000)) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= 0x007FFFFF;
        exponent += 127 - 15;
    } else if (exponent == 31) {
        exponent = 255; // Infinity or NaN
    } else {
        exponent += 127 - 15;
    }

    uint32_t f32 = sign | (exponent << 23) | mantissa;
    float f;
    std::memcpy(&f, &f32, sizeof(float));
    return f;
}

} // namespace pomai::util
