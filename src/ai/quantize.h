#pragma once
// ai/quantize.h
// Simple quantization utilities used by tests and QuantizedL2Space.
//
// NOTE: These helpers assume input floats are in [0,1]. For production you'd
// want per-vector or per-dataset scale+zero-point metadata.

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace pomai::ai::quantize
{

    // Quantize float[dim] with range [0,1] to uint8_t[dim] mapping 0->0, 1->255
    inline void quantize_u8(const float *src, uint8_t *dst, size_t dim)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            float v = src[i];
            if (v < 0.0f)
                v = 0.0f;
            if (v > 1.0f)
                v = 1.0f;
            uint8_t q = static_cast<uint8_t>(std::lround(v * 255.0f));
            dst[i] = q;
        }
    }

    // Quantize to packed 4-bit: output buffer must be (dim+1)/2 bytes.
    // Packs two 4-bit values per byte: low nibble = idx0, high nibble = idx1.
    // Quantization: round(src*15) -> [0..15].
    inline void quantize_u4(const float *src, uint8_t *dst, size_t dim)
    {
        size_t i = 0;
        size_t out_idx = 0;
        while (i + 1 < dim)
        {
            float v0 = src[i];
            float v1 = src[i + 1];
            v0 = std::min(1.0f, std::max(0.0f, v0));
            v1 = std::min(1.0f, std::max(0.0f, v1));
            uint8_t q0 = static_cast<uint8_t>(std::lround(v0 * 15.0f)) & 0xF;
            uint8_t q1 = static_cast<uint8_t>(std::lround(v1 * 15.0f)) & 0xF;
            dst[out_idx++] = static_cast<uint8_t>((q1 << 4) | q0);
            i += 2;
        }
        if (i < dim)
        {
            float v0 = src[i];
            v0 = std::min(1.0f, std::max(0.0f, v0));
            uint8_t q0 = static_cast<uint8_t>(std::lround(v0 * 15.0f)) & 0xF;
            dst[out_idx++] = static_cast<uint8_t>(q0);
        }
    }

    // Dequantize helpers for use inside dist kernel
    inline float dequant_u8(uint8_t q) { return static_cast<float>(q) / 255.0f; }
    inline float dequant_u4_low(uint8_t byte) { return static_cast<float>(byte & 0xF) / 15.0f; }
    inline float dequant_u4_high(uint8_t byte) { return static_cast<float>((byte >> 4) & 0xF) / 15.0f; }

} // namespace pomai::ai::quantize