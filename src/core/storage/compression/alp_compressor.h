#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <span>

namespace pomai::core {

/**
 * @brief ALP: Adaptive Lossless floating-point Compression.
 * Distilled from DuckDB's ALP algorithm.
 * 
 * Stores floats as (Value * 10^Exp) + Base, converted to integers.
 */
class ALPCompressor {
public:
    struct Config {
        int8_t exponent;
        float base;
    };

    /**
     * @brief Encodes a span of floats into 64-bit integers.
     * This is an "Edge-Simplified" version of the full ALP algorithm.
     */
    static Config Encode(std::span<const float> input, std::span<int64_t> output) {
        if (input.empty()) return {0, 0.0f};

        // 1. Find the min value to use as a base
        float min_val = input[0];
        for (float f : input) if (f < min_val) min_val = f;

        // 2. Estimate optimal exponent (finding max decimal precision)
        // For vector DBs, we often see 4-6 decimal places.
        int8_t best_exp = 6; 
        float factor = std::pow(10.0f, static_cast<float>(best_exp));

        // 3. Transform: Int = (Float - Base) * 10^Exp
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = static_cast<int64_t>(std::round((input[i] - min_val) * factor));
        }

        return {best_exp, min_val};
    }

    /**
     * @brief Decodes integers back to floats.
     */
    static void Decode(std::span<const int64_t> input, std::span<float> output, Config cfg) {
        float inv_factor = 1.0f / std::pow(10.0f, static_cast<float>(cfg.exponent));
        for (std::size_t i = 0; i < input.size(); ++i) {
            output[i] = static_cast<float>(input[i]) * inv_factor + cfg.base;
        }
    }
};

} // namespace pomai::core
