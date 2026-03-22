#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <cstddef>
#include "core/simd/vector_batch.h"

namespace pomai::core
{
    /**
     * Numeric mode: Prefer integer (int8/int16) paths for embedded to save memory and CPU.
     * When data is quantized (SQ8 or FP16), use DotSq8/L2SqSq8/DotFp16/L2SqFp16 instead of float Dot/L2Sq.
     * Segment scan and runtime already dispatch to these when quant_type is set.
     */
    enum class DistanceMetrics : uint8_t { 
        DOT, 
        L2SQ 
    };

    // ── Scalar distances (float; prefer SQ8/FP16 below when data is quantized) ──
    float Dot(std::span<const float> a, std::span<const float> b);
    float L2Sq(std::span<const float> a, std::span<const float> b);

    // Inner Product for SQ8 quantized codes (int8; preferred for embedded)
    float DotSq8(std::span<const float> query,
                 std::span<const uint8_t> codes,
                 float min_val, float inv_scale, float query_sum = 0.0f);

    /** L2 squared between float query and SQ8 data (min/max dequantize). ADC: dequantize in SIMD path then L2 via SimSIMD. */
    float L2SqSq8(std::span<const float> query,
                  std::span<const std::uint8_t> data,
                  float min_val, float max_val);

    // Distances for FP16 quantized codes (int16; preferred for embedded)
    float DotFp16(std::span<const float> query, std::span<const uint16_t> codes);
    float L2SqFp16(std::span<const float> query, std::span<const uint16_t> codes);

    // ── Binary Quantization (1-bit; optimized specifically for edge memory reduction) ──
    /** 32x smaller than float vectors. Encodes sign bit. */
    void BitQuantize(std::span<const float> vec, uint8_t* out_codes);
    /** Hamming distance but returning a dot-product-like score (higher=better). */
    float DotBit(std::span<const float> query, std::span<const uint8_t> codes);
    float HammingDist(std::span<const uint8_t> a, std::span<const uint8_t> b);

    // ── Batch distances ──
    void DotBatch(std::span<const float> query,
                  const float* db,
                  std::size_t n,
                  std::uint32_t dim,
                  float* results);

    void L2SqBatch(std::span<const float> query,
                   const float* db,
                   std::size_t n,
                   std::uint32_t dim,
                   float* results);

    /**
     * @brief Vectorized Batch Search (The "Orrify" Pattern).
     * Distilled from DuckDB's vectorized execution.
     */
    void SearchBatch(std::span<const float> query, const FloatBatch& batch, 
                     DistanceMetrics metric, float* results);

    // ── Setup ─────────────────────────────────────────────────────────────────
    void InitDistance();
}
