#pragma once

#include <cstdint>
#include <span>
#include <vector>
#include <cstddef>
#include "core/simd/vector_batch.h"

namespace pomai::core
{
    enum class DistanceMetrics : uint8_t { 
        DOT, 
        L2SQ 
    };

    // ── Scalar distances ──────────────────────────────────────────────────────
    float Dot(std::span<const float> a, std::span<const float> b);
    float L2Sq(std::span<const float> a, std::span<const float> b);

    // Inner Product for SQ8 quantized codes
    float DotSq8(std::span<const float> query,
                 std::span<const uint8_t> codes,
                 float min_val, float inv_scale, float query_sum = 0.0f);

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
