#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace pomai::core
{
    // ── Scalar distances ──────────────────────────────────────────────────────
    // Inner Product (Dot)
    float Dot(std::span<const float> a, std::span<const float> b);

    // L2 Squared
    float L2Sq(std::span<const float> a, std::span<const float> b);

    // Inner Product for SQ8 quantized codes
    float DotSq8(std::span<const float> query,
                 std::span<const uint8_t> codes,
                 float min_val, float inv_scale, float query_sum = 0.0f);

    // ── Batch distances (Phase 1 — for HNSW graph traversal) ─────────────────
    // Compute Dot(query, db[i]) for i in [0, n) → results[i]
    // db layout: row-major, each row is `dim` floats.
    void DotBatch(std::span<const float> query,
                  const float* db,
                  std::size_t n,
                  std::uint32_t dim,
                  float* results);

    // Compute L2Sq(query, db[i]) for i in [0, n) → results[i]
    void L2SqBatch(std::span<const float> query,
                   const float* db,
                   std::size_t n,
                   std::uint32_t dim,
                   float* results);

    // ── Setup ─────────────────────────────────────────────────────────────────
    // Runtime dispatch: AVX2 on x86-64, NEON on ARM. Call once at startup.
    void InitDistance();
}
