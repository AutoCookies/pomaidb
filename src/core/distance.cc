// distance.cc — SIMD distance kernels via third_party/simd (SimSIMD).
//
// All f32/f32 distance work (Dot, L2Sq, DotBatch, L2SqBatch) is delegated to
// SimSIMD, which provides runtime dispatch over AVX2/AVX512/NEON/SVE and serial.
// DotSq8 (f32 vs u8 with scale) has no direct SimSIMD equivalent and uses scalar.
// DotFp16/L2SqFp16 convert the f32 query to f16 and call SimSIMD dot_f16/l2sq_f16.

#include "core/distance.h"

#include <cstring>
#include <mutex>
#include <vector>

#include "util/half_float.h"

// SimSIMD: single header pulls in spatial (L2), dot (IP), types.
#include "simd/simsimd.h"

namespace pomai::core {
namespace {

// ── Scalar fallback for DotSq8 (f32 query vs u8 codes + scale; no SimSIMD equivalent) ──
float DotSq8Scalar(std::span<const float> q, std::span<const uint8_t> c,
                   float min_val, float inv_scale, float q_sum) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < q.size(); ++i)
        sum += q[i] * static_cast<float>(c[i]);
    return sum * inv_scale + q_sum * min_val;
}

// ── Wrappers using SimSIMD (result is double; we return float) ──
float DotSimSIMD(std::span<const float> a, std::span<const float> b) {
    const std::size_t n = a.size();
    if (n == 0) return 0.0f;
    simsimd_distance_t d;
    simsimd_dot_f32(a.data(), b.data(), static_cast<simsimd_size_t>(n), &d);
    return static_cast<float>(d);
}

float L2SqSimSIMD(std::span<const float> a, std::span<const float> b) {
    const std::size_t n = a.size();
    if (n == 0) return 0.0f;
    simsimd_distance_t d;
    simsimd_l2sq_f32(a.data(), b.data(), static_cast<simsimd_size_t>(n), &d);
    return static_cast<float>(d);
}

void DotBatchSimSIMD(std::span<const float> query,
                     const float* db, std::size_t n, std::uint32_t dim,
                     float* out) {
    for (std::size_t i = 0; i < n; ++i) {
        simsimd_distance_t d;
        simsimd_dot_f32(query.data(), db + i * dim, dim, &d);
        out[i] = static_cast<float>(d);
    }
}

void L2SqBatchSimSIMD(std::span<const float> query,
                      const float* db, std::size_t n, std::uint32_t dim,
                      float* out) {
    for (std::size_t i = 0; i < n; ++i) {
        simsimd_distance_t d;
        simsimd_l2sq_f32(query.data(), db + i * dim, dim, &d);
        out[i] = static_cast<float>(d);
    }
}

// FP16: SimSIMD expects both vectors as f16. We have f32 query and u16 (fp16) codes.
// Convert query to f16 then call SimSIMD. Codes are passed as simsimd_f16_t* (same bit layout as uint16_t).
float DotFp16SimSIMD(std::span<const float> q, std::span<const uint16_t> c) {
    const std::size_t n = q.size();
    if (n == 0) return 0.0f;
    std::vector<simsimd_f16_t> q_f16(n);
    for (std::size_t i = 0; i < n; ++i)
        simsimd_f32_to_f16(q[i], &q_f16[i]);
    simsimd_distance_t d;
    simsimd_dot_f16(q_f16.data(), reinterpret_cast<const simsimd_f16_t*>(c.data()),
                    static_cast<simsimd_size_t>(n), &d);
    return static_cast<float>(d);
}

float L2SqFp16SimSIMD(std::span<const float> q, std::span<const uint16_t> c) {
    const std::size_t n = q.size();
    if (n == 0) return 0.0f;
    std::vector<simsimd_f16_t> q_f16(n);
    for (std::size_t i = 0; i < n; ++i)
        simsimd_f32_to_f16(q[i], &q_f16[i]);
    simsimd_distance_t d;
    simsimd_l2sq_f16(q_f16.data(), reinterpret_cast<const simsimd_f16_t*>(c.data()),
                     static_cast<simsimd_size_t>(n), &d);
    return static_cast<float>(d);
}

std::once_flag init_flag;

void InitOnce() {
    // SimSIMD uses runtime dispatch internally; optionally warm capabilities.
    (void)simsimd_capabilities();
}

}  // namespace

void InitDistance() { std::call_once(init_flag, InitOnce); }

float Dot(std::span<const float> a, std::span<const float> b) {
    return DotSimSIMD(a, b);
}

float L2Sq(std::span<const float> a, std::span<const float> b) {
    return L2SqSimSIMD(a, b);
}

float DotSq8(std::span<const float> q, std::span<const uint8_t> c,
             float min_val, float inv_scale, float q_sum) {
    return DotSq8Scalar(q, c, min_val, inv_scale, q_sum);
}

float DotFp16(std::span<const float> q, std::span<const uint16_t> c) {
    return DotFp16SimSIMD(q, c);
}

float L2SqFp16(std::span<const float> q, std::span<const uint16_t> c) {
    return L2SqFp16SimSIMD(q, c);
}

void DotBatch(std::span<const float> query,
              const float* db, std::size_t n, std::uint32_t dim,
              float* results) {
    DotBatchSimSIMD(query, db, n, dim, results);
}

void L2SqBatch(std::span<const float> query,
               const float* db, std::size_t n, std::uint32_t dim,
               float* results) {
    L2SqBatchSimSIMD(query, db, n, dim, results);
}

void SearchBatch(std::span<const float> query, const FloatBatch& batch,
                 DistanceMetrics metric, float* results) {
    if (batch.format() == VectorFormat::FLAT) {
        if (metric == DistanceMetrics::DOT) {
            DotBatch(query, batch.data(), batch.size(), batch.dim(), results);
        } else {
            L2SqBatch(query, batch.data(), batch.size(), batch.dim(), results);
        }
    } else if (batch.format() == VectorFormat::DICTIONARY) {
        const uint32_t* sel = batch.selection();
        for (uint32_t i = 0; i < batch.size(); ++i) {
            const float* v = batch.get_vector(sel[i]);
            if (metric == DistanceMetrics::DOT) {
                simsimd_distance_t d;
                simsimd_dot_f32(query.data(), v, batch.dim(), &d);
                results[i] = static_cast<float>(d);
            } else {
                simsimd_distance_t d;
                simsimd_l2sq_f32(query.data(), v, batch.dim(), &d);
                results[i] = static_cast<float>(d);
            }
        }
    }
}

}  // namespace pomai::core
