// distance.cc — SIMD distance kernels via third_party/simd (SimSIMD).
//
// Runtime SIMD dispatch: we enable SimSIMD's dynamic dispatch so that
// simsimd_dot_f32 / simsimd_l2sq_f32 use the best backend (AVX2, AVX512, NEON, SVE)
// at runtime. InitDistance() calls simsimd_capabilities() to warm the dispatch.
// DotSq8 has no SimSIMD equivalent and uses scalar. DotFp16/L2SqFp16 use SimSIMD f16.

#include "core/distance.h"

#include <cstring>
#include <mutex>
#include <vector>

#include "util/half_float.h"

// SimSIMD: by default uses compile-time dispatch (best for current arch).
// For portable binaries with runtime dispatch (AVX2/AVX512/NEON/SVE), build with
// -DSIMSIMD_DYNAMIC_DISPATCH=1 and link SimSIMD's dynamic dispatch object.
#if !((defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && defined(__ARM_FP16_FORMAT_IEEE)) && \
    !(((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && defined(__AVX512FP16__)))
#ifndef SIMSIMD_NATIVE_F16
#define SIMSIMD_NATIVE_F16 0
#endif
#endif

#if !((defined(__GNUC__) || defined(__clang__)) && (defined(__ARM_ARCH) || defined(__aarch64__)) && defined(__ARM_BF16_FORMAT_ALTERNATIVE)) && \
    !(((defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && defined(__AVX512BF16__)))
#ifndef SIMSIMD_NATIVE_BF16
#define SIMSIMD_NATIVE_BF16 0
#endif
#endif
#include "simd/simsimd.h"

namespace pomai::core {
namespace {

// ── Scalar fallback for DotSq8 (no SimSIMD equivalent) ──
float DotSq8Scalar(std::span<const float> q, std::span<const uint8_t> c,
                   float min_val, float inv_scale, float q_sum) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < q.size(); ++i)
        sum += q[i] * static_cast<float>(c[i]);
    return sum * inv_scale + q_sum * min_val;
}

// ── F32: use SimSIMD (with runtime dispatch when SIMSIMD_DYNAMIC_DISPATCH=1) ──
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

// FP16: SimSIMD f16 (query converted f32->f16).
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
    // Warm runtime dispatch: SimSIMD selects best backend (AVX512/AVX2/NEON/SVE) once.
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

// SQ8 L2: chunked dequantize in registers / small buffer only (no full vector in RAM).
// Process in SIMD-friendly chunks; accumulate L2^2 via SimSIMD per chunk.
constexpr std::size_t kSq8Chunk = 32u;

float L2SqSq8(std::span<const float> query,
              std::span<const std::uint8_t> data,
              float min_val, float max_val) {
    const std::size_t n = query.size();
    if (n == 0 || data.size() != n) return 0.0f;
    const float inv_scale = (max_val - min_val <= 1e-9f) ? 0.0f : ((max_val - min_val) / 255.0f);
    float sum_sq = 0.0f;
    std::size_t i = 0;
    float chunk_buf[kSq8Chunk];
    for (; i + kSq8Chunk <= n; i += kSq8Chunk) {
        for (std::size_t j = 0; j < kSq8Chunk; ++j)
            chunk_buf[j] = min_val + static_cast<float>(data[i + j]) * inv_scale;
        simsimd_distance_t d;
        simsimd_l2sq_f32(query.data() + i, chunk_buf, static_cast<simsimd_size_t>(kSq8Chunk), &d);
        sum_sq += static_cast<float>(d);
    }
    if (i < n) {
        const std::size_t rem = n - i;
        for (std::size_t j = 0; j < rem; ++j)
            chunk_buf[j] = min_val + static_cast<float>(data[i + j]) * inv_scale;
        simsimd_distance_t d;
        simsimd_l2sq_f32(query.data() + i, chunk_buf, static_cast<simsimd_size_t>(rem), &d);
        sum_sq += static_cast<float>(d);
    }
    return sum_sq;
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
        if (metric == DistanceMetrics::DOT)
            DotBatch(query, batch.data(), batch.size(), batch.dim(), results);
        else
            L2SqBatch(query, batch.data(), batch.size(), batch.dim(), results);
    } else if (batch.format() == VectorFormat::DICTIONARY) {
        const std::uint32_t* sel = batch.selection();
        for (std::uint32_t i = 0; i < batch.size(); ++i) {
            const float* v = batch.get_vector(sel[i]);
            simsimd_distance_t d;
            if (metric == DistanceMetrics::DOT)
                simsimd_dot_f32(query.data(), v, batch.dim(), &d);
            else
                simsimd_l2sq_f32(query.data(), v, batch.dim(), &d);
            results[i] = static_cast<float>(d);
        }
    }
}

}  // namespace pomai::core
