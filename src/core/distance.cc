// distance.cc — SIMD-dispatched distance kernels for PomaiDB.
//
// Phase 1 update:
//  - Added NEON dispatch path for ARM (RPi 5, Android) via sse2neon.h
//  - Added DotBatch / L2SqBatch for bulk multi-vector distance (HNSW traversal)
//
// Dispatch priority: AVX2+FMA (x86) > NEON (ARM) > scalar (WASM/fallback)

#include "core/distance.h"

// ── Platform detection ────────────────────────────────────────────────────────
#if defined(__x86_64__) || defined(_M_X64)
#  define POMAI_X86_SIMD 1
#  define POMAI_ARM_SIMD 0
#elif defined(__aarch64__) || defined(__arm__)
#  define POMAI_X86_SIMD 0
#  define POMAI_ARM_SIMD 1
#else
#  define POMAI_X86_SIMD 0
#  define POMAI_ARM_SIMD 0
#endif

#if POMAI_X86_SIMD
#  include <immintrin.h>
#endif

// ARM NEON: native intrinsics (maps SSE/AVX2 concepts to NEON)
#if POMAI_ARM_SIMD
#  include <arm_neon.h>
#endif

#include <cstring>
#include <mutex>
#include <memory>
#include "util/half_float.h"

#if defined(__GNUC__) || defined(__clang__)
#  if POMAI_X86_SIMD
#    include <cpuid.h>
#  endif
#endif

namespace pomai::core
{
    namespace
    {
        // ── Scalar fallbacks ──────────────────────────────────────────────────
        float DotScalar(std::span<const float> a, std::span<const float> b)
        {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const std::size_t n = a.size();
            std::size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                s0 += a[i]   * b[i];
                s1 += a[i+1] * b[i+1];
                s2 += a[i+2] * b[i+2];
                s3 += a[i+3] * b[i+3];
            }
            float s = s0 + s1 + s2 + s3;
            for (; i < n; ++i) s += a[i] * b[i];
            return s;
        }

        float L2SqScalar(std::span<const float> a, std::span<const float> b)
        {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const std::size_t n = a.size();
            std::size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                float d0 = a[i]-b[i], d1 = a[i+1]-b[i+1],
                      d2 = a[i+2]-b[i+2], d3 = a[i+3]-b[i+3];
                s0 += d0*d0; s1 += d1*d1; s2 += d2*d2; s3 += d3*d3;
            }
            float s = s0 + s1 + s2 + s3;
            for (; i < n; ++i) { float d = a[i]-b[i]; s += d*d; }
            return s;
        }

        float DotSq8Scalar(std::span<const float> q, std::span<const uint8_t> c,
                           float min_val, float inv_scale, float q_sum)
        {
            float sum = 0.0f;
            for (std::size_t i = 0; i < q.size(); ++i)
                sum += q[i] * static_cast<float>(c[i]);
            return sum * inv_scale + q_sum * min_val;
        }

        float DotFp16Scalar(std::span<const float> q, std::span<const uint16_t> c)
        {
            float sum = 0.0f;
            for (std::size_t i = 0; i < q.size(); ++i)
                sum += q[i] * pomai::util::float16_to_float32(c[i]);
            return sum;
        }

        float L2SqFp16Scalar(std::span<const float> q, std::span<const uint16_t> c)
        {
            float sum = 0.0f;
            for (std::size_t i = 0; i < q.size(); ++i) {
                float d = q[i] - pomai::util::float16_to_float32(c[i]);
                sum += d * d;
            }
            return sum;
        }

        // Scalar batch
        void DotBatchScalar(std::span<const float> query,
                            const float* db, std::size_t n, std::uint32_t dim,
                            float* out)
        {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = DotScalar(query, {db + i * dim, dim});
        }

        void L2SqBatchScalar(std::span<const float> query,
                             const float* db, std::size_t n, std::uint32_t dim,
                             float* out)
        {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = L2SqScalar(query, {db + i * dim, dim});
        }

        // ── x86 AVX2+FMA kernels (compile-time target, runtime dispatched) ───
#if POMAI_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
        __attribute__((target("avx2,fma")))
        float DotAvx(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data(), *pb = b.data();
            std::size_t n = a.size();
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa+i),   _mm256_loadu_ps(pb+i),   s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa+i+8), _mm256_loadu_ps(pb+i+8), s1);
            }
            __m256 sum = _mm256_add_ps(s0, s1);
            for (; i + 8 <= n; i += 8)
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(pa+i), _mm256_loadu_ps(pb+i), sum);
            float t[8]; _mm256_storeu_ps(t, sum);
            float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for (; i < n; ++i) s += pa[i]*pb[i];
            return s;
        }

        __attribute__((target("avx2,fma")))
        float L2SqAvx(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data(), *pb = b.data();
            std::size_t n = a.size();
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(pa+i),   _mm256_loadu_ps(pb+i));
                __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(pa+i+8), _mm256_loadu_ps(pb+i+8));
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
            }
            __m256 sum = _mm256_add_ps(s0, s1);
            for (; i + 8 <= n; i += 8) {
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(pa+i), _mm256_loadu_ps(pb+i));
                sum = _mm256_fmadd_ps(d, d, sum);
            }
            float t[8]; _mm256_storeu_ps(t, sum);
            float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for (; i < n; ++i) { float d = pa[i]-pb[i]; s += d*d; }
            return s;
        }

        __attribute__((target("avx2,fma")))
        float DotSq8Avx(std::span<const float> q, std::span<const uint8_t> c,
                        float min_val, float inv_scale, float q_sum)
        {
            const float *pq = q.data(); const uint8_t *pc = c.data();
            std::size_t n = q.size();
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m128i cc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i));
                __m256 cf0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(cc));
                __m256 cf1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_bsrli_si128(cc, 8)));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq+i),   cf0, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pq+i+8), cf1, s1);
            }
            __m256 sum = _mm256_add_ps(s0, s1);
            for (; i + 8 <= n; i += 8) {
                __m256 cf = _mm256_cvtepi32_ps(
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pc+i))));
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(pq+i), cf, sum);
            }
            float t[8]; _mm256_storeu_ps(t, sum);
            float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for (; i < n; ++i) s += pq[i] * static_cast<float>(pc[i]);
            return s * inv_scale + q_sum * min_val;
        }

        __attribute__((target("avx2,fma,f16c")))
        float DotFp16Avx(std::span<const float> q, std::span<const uint16_t> c)
        {
            const float *pq = q.data(); const uint16_t *pc = c.data();
            std::size_t n = q.size();
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            std::size_t i = 0;
            // Process 16 elements at a time (8 per cvtph call)
            for (; i + 16 <= n; i += 16) {
                __m256 cf0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i)));
                __m256 cf1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i+8)));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pq+i),   cf0, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pq+i+8), cf1, s1);
            }
            __m256 sum = _mm256_add_ps(s0, s1);
            for (; i + 8 <= n; i += 8) {
                __m256 cf = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i)));
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(pq+i), cf, sum);
            }
            float t[8]; _mm256_storeu_ps(t, sum);
            float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for (; i < n; ++i) s += pq[i] * pomai::util::float16_to_float32(pc[i]);
            return s;
        }

        __attribute__((target("avx2,fma,f16c")))
        float L2SqFp16Avx(std::span<const float> q, std::span<const uint16_t> c)
        {
            const float *pq = q.data(); const uint16_t *pc = c.data();
            std::size_t n = q.size();
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m256 cf0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i)));
                __m256 cf1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i+8)));
                __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(pq+i),   cf0);
                __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(pq+i+8), cf1);
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
            }
            __m256 sum = _mm256_add_ps(s0, s1);
            for (; i + 8 <= n; i += 8) {
                __m256 cf = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pc+i)));
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(pq+i), cf);
                sum = _mm256_fmadd_ps(d, d, sum);
            }
            float t[8]; _mm256_storeu_ps(t, sum);
            float s = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for (; i < n; ++i) {
                float d = pq[i] - pomai::util::float16_to_float32(pc[i]);
                s += d * d;
            }
            return s;
        }

        // Batch versions — call per-row scalar to allow auto-vectorisation
        // at the outer loop level (compiler can hoist the query broadcast).
        __attribute__((target("avx2,fma")))
        void DotBatchAvx(std::span<const float> query,
                         const float* db, std::size_t n, std::uint32_t dim,
                         float* out)
        {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = DotAvx(query, {db + i*dim, dim});
        }

        __attribute__((target("avx2,fma")))
        void L2SqBatchAvx(std::span<const float> query,
                          const float* db, std::size_t n, std::uint32_t dim,
                          float* out)
        {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = L2SqAvx(query, {db + i*dim, dim});
        }
#else
        // Non-GCC/Clang x86 fallbacks
        float DotAvx(std::span<const float> a, std::span<const float> b)
            { return DotScalar(a, b); }
        float L2SqAvx(std::span<const float> a, std::span<const float> b)
            { return L2SqScalar(a, b); }
        float DotSq8Avx(std::span<const float> q, std::span<const uint8_t> c,
                        float min_val, float inv_scale, float q_sum)
            { return DotSq8Scalar(q, c, min_val, inv_scale, q_sum); }
        float DotFp16Avx(std::span<const float> q, std::span<const uint16_t> c)
            { return DotFp16Scalar(q, c); }
        float L2SqFp16Avx(std::span<const float> q, std::span<const uint16_t> c)
            { return L2SqFp16Scalar(q, c); }
        void DotBatchAvx(std::span<const float> q, const float* db,
                         std::size_t n, std::uint32_t dim, float* out)
            { DotBatchScalar(q, db, n, dim, out); }
        void L2SqBatchAvx(std::span<const float> q, const float* db,
                          std::size_t n, std::uint32_t dim, float* out)
            { L2SqBatchScalar(q, db, n, dim, out); }
#endif

        // ── ARM NEON kernels ──────────────────────────────────────────────────
#if POMAI_ARM_SIMD
        float DotNeon(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data(), *pb = b.data();
            std::size_t n = a.size();
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                s0 = vmlaq_f32(s0, vld1q_f32(pa+i),   vld1q_f32(pb+i));
                s1 = vmlaq_f32(s1, vld1q_f32(pa+i+4), vld1q_f32(pb+i+4));
            }
            float32x4_t sum = vaddq_f32(s0, s1);
            for (; i + 4 <= n; i += 4)
                sum = vmlaq_f32(sum, vld1q_f32(pa+i), vld1q_f32(pb+i));
            float s = vaddvq_f32(sum);
            for (; i < n; ++i) s += pa[i]*pb[i];
            return s;
        }

        float L2SqNeon(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data(), *pb = b.data();
            std::size_t n = a.size();
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                float32x4_t d0 = vsubq_f32(vld1q_f32(pa+i),   vld1q_f32(pb+i));
                float32x4_t d1 = vsubq_f32(vld1q_f32(pa+i+4), vld1q_f32(pb+i+4));
                s0 = vmlaq_f32(s0, d0, d0);
                s1 = vmlaq_f32(s1, d1, d1);
            }
            float32x4_t sum = vaddq_f32(s0, s1);
            for (; i + 4 <= n; i += 4) {
                float32x4_t d = vsubq_f32(vld1q_f32(pa+i), vld1q_f32(pb+i));
                sum = vmlaq_f32(sum, d, d);
            }
            float s = vaddvq_f32(sum);
            for (; i < n; ++i) { float d=pa[i]-pb[i]; s += d*d; }
            return s;
        }

        float DotSq8Neon(std::span<const float> q, std::span<const uint8_t> c,
                         float min_val, float inv_scale, float q_sum)
        {
            const float *pq = q.data(); const uint8_t *pc = c.data();
            std::size_t n = q.size();
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                uint8x8_t  cu  = vld1_u8(pc+i);
                uint16x8_t cu16= vmovl_u8(cu);
                float32x4_t cf0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(cu16)));
                float32x4_t cf1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(cu16)));
                s0 = vmlaq_f32(s0, vld1q_f32(pq+i),   cf0);
                s1 = vmlaq_f32(s1, vld1q_f32(pq+i+4), cf1);
            }
            float s = vaddvq_f32(vaddq_f32(s0, s1));
            for (; i < n; ++i) s += pq[i] * static_cast<float>(pc[i]);
            return s * inv_scale + q_sum * min_val;
        }

        float DotFp16Neon(std::span<const float> q, std::span<const uint16_t> c)
        {
            const float *pq = q.data(); const uint16_t *pc = c.data();
            std::size_t n = q.size();
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                float32x4_t cf0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(pc+i)));
                float32x4_t cf1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(pc+i+4)));
                s0 = vmlaq_f32(s0, vld1q_f32(pq+i),   cf0);
                s1 = vmlaq_f32(s1, vld1q_f32(pq+i+4), cf1);
            }
            float s = vaddvq_f32(vaddq_f32(s0, s1));
            for (; i < n; ++i) s += pq[i] * pomai::util::float16_to_float32(pc[i]);
            return s;
        }

        float L2SqFp16Neon(std::span<const float> q, std::span<const uint16_t> c)
        {
            const float *pq = q.data(); const uint16_t *pc = c.data();
            std::size_t n = q.size();
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                float32x4_t cf0 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(pc+i)));
                float32x4_t cf1 = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(pc+i+4)));
                float32x4_t d0 = vsubq_f32(vld1q_f32(pq+i),   cf0);
                float32x4_t d1 = vsubq_f32(vld1q_f32(pq+i+4), cf1);
                s0 = vmlaq_f32(s0, d0, d0);
                s1 = vmlaq_f32(s1, d1, d1);
            }
            float s = vaddvq_f32(vaddq_f32(s0, s1));
            for (; i < n; ++i) {
                float d = pq[i] - pomai::util::float16_to_float32(pc[i]);
                s += d * d;
            }
            return s;
        }

        void DotBatchNeon(std::span<const float> query,
                          const float* db, std::size_t n, std::uint32_t dim,
                          float* out)
        {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = DotNeon(query, {db + i*dim, dim});
        }

        void L2SqBatchNeon(std::span<const float> query,
                           const float* db, std::size_t n, std::uint32_t dim,
                           float* out)
        {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = L2SqNeon(query, {db + i*dim, dim});
        }
#endif // POMAI_ARM_SIMD

        // ── Dispatch tables ───────────────────────────────────────────────────
        using DistFn    = float (*)(std::span<const float>, std::span<const float>);
        using DotSq8Fn  = float (*)(std::span<const float>, std::span<const uint8_t>,
                                    float, float, float);
        using DistFp16Fn = float (*)(std::span<const float>, std::span<const uint16_t>);
        using BatchDistFn = void (*)(std::span<const float>, const float*,
                                    std::size_t, std::uint32_t, float*);

        DistFn      dot_fn      = DotScalar;
        DistFn      l2_fn       = L2SqScalar;
        DotSq8Fn    dot_sq8_fn  = DotSq8Scalar;
        DistFp16Fn  dot_fp16_fn = DotFp16Scalar;
        DistFp16Fn  l2_fp16_fn  = L2SqFp16Scalar;
        BatchDistFn dot_batch_fn  = DotBatchScalar;
        BatchDistFn l2_batch_fn   = L2SqBatchScalar;
        std::once_flag init_flag;

        void InitOnce()
        {
#if POMAI_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
            if (__builtin_cpu_supports("avx2")) {
                dot_fn       = DotAvx;
                l2_fn        = L2SqAvx;
                dot_sq8_fn   = DotSq8Avx;
                dot_batch_fn = DotBatchAvx;
                l2_batch_fn  = L2SqBatchAvx;
                dot_fp16_fn  = DotFp16Avx;
                l2_fp16_fn   = L2SqFp16Avx;
                return;
            }
#endif
#if POMAI_ARM_SIMD
            // NEON is always present on AArch64. On 32-bit ARMv7 it's optional;
            // for edge targets (RPi 5, Android arm64) we assume AArch64.
            dot_fn       = DotNeon;
            l2_fn        = L2SqNeon;
            dot_sq8_fn   = DotSq8Neon;
            dot_batch_fn = DotBatchNeon;
            l2_batch_fn  = L2SqBatchNeon;
            dot_fp16_fn  = DotFp16Neon;
            l2_fp16_fn   = L2SqFp16Neon;
#endif
        }
    } // anonymous namespace

    // ── Public API ────────────────────────────────────────────────────────────
    void InitDistance()  { std::call_once(init_flag, InitOnce); }

    float Dot(std::span<const float> a, std::span<const float> b)
        { return dot_fn(a, b); }

    float L2Sq(std::span<const float> a, std::span<const float> b)
        { return l2_fn(a, b); }

    float DotSq8(std::span<const float> q, std::span<const uint8_t> c,
                 float min_val, float inv_scale, float q_sum)
        { return dot_sq8_fn(q, c, min_val, inv_scale, q_sum); }

    float DotFp16(std::span<const float> q, std::span<const uint16_t> c)
        { return dot_fp16_fn(q, c); }

    float L2SqFp16(std::span<const float> q, std::span<const uint16_t> c)
        { return l2_fp16_fn(q, c); }

    void DotBatch(std::span<const float> query,
                  const float* db, std::size_t n, std::uint32_t dim,
                  float* results)
        { dot_batch_fn(query, db, n, dim, results); }

    void L2SqBatch(std::span<const float> query,
                   const float* db, std::size_t n, std::uint32_t dim,
                   float* results)
        { l2_batch_fn(query, db, n, dim, results); }

    /**
     * @brief Vectorized Batch Search (The "Orrify" Pattern).
     * Distilled from DuckDB's vectorized execution.
     */
    void SearchBatch(std::span<const float> query, const FloatBatch& batch, 
                     DistanceMetrics metric, float* results) {
        if (batch.format() == VectorFormat::FLAT) {
            if (metric == DistanceMetrics::DOT) {
                DotBatch(query, batch.data(), batch.size(), batch.dim(), results);
            } else {
                L2SqBatch(query, batch.data(), batch.size(), batch.dim(), results);
            }
        } else if (batch.format() == VectorFormat::DICTIONARY) {
            // Indirection (Selection Vector) processing
            const uint32_t* sel = batch.selection();
            for (uint32_t i = 0; i < batch.size(); ++i) {
                const float* v = batch.get_vector(sel[i]);
                if (metric == DistanceMetrics::DOT) {
                    results[i] = Dot(query, {v, batch.dim()});
                } else {
                    results[i] = L2Sq(query, {v, batch.dim()});
                }
            }
        }
    }

} // namespace pomai::core
