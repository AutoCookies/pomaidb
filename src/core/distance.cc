#include "core/distance.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define POMAI_X86_SIMD 1
#else
#define POMAI_X86_SIMD 0
#endif
#if POMAI_X86_SIMD
#include <immintrin.h>
#endif
#include <mutex>
#include <memory>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace pomai::core
{
    namespace
    {
        float DotScalar(std::span<const float> a, std::span<const float> b)
        {
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            const std::size_t n = a.size();
            std::size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                s0 += a[i] * b[i];
                s1 += a[i + 1] * b[i + 1];
                s2 += a[i + 2] * b[i + 2];
                s3 += a[i + 3] * b[i + 3];
            }
            float s = s0 + s1 + s2 + s3;
            for (; i < n; ++i) {
                s += a[i] * b[i];
            }
            return s;
        }

        float L2SqScalar(std::span<const float> a, std::span<const float> b)
        {
            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            const std::size_t n = a.size();
            std::size_t i = 0;
            for (; i + 4 <= n; i += 4)
            {
                float d0 = a[i] - b[i];
                float d1 = a[i + 1] - b[i + 1];
                float d2 = a[i + 2] - b[i + 2];
                float d3 = a[i + 3] - b[i + 3];
                s0 += d0 * d0;
                s1 += d1 * d1;
                s2 += d2 * d2;
                s3 += d3 * d3;
            }
            float s = s0 + s1 + s2 + s3;
            for (; i < n; ++i)
            {
                float d = a[i] - b[i];
                s += d * d;
            }
            return s;
        }

        float DotSq8Scalar(std::span<const float> query, std::span<const uint8_t> codes, float min_val, float inv_scale, float query_sum)
        {
            float sum_qc = 0.0f;
            const size_t dim = query.size();
            for (size_t i = 0; i < dim; ++i) {
                sum_qc += query[i] * static_cast<float>(codes[i]);
            }
            return sum_qc * inv_scale + query_sum * min_val;
        }

#if POMAI_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
        __attribute__((target("avx2,fma")))
        float DotAvx(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data();
            const float *pb = b.data();
            std::size_t n = a.size();
            
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16)
            {
                __m256 va0 = _mm256_loadu_ps(pa + i);
                __m256 vb0 = _mm256_loadu_ps(pb + i);
                __m256 va1 = _mm256_loadu_ps(pa + i + 8);
                __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
                sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
                sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
            }
            __m256 sum = _mm256_add_ps(sum0, sum1);
            for (; i + 8 <= n; i += 8)
            {
                __m256 va = _mm256_loadu_ps(pa + i);
                __m256 vb = _mm256_loadu_ps(pb + i);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            
            // Horizontal sum
            // (There are faster ways but this is readable)
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            float s = 0.0f;
            for (int k = 0; k < 8; ++k) s += temp[k];

            // Tail
            for (; i < n; ++i)
                s += pa[i] * pb[i];
            return s;
        }

        __attribute__((target("avx2,fma")))
        float L2SqAvx(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data();
            const float *pb = b.data();
            std::size_t n = a.size();

            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16)
            {
                __m256 va0 = _mm256_loadu_ps(pa + i);
                __m256 vb0 = _mm256_loadu_ps(pb + i);
                __m256 va1 = _mm256_loadu_ps(pa + i + 8);
                __m256 vb1 = _mm256_loadu_ps(pb + i + 8);
                __m256 d0 = _mm256_sub_ps(va0, vb0);
                __m256 d1 = _mm256_sub_ps(va1, vb1);
                sum0 = _mm256_fmadd_ps(d0, d0, sum0);
                sum1 = _mm256_fmadd_ps(d1, d1, sum1);
            }
            __m256 sum = _mm256_add_ps(sum0, sum1);
            for (; i + 8 <= n; i += 8)
            {
                __m256 va = _mm256_loadu_ps(pa + i);
                __m256 vb = _mm256_loadu_ps(pb + i);
                __m256 d = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(d, d, sum);
            }

            float temp[8];
            _mm256_storeu_ps(temp, sum);
            float s = 0.0f;
            for (int k = 0; k < 8; ++k) s += temp[k];

            for (; i < n; ++i)
            {
                float d = pa[i] - pb[i];
                s += d * d;
            }
            return s;
        }

        __attribute__((target("avx2,fma")))
        float DotSq8Avx(std::span<const float> query, std::span<const uint8_t> codes, float min_val, float inv_scale, float query_sum)
        {
            const float *pq = query.data();
            const uint8_t *pc = codes.data();
            std::size_t n = query.size();
            
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            
            std::size_t i = 0;
            // Unroll by 16 (2 AVX registers)
            for (; i + 16 <= n; i += 16)
            {
                // Load 16 uint8s (128 bits total)
                __m128i c_chars = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pc + i));
                
                // Expand lower 8 bytes to 32-bit uints then floats
                __m256i c_ints0 = _mm256_cvtepu8_epi32(c_chars);
                __m256 c_floats0 = _mm256_cvtepi32_ps(c_ints0);
                
                // Expand upper 8 bytes. Shift right by 8 bytes.
                __m128i c_chars_hi = _mm_bsrli_si128(c_chars, 8);
                __m256i c_ints1 = _mm256_cvtepu8_epi32(c_chars_hi);
                __m256 c_floats1 = _mm256_cvtepi32_ps(c_ints1);
                
                // Vector dot product
                __m256 vq0 = _mm256_loadu_ps(pq + i);
                __m256 vq1 = _mm256_loadu_ps(pq + i + 8);
                
                sum0 = _mm256_fmadd_ps(vq0, c_floats0, sum0);
                sum1 = _mm256_fmadd_ps(vq1, c_floats1, sum1);
            }
            __m256 sum = _mm256_add_ps(sum0, sum1);
            
            // Unroll by 8 (1 AVX register)
            for (; i + 8 <= n; i += 8)
            {
                // _mm_loadl_epi64 to load 8 bytes into lower 64 bits of 128-bit reg
                __m128i c_chars = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pc + i));
                __m256i c_ints = _mm256_cvtepu8_epi32(c_chars);
                __m256 c_floats = _mm256_cvtepi32_ps(c_ints);
                
                __m256 vq = _mm256_loadu_ps(pq + i);
                sum = _mm256_fmadd_ps(vq, c_floats, sum);
            }
            
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            float s = 0.0f;
            for (int k = 0; k < 8; ++k) s += temp[k];

            // Tail processing
            for (; i < n; ++i)
            {
                s += pq[i] * static_cast<float>(pc[i]);
            }
            return s * inv_scale + query_sum * min_val;
        }
#else
        // If not GCC/Clang, simple fallback (or rely on global flags if MSVC)
        float DotAvx(std::span<const float> a, std::span<const float> b) { return DotScalar(a, b); }
        float L2SqAvx(std::span<const float> a, std::span<const float> b) { return L2SqScalar(a, b); }
        float DotSq8Avx(std::span<const float> query, std::span<const uint8_t> codes, float min_val, float inv_scale, float query_sum) { return DotSq8Scalar(query, codes, min_val, inv_scale, query_sum); }
#endif

        using DistFn = float (*)(std::span<const float>, std::span<const float>);
        using DotSq8Fn = float (*)(std::span<const float>, std::span<const uint8_t>, float, float, float);
        
        DistFn dot_fn = DotScalar;
        DistFn l2_fn = L2SqScalar;
        DotSq8Fn dot_sq8_fn = DotSq8Scalar;
        std::once_flag init_flag;

        void InitOnce()
        {
#if POMAI_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
            if (__builtin_cpu_supports("avx2"))
            {
                dot_fn = DotAvx;
                l2_fn = L2SqAvx;
                dot_sq8_fn = DotSq8Avx;
            }
#endif
        }
    }

    void InitDistance()
    {
        std::call_once(init_flag, InitOnce);
    }

    float Dot(std::span<const float> a, std::span<const float> b)
    {
        return dot_fn(a, b);
    }

    float L2Sq(std::span<const float> a, std::span<const float> b)
    {
        return l2_fn(a, b);
    }

    float DotSq8(std::span<const float> query, std::span<const uint8_t> codes, float min_val, float inv_scale, float query_sum)
    {
        return dot_sq8_fn(query, codes, min_val, inv_scale, query_sum);
    }
}
