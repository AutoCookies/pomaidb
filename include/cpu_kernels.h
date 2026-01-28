#pragma once

#include <cstddef>
#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace pomai::kernels
{
    static inline float L2SqrKernel(const float *a, const float *b, std::size_t dim)
    {
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();

        std::size_t i = 0;

        for (; i + 32 <= dim; i += 32)
        {
            __m256 v1 = _mm256_loadu_ps(a + i);
            __m256 q1 = _mm256_loadu_ps(b + i);
            __m256 d1 = _mm256_sub_ps(v1, q1);
            sum1 = _mm256_fmadd_ps(d1, d1, sum1);

            __m256 v2 = _mm256_loadu_ps(a + i + 8);
            __m256 q2 = _mm256_loadu_ps(b + i + 8);
            __m256 d2 = _mm256_sub_ps(v2, q2);
            sum2 = _mm256_fmadd_ps(d2, d2, sum2);

            __m256 v3 = _mm256_loadu_ps(a + i + 16);
            __m256 q3 = _mm256_loadu_ps(b + i + 16);
            __m256 d3 = _mm256_sub_ps(v3, q3);
            sum3 = _mm256_fmadd_ps(d3, d3, sum3);

            __m256 v4 = _mm256_loadu_ps(a + i + 24);
            __m256 q4 = _mm256_loadu_ps(b + i + 24);
            __m256 d4 = _mm256_sub_ps(v4, q4);
            sum4 = _mm256_fmadd_ps(d4, d4, sum4);
        }

        __m256 sum = _mm256_add_ps(_mm256_add_ps(sum1, sum2), _mm256_add_ps(sum3, sum4));

        for (; i + 8 <= dim; i += 8)
        {
            __m256 v = _mm256_loadu_ps(a + i);
            __m256 q = _mm256_loadu_ps(b + i);
            __m256 d = _mm256_sub_ps(v, q);
            sum = _mm256_fmadd_ps(d, d, sum);
        }

        alignas(32) float temp[8];
        _mm256_storeu_ps(temp, sum);
        float total = 0.0f;
        for (int k = 0; k < 8; ++k)
            total += temp[k];

        for (; i < dim; ++i)
        {
            float d = a[i] - b[i];
            total += d * d;
        }

        return total;
    }

    /**
     * @brief Computes Squared L2 distance between two vectors using AVX2.
     * * Optimizations:
     * - Loop unrolling (4x) to maximize instruction pipelining and hide latency.
     * - Fused Multiply-Add (FMA) for throughput.
     * - Unaligned loads (loadu) to accept arbitrary memory addresses (safe on Haswell+).
     * * @param a Pointer to first vector (aligned or unaligned).
     * @param b Pointer to second vector (aligned or unaligned).
     * @param dim Dimension of vectors.
     * @return float Squared Euclidean distance.
     */
    static inline float L2Sqr(const float *a, const float *b, std::size_t dim)
    {
        return L2SqrKernel(a, b, dim);
    }

    /**
     * @brief Computes squared L2 distances for a bucket of vectors against a query.
     * @param base Pointer to the first vector in the bucket (row-major).
     * @param query Pointer to query vector.
     * @param dim Vector dimension.
     * @param count Number of vectors in the bucket.
     * @param out_distances Output buffer of length `count`.
     */
    static inline void ScanBucketAVX2(const float *base,
                                      const float *query,
                                      std::size_t dim,
                                      std::size_t count,
                                      float *out_distances)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const float *v = base + i * dim;
            out_distances[i] = L2SqrKernel(v, query, dim);
        }
    }

    /**
     * @brief Computes squared L2 distance between two SQ8 quantized vectors.
     * @param qdata Pointer to quantized vector (uint8_t per dimension).
     * @param qquery Pointer to quantized query vector (uint8_t per dimension).
     * @param dim Dimension of vectors.
     * @return float Squared Euclidean distance.
     */
    static inline float L2Sqr_SQ8_AVX2(const std::uint8_t *qdata,
                                      const std::uint8_t *qquery,
                                      std::size_t dim)
    {
        __m256i acc32 = _mm256_setzero_si256();
        const __m256i clamp = _mm256_set1_epi8(127);
        const __m256i ones = _mm256_set1_epi16(1);

        std::size_t i = 0;
        for (; i + 32 <= dim; i += 32)
        {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qdata + i));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(qquery + i));

            __m256i diff1 = _mm256_subs_epu8(a, b);
            __m256i diff2 = _mm256_subs_epu8(b, a);
            __m256i diff = _mm256_adds_epu8(diff1, diff2);
            __m256i diff_clamped = _mm256_min_epu8(diff, clamp);

            __m256i squares16 = _mm256_maddubs_epi16(diff_clamped, diff_clamped);
            __m256i squares32 = _mm256_madd_epi16(squares16, ones);
            acc32 = _mm256_add_epi32(acc32, squares32);
        }

        alignas(32) std::uint32_t temp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc32);
        std::uint64_t total = 0;
        for (int k = 0; k < 8; ++k)
            total += temp[k];

        for (; i < dim; ++i)
        {
            int diff = static_cast<int>(qdata[i]) - static_cast<int>(qquery[i]);
            total += static_cast<std::uint64_t>(diff * diff);
        }

        return static_cast<float>(total);
    }

} // namespace pomai::kernels
