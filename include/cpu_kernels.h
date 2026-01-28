#pragma once

#include <cstddef>
#include <immintrin.h>
#include <cmath>

namespace pomai::kernels
{

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
        // Accumulators for parallel dependency chains.
        // We use 4 accumulators to break data dependency on 'sum' register,
        // allowing CPU to execute multiple FMA instructions per cycle.
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();

        std::size_t i = 0;

        // Main loop: Process 32 floats (4 AVX registers) per iteration.
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

        // Reduce 4 accumulators into 1
        __m256 sum = _mm256_add_ps(_mm256_add_ps(sum1, sum2), _mm256_add_ps(sum3, sum4));

        // Handle remaining blocks of 8
        for (; i + 8 <= dim; i += 8)
        {
            __m256 v = _mm256_loadu_ps(a + i);
            __m256 q = _mm256_loadu_ps(b + i);
            __m256 d = _mm256_sub_ps(v, q);
            sum = _mm256_fmadd_ps(d, d, sum);
        }

        // Horizontal reduction: Sum 8 floats inside the YMM register
        // This is slow, but done only once per vector.
        float temp[8];
        _mm256_storeu_ps(temp, sum);
        float total = 0.0f;
        for (int k = 0; k < 8; ++k)
            total += temp[k];

        // Scalar tail for dimensions not divisible by 8
        for (; i < dim; ++i)
        {
            float d = a[i] - b[i];
            total += d * d;
        }

        return total;
    }

} // namespace pomai::kernels