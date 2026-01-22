#pragma once
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <mutex>

namespace pomai::core::algo
{
    namespace detail
    {
        // [BIG TECH]: AVX2 + FMA + 4-Way Unrolling + Unaligned Load
        __attribute__((target("avx2,fma"))) inline float l2sq_avx2_fma_v20(const float *a, const float *b, size_t dim)
        {
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 32 <= dim; i += 32)
            {
                __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
                __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
                __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
                __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
                s2 = _mm256_fmadd_ps(d2, d2, s2);
                s3 = _mm256_fmadd_ps(d3, d3, s3);
            }
            s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
            __m128 vlow = _mm256_castps256_ps128(s0);
            __m128 vhigh = _mm256_extractf128_ps(s0, 1);
            vlow = _mm_add_ps(vlow, vhigh);
            __m128 shuf = _mm_movehdup_ps(vlow);
            __m128 sums = _mm_add_ps(vlow, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float res = _mm_cvtss_f32(sums);
            for (; i < dim; ++i)
            {
                float d = a[i] - b[i];
                res += d * d;
            }
            return res;
        }
    } // detail

    class BlitzKernels
    {
    public:
        using L2Func = float (*)(const float *, const float *, size_t);
        static inline L2Func l2sq = nullptr;
        static void init()
        {
            static std::once_flag flag;
            std::call_once(flag, []()
                           {
#if defined(__x86_64__)
                               if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
                                   l2sq = detail::l2sq_avx2_fma_v20;
                               else
                                   l2sq = [](const float *a, const float *b, size_t dim)
                                   {
                                       float s = 0;
                                       for (size_t i = 0; i < dim; ++i)
                                       {
                                           float d = a[i] - b[i];
                                           s += d * d;
                                       }
                                       return s;
                                   };
#endif
                           });
        }
    };
    inline float blitz_l2sq(const float *a, const float *b, size_t dim)
    {
        if (!BlitzKernels::l2sq)
            BlitzKernels::init();
        return BlitzKernels::l2sq(a, b, dim);
    }
}