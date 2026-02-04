#include "core/distance.h"
#include <immintrin.h>
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
            float s = 0.0f;
            const std::size_t n = a.size();
            for (std::size_t i = 0; i < n; ++i)
                s += a[i] * b[i];
            return s;
        }

        float L2SqScalar(std::span<const float> a, std::span<const float> b)
        {
            float s = 0.0f;
            const std::size_t n = a.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                float d = a[i] - b[i];
                s += d * d;
            }
            return s;
        }

#if defined(__GNUC__) || defined(__clang__)
        __attribute__((target("avx2,fma")))
        float DotAvx(std::span<const float> a, std::span<const float> b)
        {
            const float *pa = a.data();
            const float *pb = b.data();
            std::size_t n = a.size();
            
            __m256 sum = _mm256_setzero_ps();
            std::size_t i = 0;
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

            __m256 sum = _mm256_setzero_ps();
            std::size_t i = 0;
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
#else
        // If not GCC/Clang, simple fallback (or rely on global flags if MSVC)
        float DotAvx(std::span<const float> a, std::span<const float> b) { return DotScalar(a, b); }
        float L2SqAvx(std::span<const float> a, std::span<const float> b) { return L2SqScalar(a, b); }
#endif

        using DistFn = float (*)(std::span<const float>, std::span<const float>);
        
        DistFn dot_fn = DotScalar;
        DistFn l2_fn = L2SqScalar;
        std::once_flag init_flag;

        void InitOnce()
        {
#if defined(__GNUC__) || defined(__clang__)
            if (__builtin_cpu_supports("avx2"))
            {
                dot_fn = DotAvx;
                l2_fn = L2SqAvx;
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
        std::call_once(init_flag, InitOnce);
        return dot_fn(a, b);
    }

    float L2Sq(std::span<const float> a, std::span<const float> b)
    {
        std::call_once(init_flag, InitOnce);
        return l2_fn(a, b);
    }
}
