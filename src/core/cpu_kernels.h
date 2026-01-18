#pragma once
// name: src/core/cpu_kernels.h
// Enhanced CPU kernels supporting multiple stored data types (float32, float64, int32, int8, float16).
// Uses src/core/types.h for a centralized DataType enum.

#include "src/core/types.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <cstdlib>
#include <mutex> // std::call_once
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__GNUC__) || defined(__clang__)
#define POMAI_HAS_BUILTIN_CPU_SUPPORTS 1
#else
#define POMAI_HAS_BUILTIN_CPU_SUPPORTS 0
#endif

// --------------------- Kernel prototypes and types ------------------------
using L2Func = float (*)(const float *a, const float *b, size_t dim);
using DotFunc = float (*)(const float *a, const float *b, size_t dim);
using FmaFunc = void (*)(float *acc, const float *val, float scale, size_t dim);
using PackedSignedDotFunc = double (*)(const uint8_t *sign_bytes, const float *pvec, uint32_t bits);

// --------------------- Helper: float16 conversion (IEEE 754 half) -----------
static inline float fp16_to_fp32(uint16_t h) noexcept
{
    // Convert IEEE-754 binary16 to float (binary32)
    uint32_t s = (h >> 15) & 0x00000001u;
    uint32_t e = (h >> 10) & 0x0000001Fu;
    uint32_t m = h & 0x03FFu;

    uint32_t out;
    if (e == 0)
    {
        if (m == 0)
        {
            out = s << 31;
        }
        else
        {
            // subnormal
            while ((m & 0x0400u) == 0)
            {
                m <<= 1;
                e -= 1;
            }
            e += 1;
            m &= ~0x0400u;
            uint32_t exp = (e + (127 - 15)) & 0xFFu;
            out = (s << 31) | (exp << 23) | (m << 13);
        }
    }
    else if (e == 31)
    {
        // Inf/NaN
        out = (s << 31) | 0x7F800000u | (m << 13);
    }
    else
    {
        uint32_t exp = (e + (127 - 15)) & 0xFFu;
        out = (s << 31) | (exp << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

static inline uint16_t fp32_to_fp16(float f) noexcept
{
    // Simple (non-rounding-optimized) conversion; good for storage.
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (exp <= 0)
    {
        // subnormal or zero
        return sign;
    }
    else if (exp >= 31)
    {
        // overflow -> inf
        return sign | 0x7C00u;
    }
    uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    return out;
}

// --------------------- Scalar reference kernels (float32 query vs float32 storage) ---------------------------
static inline float l2sq_scalar(const float *a, const float *b, size_t dim)
{
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i)
    {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float dot_scalar(const float *a, const float *b, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i) acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(acc);
}

static inline void fma_scalar(float *acc, const float *val, float scale, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) acc[i] += val[i] * scale;
}

static double packed_signed_dot_scalar(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    if (!sign_bytes || !pvec || bits == 0) return 0.0;
    uint32_t bytes = (bits + 7) / 8;
    double sum_bit = 0.0;
    double sum_all = 0.0;
    uint32_t idx = 0;
    for (uint32_t b = 0; b < bytes; ++b)
    {
        uint8_t v = sign_bytes[b];
        uint32_t chunk_len = (b == bytes - 1) ? (bits - (bytes - 1) * 8) : 8;
        if (b == bytes - 1) v &= static_cast<uint8_t>((1u << chunk_len) - 1u);

        for (uint32_t k = 0; k < chunk_len; ++k, ++idx)
        {
            float pv = pvec[idx];
            sum_all += pv;
            if ((v >> k) & 1u) sum_bit += pv;
        }
    }
    return 2.0 * sum_bit - sum_all;
}

// --------------------- AVX2 kernels (float32) --------------------------------------
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>

__attribute__((target("avx2,fma"))) static float l2sq_avx2(const float *a, const float *b, size_t dim)
{
    size_t i = 0;
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(va, vb);
        vsum = _mm256_fmadd_ps(d, d, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i)
    {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

__attribute__((target("avx2,fma"))) static float dot_avx2(const float *a, const float *b, size_t dim)
{
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    double total = 0.0;
    for (int k = 0; k < 8; ++k) total += tmp[k];
    for (; i < dim; ++i) total += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(total);
}

__attribute__((target("avx2,fma"))) static void fma_avx2(float *acc, const float *val, float scale, size_t dim)
{
    __m256 vscale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8)
    {
        __m256 vacc = _mm256_loadu_ps(acc + i);
        __m256 vval = _mm256_loadu_ps(val + i);
        vacc = _mm256_fmadd_ps(vval, vscale, vacc);
        _mm256_storeu_ps(acc + i, vacc);
    }
    for (; i < dim; ++i)
        acc[i] += val[i] * scale;
}
#else
static float l2sq_avx2(const float *a, const float *b, size_t dim) { return l2sq_scalar(a, b, dim); }
static float dot_avx2(const float *a, const float *b, size_t dim) { return dot_scalar(a, b, dim); }
static void fma_avx2(float *acc, const float *val, float scale, size_t dim) { fma_scalar(acc, val, scale, dim); }
#endif

// AVX512 placeholders (fallback to avx2/scalar)
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f"))) static float l2sq_avx512(const float *a, const float *b, size_t dim) { return l2sq_avx2(a,b,dim); }
__attribute__((target("avx512f"))) static float dot_avx512(const float *a, const float *b, size_t dim) { return dot_avx2(a,b,dim); }
__attribute__((target("avx512f"))) static void fma_avx512(float *acc, const float *val, float scale, size_t dim) { fma_avx2(acc,val,scale,dim); }
#else
static float l2sq_avx512(const float *a, const float *b, size_t dim) { return l2sq_avx2(a,b,dim); }
static float dot_avx512(const float *a, const float *b, size_t dim) { return dot_avx2(a,b,dim); }
static void fma_avx512(float *acc, const float *val, float scale, size_t dim) { fma_avx2(acc,val,scale,dim); }
#endif

// ARM NEON placeholders
static float l2sq_neon(const float *a, const float *b, size_t dim) { return l2sq_scalar(a,b,dim); }
static float dot_neon(const float *a, const float *b, size_t dim) { return dot_scalar(a,b,dim); }
static void fma_neon(float *acc, const float *val, float scale, size_t dim) { fma_scalar(acc,val,scale,dim); }

// --------------------- Packed-signed-dot SIMD kernels ---------------------
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
static inline const float *get_packed_mask_table()
{
    static float table[256][8];
    static std::once_flag flag;
    std::call_once(flag, [](){
        for (int v = 0; v < 256; ++v)
            for (int k = 0; k < 8; ++k)
                table[v][k] = ((v >> k) & 1) ? 1.0f : 0.0f;
    });
    return &table[0][0];
}

__attribute__((target("avx2,fma"))) static double packed_signed_dot_avx2(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    if (!sign_bytes || !pvec || bits == 0) return 0.0;
    const float *mask_table = get_packed_mask_table();
    uint32_t full_chunks = bits / 8;
    uint32_t rem = bits - full_chunks * 8;
    const uint8_t *sb = sign_bytes;
    const float *pv = pvec;

    __m256 vsum_bit = _mm256_setzero_ps();
    __m256 vsum_all = _mm256_setzero_ps();

    for (uint32_t i = 0; i < full_chunks; ++i)
    {
        uint8_t v = sb[i];
        const float *mask_ptr = mask_table + (static_cast<size_t>(v) * 8);
        __m256 vmask = _mm256_loadu_ps(mask_ptr);
        __m256 vp = _mm256_loadu_ps(pv + i * 8);
        vsum_bit = _mm256_fmadd_ps(vmask, vp, vsum_bit);
        vsum_all = _mm256_add_ps(vsum_all, vp);
    }

    alignas(32) float tmp_bit[8];
    alignas(32) float tmp_all[8];
    _mm256_store_ps(tmp_bit, vsum_bit);
    _mm256_store_ps(tmp_all, vsum_all);
    double sum_bit = 0.0, sum_all = 0.0;
    for (int i = 0; i < 8; ++i) { sum_bit += tmp_bit[i]; sum_all += tmp_all[i]; }

    uint32_t idx = full_chunks * 8;
    if (rem)
    {
        uint8_t v = sb[full_chunks];
        uint8_t mask = static_cast<uint8_t>((1u << rem) - 1u);
        v &= mask;
        for (uint32_t k = 0; k < rem; ++k, ++idx)
        {
            float pvv = pvec[idx];
            sum_all += pvv;
            if ((v >> k) & 1u) sum_bit += pvv;
        }
    }

    return 2.0 * sum_bit - sum_all;
}
#else
static double packed_signed_dot_avx2(const uint8_t *sign_bytes, const float *pvec, uint32_t bits) { return packed_signed_dot_scalar(sign_bytes, pvec, bits); }
#endif

#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f"))) static double packed_signed_dot_avx512(const uint8_t *s, const float *p, uint32_t b) { return packed_signed_dot_avx2(s,p,b); }
#else
static double packed_signed_dot_avx512(const uint8_t *s, const float *p, uint32_t b) { return packed_signed_dot_avx2(s,p,b); }
#endif

static double packed_signed_dot_neon(const uint8_t *s, const float *p, uint32_t b) { return packed_signed_dot_scalar(s,p,b); }

// --------------------- Typed L2 kernels (query float32 vs multiple stored types) --------------------

// float32 storage
static inline float l2sq_f32(const float *q, const float *s, size_t dim)
{
#if POMAI_HAS_BUILTIN_CPU_SUPPORTS
    if (__builtin_cpu_supports("avx2"))
        return l2sq_avx2(q, s, dim);
#endif
    return l2sq_scalar(q, s, dim);
}

// float64 (double) storage
static inline float l2sq_f64_scalar(const float *q, const double *s, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double d = static_cast<double>(q[i]) - s[i];
        acc += d * d;
    }
    return static_cast<float>(acc);
}

#if defined(__AVX2__)
__attribute__((target("avx2,fma"))) static inline float l2sq_f64_avx2(const float *q, const double *s, size_t dim)
{
    size_t i = 0;
    __m256d vacc = _mm256_setzero_pd();
    for (; i + 4 <= dim; i += 4)
    {
        __m256d vs = _mm256_loadu_pd(s + i);
        __m128 vq_f = _mm_loadu_ps(q + i);
        __m256d vq = _mm256_cvtps_pd(vq_f);
        __m256d vd = _mm256_sub_pd(vq, vs);
        vacc = _mm256_fmadd_pd(vd, vd, vacc);
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, vacc);
    double acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < dim; ++i)
    {
        double d = static_cast<double>(q[i]) - s[i];
        acc += d * d;
    }
    return static_cast<float>(acc);
}
#endif

static inline float l2sq_f64(const float *q, const double *s, size_t dim)
{
#if defined(__AVX2__)
    if (__builtin_cpu_supports("avx2"))
        return l2sq_f64_avx2(q, s, dim);
#endif
    return l2sq_f64_scalar(q, s, dim);
}

// int32 storage
static inline float l2sq_i32_scalar(const float *q, const int32_t *s, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double d = static_cast<double>(q[i]) - static_cast<double>(s[i]);
        acc += d * d;
    }
    return static_cast<float>(acc);
}

#if defined(__AVX2__)
__attribute__((target("avx2,fma"))) static inline float l2sq_i32_avx2(const float *q, const int32_t *s, size_t dim)
{
    size_t i = 0;
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8)
    {
        __m256 vq = _mm256_loadu_ps(q + i);
        __m128i vi_low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i)); // 4 ints
        __m128i vi_high = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s + i + 4));
        __m128 vlow = _mm_cvtepi32_ps(vi_low);
        __m128 vhigh = _mm_cvtepi32_ps(vi_high);
        __m256 vs = _mm256_set_m128(vhigh, vlow);
        __m256 vd = _mm256_sub_ps(vq, vs);
        vsum = _mm256_fmadd_ps(vd, vd, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    float acc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i)
    {
        float d = q[i] - static_cast<float>(s[i]);
        acc += d * d;
    }
    return acc;
}
#endif

static inline float l2sq_i32(const float *q, const int32_t *s, size_t dim)
{
#if defined(__AVX2__)
    if (__builtin_cpu_supports("avx2"))
        return l2sq_i32_avx2(q, s, dim);
#endif
    return l2sq_i32_scalar(q, s, dim);
}

// int8 storage (signed)
static inline float l2sq_i8_scalar(const float *q, const int8_t *s, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        double d = static_cast<double>(q[i]) - static_cast<double>(s[i]);
        acc += d * d;
    }
    return static_cast<float>(acc);
}

// float16 storage (stored as uint16_t)
static inline float l2sq_f16_scalar(const float *q, const uint16_t *s, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i)
    {
        float sv = fp16_to_fp32(s[i]);
        double d = static_cast<double>(q[i]) - static_cast<double>(sv);
        acc += d * d;
    }
    return static_cast<float>(acc);
}

// Dispatcher: query is float*, stored pointer is void* interpreted by dtype
static inline float l2sq_mixed(const float *q, const void *stored, size_t dim, pomai::core::DataType dtype)
{
    if (!q || !stored || dim == 0) return std::numeric_limits<float>::infinity();
    switch (dtype)
    {
    case pomai::core::DataType::FLOAT32:
        return l2sq_f32(q, reinterpret_cast<const float *>(stored), dim);
    case pomai::core::DataType::FLOAT64:
        return l2sq_f64(q, reinterpret_cast<const double *>(stored), dim);
    case pomai::core::DataType::INT32:
        return l2sq_i32(q, reinterpret_cast<const int32_t *>(stored), dim);
    case pomai::core::DataType::INT8:
        return l2sq_i8_scalar(q, reinterpret_cast<const int8_t *>(stored), dim);
    case pomai::core::DataType::FLOAT16:
        return l2sq_f16_scalar(q, reinterpret_cast<const uint16_t *>(stored), dim);
    default:
        return l2sq_f32(q, reinterpret_cast<const float *>(stored), dim);
    }
}

// Typed dot helpers (float query dot stored double/int)
static inline float dot_f64(const float *q, const double *s, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i)
        acc += static_cast<double>(q[i]) * s[i];
    return static_cast<float>(acc);
}

static inline float dot_i32(const float *q, const int32_t *s, size_t dim)
{
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i)
        acc += static_cast<double>(q[i]) * static_cast<double>(s[i]);
    return static_cast<float>(acc);
}

// --------------------- Dispatcher & init --------------------------------
namespace pomai::core::kernels_internal {
    inline L2Func impl_l2sq = l2sq_scalar;
    inline DotFunc impl_dot = dot_scalar;
    inline FmaFunc impl_fma = fma_scalar;
    inline PackedSignedDotFunc impl_packed = packed_signed_dot_scalar;
}

inline void pomai_init_cpu_kernels()
{
    static std::once_flag flag;
    std::call_once(flag, [](){
        const char *force = std::getenv("POMAI_FORCE_KERNEL");
        if (force && std::strlen(force) > 0)
        {
            std::string_view fv(force);
            if (fv == "avx512")
            {
                pomai::core::kernels_internal::impl_l2sq = l2sq_avx512;
                pomai::core::kernels_internal::impl_dot = dot_avx512;
                pomai::core::kernels_internal::impl_fma = fma_avx512;
                pomai::core::kernels_internal::impl_packed = packed_signed_dot_avx512;
                std::clog << "[CPU] Forced kernel: avx512\n";
            }
            else if (fv == "avx2")
            {
                pomai::core::kernels_internal::impl_l2sq = l2sq_avx2;
                pomai::core::kernels_internal::impl_dot = dot_avx2;
                pomai::core::kernels_internal::impl_fma = fma_avx2;
                pomai::core::kernels_internal::impl_packed = packed_signed_dot_avx2;
                std::clog << "[CPU] Forced kernel: avx2\n";
            }
            else if (fv == "neon")
            {
                pomai::core::kernels_internal::impl_l2sq = l2sq_neon;
                pomai::core::kernels_internal::impl_dot = dot_neon;
                pomai::core::kernels_internal::impl_fma = fma_neon;
                pomai::core::kernels_internal::impl_packed = packed_signed_dot_neon;
                std::clog << "[CPU] Forced kernel: neon\n";
            }
            else
            {
                std::clog << "[CPU] Forced kernel: scalar\n";
            }
            return;
        }

#if POMAI_HAS_BUILTIN_CPU_SUPPORTS
        if (__builtin_cpu_supports("avx512f"))
        {
            pomai::core::kernels_internal::impl_l2sq = l2sq_avx512;
            pomai::core::kernels_internal::impl_dot = dot_avx512;
            pomai::core::kernels_internal::impl_fma = fma_avx512;
            pomai::core::kernels_internal::impl_packed = packed_signed_dot_avx512;
            std::clog << "[CPU] Detected AVX-512 -> using avx512 kernels\n";
            return;
        }
        if (__builtin_cpu_supports("avx2"))
        {
            pomai::core::kernels_internal::impl_l2sq = l2sq_avx2;
            pomai::core::kernels_internal::impl_dot = dot_avx2;
            pomai::core::kernels_internal::impl_fma = fma_avx2;
            pomai::core::kernels_internal::impl_packed = packed_signed_dot_avx2;
            std::clog << "[CPU] Detected AVX2 -> using avx2 kernels\n";
            return;
        }
#endif

        std::clog << "[CPU] Using scalar kernels\n";
    });
}

inline L2Func get_pomai_l2sq_kernel() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_l2sq; }
inline DotFunc get_pomai_dot_kernel() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_dot; }
inline FmaFunc get_pomai_fma_kernel() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_fma; }
inline PackedSignedDotFunc get_pomai_packed_signed_dot() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_packed; }

// Export wrappers (unchanged API for float32 callers)
inline float l2sq(const float *a, const float *b, size_t dim)
{
    return get_pomai_l2sq_kernel()(a, b, dim);
}

inline float pomai_dot(const float *a, const float *b, size_t dim)
{
    return get_pomai_dot_kernel()(a, b, dim);
}

inline void pomai_fma(float *acc, const float *val, float scale, size_t dim)
{
    get_pomai_fma_kernel()(acc, val, scale, dim);
}

inline double pomai_packed_signed_dot(const uint8_t *s, const float *p, uint32_t b)
{
    return get_pomai_packed_signed_dot()(s, p, b);
}
