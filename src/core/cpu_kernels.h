// name: src/core/cpu_kernels.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <cstdlib>
#include <mutex> // Required for std::call_once

// Platform/compiler helpers
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

// --------------------- Scalar reference kernels ---------------------------
static inline float l2sq_scalar(const float *a, const float *b, size_t dim)
{
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static inline float dot_scalar(const float *a, const float *b, size_t dim)
{
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
    return sum;
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

// --------------------- AVX2 kernels --------------------------------------
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
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) total += a[i] * b[i];
    return total;
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
// Fallback stubs
static float l2sq_avx2(const float *a, const float *b, size_t dim) { return l2sq_scalar(a, b, dim); }
static float dot_avx2(const float *a, const float *b, size_t dim) { return dot_scalar(a, b, dim); }
static void fma_avx2(float *acc, const float *val, float scale, size_t dim) { fma_scalar(acc, val, scale, dim); }
#endif

// --------------------- AVX-512 placeholders -------------------------------
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f"))) static float l2sq_avx512(const float *a, const float *b, size_t dim)
{
    return l2sq_avx2(a, b, dim); // Placeholder: fallback to avx2
}
__attribute__((target("avx512f"))) static float dot_avx512(const float *a, const float *b, size_t dim)
{
    return dot_avx2(a, b, dim);
}
__attribute__((target("avx512f"))) static void fma_avx512(float *acc, const float *val, float scale, size_t dim)
{
    fma_avx2(acc, val, scale, dim);
}
#else
static float l2sq_avx512(const float *a, const float *b, size_t dim) { return l2sq_scalar(a, b, dim); }
static float dot_avx512(const float *a, const float *b, size_t dim) { return dot_avx2(a, b, dim); }
static void fma_avx512(float *acc, const float *val, float scale, size_t dim) { fma_avx2(acc, val, scale, dim); }
#endif

// --------------------- ARM NEON placeholders -------------------------------
static float l2sq_neon(const float *a, const float *b, size_t dim) { return l2sq_scalar(a, b, dim); }
static float dot_neon(const float *a, const float *b, size_t dim) { return dot_scalar(a, b, dim); }
static void fma_neon(float *acc, const float *val, float scale, size_t dim) { fma_scalar(acc, val, scale, dim); }

// --------------------- Packed-signed-dot SIMD kernels ---------------------
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
// Helper: build mask-table (256 entries x 8 floats) on first use.
static inline const float *get_packed_mask_table()
{
    static float table[256][8];
    // [FIX] std::call_once ensures this table is built safely even if multiple threads call this function.
    static std::once_flag flag;
    std::call_once(flag, [](){
        for (int v = 0; v < 256; ++v)
        {
            for (int k = 0; k < 8; ++k)
                table[v][k] = ((v >> k) & 1) ? 1.0f : 0.0f;
        }
    });
    return &table[0][0];
}

__attribute__((target("avx2,fma"))) static double packed_signed_dot_avx2(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    if (!sign_bytes || !pvec || bits == 0) return 0.0;
    const float *mask_table = get_packed_mask_table(); // row-major [256][8]
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
        __m256 vmask = _mm256_loadu_ps(mask_ptr);            // 0/1 mask
        __m256 vp = _mm256_loadu_ps(pv + i * 8);             // pvec chunk
        vsum_bit = _mm256_fmadd_ps(vmask, vp, vsum_bit);     // vsum_bit += mask * vp
        vsum_all = _mm256_add_ps(vsum_all, vp);              // vsum_all += vp
    }

    // horizontal sum for vector accumulators
    alignas(32) float tmp_bit[8];
    alignas(32) float tmp_all[8];
    _mm256_store_ps(tmp_bit, vsum_bit);
    _mm256_store_ps(tmp_all, vsum_all);
    double sum_bit = 0.0;
    double sum_all = 0.0;
    for (int i = 0; i < 8; ++i) { sum_bit += tmp_bit[i]; sum_all += tmp_all[i]; }

    // handle remainder (if any)
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

    double signed_sum = 2.0 * sum_bit - sum_all;
    return signed_sum;
}
#else
static double packed_signed_dot_avx2(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    return packed_signed_dot_scalar(sign_bytes, pvec, bits);
}
#endif

#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f"))) static double packed_signed_dot_avx512(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    return packed_signed_dot_avx2(sign_bytes, pvec, bits);
}
#else
static double packed_signed_dot_avx512(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    return packed_signed_dot_avx2(sign_bytes, pvec, bits);
}
#endif

static double packed_signed_dot_neon(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    return packed_signed_dot_scalar(sign_bytes, pvec, bits);
}

// --------------------- Dispatcher & init --------------------------------
// [FIX] Namespace encapsulation to avoid global pollution, keeping state internal.
namespace pomai::core::kernels_internal {
    // [FIX] 'inline' variables ensure a single instance across all translation units (C++17)
    inline L2Func impl_l2sq = l2sq_scalar; 
    inline DotFunc impl_dot = dot_scalar;
    inline FmaFunc impl_fma = fma_scalar;
    inline PackedSignedDotFunc impl_packed = packed_signed_dot_scalar;
}

inline void pomai_init_cpu_kernels()
{
    // [FIX] std::call_once to guarantee this runs exactly once, thread-safely.
    static std::once_flag flag;
    std::call_once(flag, [](){
        // Environment override: POMAI_FORCE_KERNEL=scalar|avx2|avx512|neon
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
                // Scalar
                std::clog << "[CPU] Forced kernel: scalar\n";
            }
            return;
        }

#if POMAI_HAS_BUILTIN_CPU_SUPPORTS
        // prefer avx512, then avx2, then scalar
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

        // Fallback: scalar
        std::clog << "[CPU] Using scalar kernels\n";
    });
}

// Accessors: Initialize once, then return the function pointer.
inline L2Func get_pomai_l2sq_kernel() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_l2sq; }
inline DotFunc get_pomai_dot_kernel() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_dot; }
inline FmaFunc get_pomai_fma_kernel() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_fma; }
inline PackedSignedDotFunc get_pomai_packed_signed_dot() { pomai_init_cpu_kernels(); return pomai::core::kernels_internal::impl_packed; }

// [FIX] Helper function restored for main.cc compatibility
inline const char *kernel_name_from_ptr(L2Func f)
{
    if (f == l2sq_scalar) return "scalar";
    if (f == l2sq_avx2) return "avx2";
    if (f == l2sq_avx512) return "avx512";
    if (f == l2sq_neon) return "neon";
    return "unknown";
}

// Export inline symbols (Reflects current dispatch function, accessible globally)
// These maintain the same API you had, but backed by the safe internal pointers.
inline L2Func pomai_l2sq_kernel = get_pomai_l2sq_kernel();
inline DotFunc pomai_dot_kernel = get_pomai_dot_kernel();
inline FmaFunc pomai_fma_kernel = get_pomai_fma_kernel();
inline PackedSignedDotFunc pomai_packed_signed_dot_kernel = get_pomai_packed_signed_dot();

// --------------------- Convenience Wrappers ---------------------
// 10/10 Perf: Indirect call overhead only (no if checks here)
static inline float l2sq(const float *a, const float *b, size_t dim)
{
    return pomai_l2sq_kernel(a, b, dim);
}

static inline float pomai_dot(const float *a, const float *b, size_t dim)
{
    return pomai_dot_kernel(a, b, dim);
}

static inline void pomai_fma(float *acc, const float *val, float scale, size_t dim)
{
    pomai_fma_kernel(acc, val, scale, dim);
}

static inline double pomai_packed_signed_dot(const uint8_t *s, const float *p, uint32_t b)
{
    return pomai_packed_signed_dot_kernel(s, p, b);
}