// name: src/core/cpu_kernels.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <cstdlib>

// Platform/compiler helpers
#if defined(__GNUC__) || defined(__clang__)
#define POMAI_HAS_BUILTIN_CPU_SUPPORTS 1
#else
#define POMAI_HAS_BUILTIN_CPU_SUPPORTS 0
#endif

// --------------------- Kernel prototypes and types ------------------------
using L2Func = float (*)(const float *a, const float *b, size_t dim);

// New kernel types
using DotFunc = float (*)(const float *a, const float *b, size_t dim);
using FmaFunc = void (*)(float *acc, const float *val, float scale, size_t dim);

// New: packed-sign dot kernel type
// Computes sum_j sgn_j * pvec[j] where sgn_j = +1 if bit==1 else -1.
// - sign_bytes: packed LSB-first bits covering 'bits' entries (ceil(bits/8) bytes).
// - pvec: pointer to float array length == bits
// Returns double (accumulated) for precision.
using PackedSignedDotFunc = double (*)(const uint8_t *sign_bytes, const float *pvec, uint32_t bits);

// Default forward declarations (will be set at init)
// These externs mirror the inline-exported symbols at the file bottom (keeps usage similar to prior design).
extern L2Func pomai_l2sq_kernel;
extern DotFunc pomai_dot_kernel;
extern FmaFunc pomai_fma_kernel;
extern PackedSignedDotFunc pomai_packed_signed_dot_kernel;

// --------------------- Scalar reference kernels ---------------------------
static inline float l2sq_scalar(const float *a, const float *b, size_t dim)
{
    const float *pa = a;
    const float *pb = b;
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i)
    {
        float d = pa[i] - pb[i];
        sum += d * d;
    }
    return sum;
}

static inline float dot_scalar(const float *a, const float *b, size_t dim)
{
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i)
        sum += a[i] * b[i];
    return sum;
}

static inline void fma_scalar(float *acc, const float *val, float scale, size_t dim)
{
    for (size_t i = 0; i < dim; ++i)
        acc[i] += val[i] * scale;
}

// Scalar packed-signed-dot fallback
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
        uint32_t chunk_len = 8;
        if (b == bytes - 1)
        {
            // last chunk may be partial
            uint32_t rem = bits - (bytes - 1) * 8;
            chunk_len = rem;
            v &= static_cast<uint8_t>((1u << rem) - 1u);
        }
        for (uint32_t k = 0; k < chunk_len; ++k, ++idx)
        {
            float pv = pvec[idx];
            sum_all += pv;
            if ((v >> k) & 1u) sum_bit += pv;
        }
    }
    // signed sum = 2*sum_bit - sum_all
    return 2.0 * sum_bit - sum_all;
}

// --------------------- AVX2 kernels --------------------------------------
// compile-time attribute: allows building without -mavx2 globally on GCC/Clang
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
        vsum = _mm256_fmadd_ps(d, d, vsum); // vsum += d*d
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
    for (; i < dim; ++i)
        total += a[i] * b[i];
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
        vacc = _mm256_fmadd_ps(vval, vscale, vacc); // acc = val * scale + acc
        _mm256_storeu_ps(acc + i, vacc);
    }
    for (; i < dim; ++i)
        acc[i] += val[i] * scale;
}
#else
// Fallback stubs (use scalar implementations)
static float l2sq_avx2(const float *a, const float *b, size_t dim) { return l2sq_scalar(a, b, dim); }
static float dot_avx2(const float *a, const float *b, size_t dim) { return dot_scalar(a, b, dim); }
static void fma_avx2(float *acc, const float *val, float scale, size_t dim) { fma_scalar(acc, val, scale, dim); }
#endif

// --------------------- AVX-512 placeholders -------------------------------
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f"))) static float l2sq_avx512(const float *a, const float *b, size_t dim)
{
    // Placeholder: fallback to avx2 if available
    return l2sq_avx2(a, b, dim);
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
// (Add NEON implementations similarly if targeting ARM; placeholders kept)
static float l2sq_neon(const float *a, const float *b, size_t dim) { return l2sq_scalar(a, b, dim); }
static float dot_neon(const float *a, const float *b, size_t dim) { return dot_scalar(a, b, dim); }
static void fma_neon(float *acc, const float *val, float scale, size_t dim) { fma_scalar(acc, val, scale, dim); }

// --------------------- Packed-signed-dot SIMD kernels ---------------------
// AVX2 implementation for packed signed dot (uses mask table + fmadd)
#if defined(__AVX2__) || defined(__GNUC__) || defined(__clang__)
// Helper: build mask-table (256 entries x 8 floats) on first use.
// Each entry maps a byte value (bits for 8 positions) to an __m256 of 0.0/1.0 floats.
static inline const float *get_packed_mask_table()
{
    static float table[256][8];
    static bool init = false;
    if (!init)
    {
        for (int v = 0; v < 256; ++v)
        {
            for (int k = 0; k < 8; ++k)
                table[v][k] = ((v >> k) & 1) ? 1.0f : 0.0f;
        }
        init = true;
    }
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
        // mask off bits beyond 'rem'
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
    // For now reuse AVX2 implementation (fallback). An optimized AVX-512 path can be added later.
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
static L2Func pomai_l2sq_kernel_internal = l2sq_scalar; // default
static DotFunc pomai_dot_kernel_internal = dot_scalar;
static FmaFunc pomai_fma_kernel_internal = fma_scalar;
static PackedSignedDotFunc pomai_packed_signed_dot_internal = packed_signed_dot_scalar;

inline const char *kernel_name_from_ptr(L2Func f)
{
    if (f == l2sq_scalar)
        return "scalar";
    if (f == l2sq_avx2)
        return "avx2";
    if (f == l2sq_avx512)
        return "avx512";
    if (f == l2sq_neon)
        return "neon";
    return "unknown";
}

inline void pomai_init_cpu_kernels()
{
    // Environment override: POMAI_FORCE_KERNEL=scalar|avx2|avx512|neon
    const char *force = std::getenv("POMAI_FORCE_KERNEL");
    if (force && std::strlen(force) > 0)
    {
        std::string_view fv(force);
        if (fv == "avx512")
        {
            pomai_l2sq_kernel_internal = l2sq_avx512;
            pomai_dot_kernel_internal = dot_avx512;
            pomai_fma_kernel_internal = fma_avx512;
            pomai_packed_signed_dot_internal = packed_signed_dot_avx512;
            std::clog << "[CPU] Forced kernel: avx512\n";
        }
        else if (fv == "avx2")
        {
            pomai_l2sq_kernel_internal = l2sq_avx2;
            pomai_dot_kernel_internal = dot_avx2;
            pomai_fma_kernel_internal = fma_avx2;
            pomai_packed_signed_dot_internal = packed_signed_dot_avx2;
            std::clog << "[CPU] Forced kernel: avx2\n";
        }
        else if (fv == "neon")
        {
            pomai_l2sq_kernel_internal = l2sq_neon;
            pomai_dot_kernel_internal = dot_neon;
            pomai_fma_kernel_internal = fma_neon;
            pomai_packed_signed_dot_internal = packed_signed_dot_neon;
            std::clog << "[CPU] Forced kernel: neon\n";
        }
        else
        {
            pomai_l2sq_kernel_internal = l2sq_scalar;
            pomai_dot_kernel_internal = dot_scalar;
            pomai_fma_kernel_internal = fma_scalar;
            pomai_packed_signed_dot_internal = packed_signed_dot_scalar;
            std::clog << "[CPU] Forced kernel: scalar\n";
        }
        return;
    }

#if POMAI_HAS_BUILTIN_CPU_SUPPORTS
    // prefer avx512, then avx2, then scalar
    if (__builtin_cpu_supports("avx512f"))
    {
        pomai_l2sq_kernel_internal = l2sq_avx512;
        pomai_dot_kernel_internal = dot_avx512;
        pomai_fma_kernel_internal = fma_avx512;
        pomai_packed_signed_dot_internal = packed_signed_dot_avx512;
        std::clog << "[CPU] Detected AVX-512 -> using avx512 kernels\n";
        return;
    }
    if (__builtin_cpu_supports("avx2"))
    {
        pomai_l2sq_kernel_internal = l2sq_avx2;
        pomai_dot_kernel_internal = dot_avx2;
        pomai_fma_kernel_internal = fma_avx2;
        pomai_packed_signed_dot_internal = packed_signed_dot_avx2;
        std::clog << "[CPU] Detected AVX2 -> using avx2 kernels\n";
        return;
    }
#endif

    // Fallback: scalar
    pomai_l2sq_kernel_internal = l2sq_scalar;
    pomai_dot_kernel_internal = dot_scalar;
    pomai_fma_kernel_internal = fma_scalar;
    pomai_packed_signed_dot_internal = packed_signed_dot_scalar;
    std::clog << "[CPU] Using scalar kernels\n";
}

// Accessors
inline L2Func get_pomai_l2sq_kernel() { return pomai_l2sq_kernel_internal; }
inline DotFunc get_pomai_dot_kernel() { return pomai_dot_kernel_internal; }
inline FmaFunc get_pomai_fma_kernel() { return pomai_fma_kernel_internal; }
inline PackedSignedDotFunc get_pomai_packed_signed_dot() { return pomai_packed_signed_dot_internal; }

// Public wrappers
static inline float l2sq(const float *a, const float *b, size_t dim)
{
    return pomai_l2sq_kernel_internal(a, b, dim);
}

static inline float pomai_dot(const float *a, const float *b, size_t dim)
{
    return pomai_dot_kernel_internal(a, b, dim);
}

static inline void pomai_fma(float *acc, const float *val, float scale, size_t dim)
{
    pomai_fma_kernel_internal(acc, val, scale, dim);
}

// Packed-signed-dot wrapper: returns double
static inline double pomai_packed_signed_dot(const uint8_t *sign_bytes, const float *pvec, uint32_t bits)
{
    return pomai_packed_signed_dot_internal(sign_bytes, pvec, bits);
}

// Export inline symbols (reflect current dispatch function)
inline L2Func pomai_l2sq_kernel = get_pomai_l2sq_kernel();
inline DotFunc pomai_dot_kernel = get_pomai_dot_kernel();
inline FmaFunc pomai_fma_kernel = get_pomai_fma_kernel();
inline PackedSignedDotFunc pomai_packed_signed_dot_kernel = get_pomai_packed_signed_dot();