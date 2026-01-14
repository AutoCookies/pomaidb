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

// Default forward declarations (will be set at init)
extern L2Func pomai_l2sq_kernel;
extern DotFunc pomai_dot_kernel;
extern FmaFunc pomai_fma_kernel;

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

// --------------------- Dispatcher & init --------------------------------
static L2Func pomai_l2sq_kernel_internal = l2sq_scalar; // default
static DotFunc pomai_dot_kernel_internal = dot_scalar;
static FmaFunc pomai_fma_kernel_internal = fma_scalar;

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
            std::clog << "[CPU] Forced kernel: avx512\n";
        }
        else if (fv == "avx2")
        {
            pomai_l2sq_kernel_internal = l2sq_avx2;
            pomai_dot_kernel_internal = dot_avx2;
            pomai_fma_kernel_internal = fma_avx2;
            std::clog << "[CPU] Forced kernel: avx2\n";
        }
        else if (fv == "neon")
        {
            pomai_l2sq_kernel_internal = l2sq_neon;
            pomai_dot_kernel_internal = dot_neon;
            pomai_fma_kernel_internal = fma_neon;
            std::clog << "[CPU] Forced kernel: neon\n";
        }
        else
        {
            pomai_l2sq_kernel_internal = l2sq_scalar;
            pomai_dot_kernel_internal = dot_scalar;
            pomai_fma_kernel_internal = fma_scalar;
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
        std::clog << "[CPU] Detected AVX-512 -> using avx512 kernels\n";
        return;
    }
    if (__builtin_cpu_supports("avx2"))
    {
        pomai_l2sq_kernel_internal = l2sq_avx2;
        pomai_dot_kernel_internal = dot_avx2;
        pomai_fma_kernel_internal = fma_avx2;
        std::clog << "[CPU] Detected AVX2 -> using avx2 kernels\n";
        return;
    }
#endif

    // Fallback: scalar
    pomai_l2sq_kernel_internal = l2sq_scalar;
    pomai_dot_kernel_internal = dot_scalar;
    pomai_fma_kernel_internal = fma_scalar;
    std::clog << "[CPU] Using scalar kernels\n";
}

// Accessors
inline L2Func get_pomai_l2sq_kernel() { return pomai_l2sq_kernel_internal; }
inline DotFunc get_pomai_dot_kernel() { return pomai_dot_kernel_internal; }
inline FmaFunc get_pomai_fma_kernel() { return pomai_fma_kernel_internal; }

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

// Export inline symbols (reflect current dispatch function)
inline L2Func pomai_l2sq_kernel = get_pomai_l2sq_kernel();
inline DotFunc pomai_dot_kernel = get_pomai_dot_kernel();
inline FmaFunc pomai_fma_kernel = get_pomai_fma_kernel();