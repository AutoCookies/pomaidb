#include "core/simd/simd_dispatch.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace pomai::core::simd {

namespace {
constexpr double kEarthRadiusMeters = 6371000.0;

// Internal AVX2 implementation for L2 distance (used by RMSD)
__attribute__((target("avx2,fma")))
float MeshL2Avx2(const float* a, const float* b, std::size_t n) {
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        // sum = sum + diff * diff
#ifdef __FMA__
        sum = _mm256_fmadd_ps(diff, diff, sum);
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
#endif
    }
    
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    
    for (; i < n; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

} // namespace

double HaversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const double dlat = (lat2 - lat1) * M_PI / 180.0;
    const double dlon = (lon2 - lon1) * M_PI / 180.0;
    const double a1 = lat1 * M_PI / 180.0;
    const double a2 = lat2 * M_PI / 180.0;
    const double h = std::sin(dlat * 0.5) * std::sin(dlat * 0.5) +
                     std::cos(a1) * std::cos(a2) *
                     std::sin(dlon * 0.5) * std::sin(dlon * 0.5);
    const double c = 2.0 * std::atan2(std::sqrt(h), std::sqrt(1.0 - h));
    return kEarthRadiusMeters * c;
}

double MeshRmsdF32(const float* a_xyz, const float* b_xyz, std::size_t points) {
    if (!a_xyz || !b_xyz || points == 0) return 0.0;
    const std::size_t n = points * 3;
    
    double total_l2 = 0.0;
#if defined(__AVX2__)
    total_l2 = static_cast<double>(MeshL2Avx2(a_xyz, b_xyz, n));
#else
    // Fallback unrolled serial
    std::size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float d0 = a_xyz[i] - b_xyz[i];
        float d1 = a_xyz[i+1] - b_xyz[i+1];
        float d2 = a_xyz[i+2] - b_xyz[i+2];
        float d3 = a_xyz[i+3] - b_xyz[i+3];
        total_l2 += static_cast<double>(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
    }
    for (; i < n; ++i) {
        float d = a_xyz[i] - b_xyz[i];
        total_l2 += static_cast<double>(d * d);
    }
#endif
    return std::sqrt(total_l2 / static_cast<double>(n));
}

double SparseDotU32F32(const std::uint32_t* a_idx, const float* a_w, std::size_t a_n,
                       const std::uint32_t* b_idx, const float* b_w, std::size_t b_n) {
    if (!a_idx || !b_idx || !a_w || !b_w || a_n == 0 || b_n == 0) return 0.0;
    std::size_t i = 0, j = 0;
    double sum = 0.0;
    while (i < a_n && j < b_n) {
        if (a_idx[i] == b_idx[j]) {
            sum += static_cast<double>(a_w[i]) * static_cast<double>(b_w[j]);
            ++i;
             j++;
        } else if (a_idx[i] < b_idx[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return sum;
}

std::uint32_t SparseIntersectU32(const std::uint32_t* a_idx, std::size_t a_n,
                                 const std::uint32_t* b_idx, std::size_t b_n) {
    if (!a_idx || !b_idx || a_n == 0 || b_n == 0) return 0;
    std::size_t i = 0, j = 0;
    std::uint32_t inter = 0;
    while (i < a_n && j < b_n) {
        if (a_idx[i] == b_idx[j]) {
            ++inter;
            ++i;
            ++j;
        } else if (a_idx[i] < b_idx[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return inter;
}

__attribute__((target("avx2")))
double BitsetHamming(const std::uint8_t* a, const std::uint8_t* b, std::size_t n_bytes) {
    if (!a || !b || n_bytes == 0) return 0.0;
    std::uint64_t diff = 0;
    
    std::size_t i = 0;
#if defined(__AVX2__)
    // AVX2 XORing + scalar popcount for precision/simplicity in audit
    // (A full pshufb popcount is faster but much more complex)
    for (; i + 31 < n_bytes; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i res = _mm256_xor_si256(va, vb);
        
        // Extract to 64-bit and popcount
        uint64_t v[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(v), res);
        diff += __builtin_popcountll(v[0]) + __builtin_popcountll(v[1]) + 
                __builtin_popcountll(v[2]) + __builtin_popcountll(v[3]);
    }
#endif
    
    const std::uint64_t* a64 = reinterpret_cast<const std::uint64_t*>(a + i);
    const std::uint64_t* b64 = reinterpret_cast<const std::uint64_t*>(b + i);
    std::size_t remaining_bytes = n_bytes - i;
    std::size_t n64 = remaining_bytes / 8;
    
    for (std::size_t k = 0; k < n64; ++k) {
        diff += __builtin_popcountll(a64[k] ^ b64[k]);
    }
    
    for (std::size_t k = n64 * 8 + i; k < n_bytes; ++k) {
        diff += __builtin_popcount(a[k] ^ b[k]);
    }
    
    return static_cast<double>(diff);
}

__attribute__((target("avx2")))
double BitsetJaccard(const std::uint8_t* a, const std::uint8_t* b, std::size_t n_bytes) {
    if (!a || !b || n_bytes == 0) return 1.0;
    std::uint64_t inter = 0;
    std::uint64_t uni = 0;
    
    std::size_t i = 0;
#if defined(__AVX2__)
    for (; i + 31 < n_bytes; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        
        __m256i v_inter = _mm256_and_si256(va, vb);
        __m256i v_uni = _mm256_or_si256(va, vb);
        
        uint64_t vi[4], vu[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(vi), v_inter);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(vu), v_uni);
        
        inter += __builtin_popcountll(vi[0]) + __builtin_popcountll(vi[1]) + 
                 __builtin_popcountll(vi[2]) + __builtin_popcountll(vi[3]);
        uni += __builtin_popcountll(vu[0]) + __builtin_popcountll(vu[1]) + 
               __builtin_popcountll(vu[2]) + __builtin_popcountll(vu[3]);
    }
#endif

    for (; i < n_bytes; ++i) {
        inter += __builtin_popcount(a[i] & b[i]);
        uni += __builtin_popcount(a[i] | b[i]);
    }
    if (uni == 0) return 1.0;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

} // namespace pomai::core::simd

