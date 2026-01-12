/* src/ai/pq_eval.cc */
#include "src/ai/pq_eval.h"
#include "src/ai/pq.h"
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pomai::ai
{
    // Scalar fallback (đơn luồng, an toàn)
    float pq_approx_dist_single(const float *tables, size_t m, size_t k, const uint8_t *code)
    {
        if (!tables || !code) return std::numeric_limits<float>::infinity();
        float acc = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            acc += tables[i * k + code[i]];
        }
        return acc;
    }

    void pq_approx_dist_batch_scalar(const float *tables, size_t m, size_t k,
                                     const uint8_t *codes, size_t n, float *out) {
        for(size_t i=0; i<n; ++i) 
            out[i] = pq_approx_dist_single(tables, m, k, codes + i*m);
    }

    // --- SIMD AVX2 Implementation ---
#if defined(__AVX2__)
    static inline bool cpu_supports_avx2() {
        return __builtin_cpu_supports("avx2");
    }

    void pq_approx_dist_batch_packed4_avx2(
        const float *tables, size_t m, size_t k,
        const uint8_t *packed_codes, size_t n, float *out)
    {
        size_t packed_bytes = (m + 1) / 2;
        size_t i = 0;

        // [CRITICAL FIX] Dùng std::vector thay vì mảng tĩnh unpacked[8][256]
        // Để tránh Stack Overflow khi Dimension > 256 (VD: 512, 1024)
        // Buffer chứa 8 vector đã giải nén. Kích thước: 8 * m.
        std::vector<uint8_t> unpack_buffer(8 * m);

        // Xử lý từng nhóm 8 vector
        for (; i + 8 <= n; i += 8)
        {
            __m256 acc = _mm256_setzero_ps(); 

            // Unpack 8 vector vào buffer chung
            for (int v = 0; v < 8; ++v) {
                const uint8_t* src = packed_codes + (i + v) * packed_bytes;
                uint8_t* dst = unpack_buffer.data() + v * m; // Trỏ đến vùng của vector v
                
                for (size_t j = 0; j < m; ++j) {
                    uint8_t byte = src[j / 2];
                    dst[j] = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                }
            }

            // Duyệt qua từng Sub-quantizer (Dimension m)
            for (size_t s = 0; s < m; ++s)
            {
                const float* table_base = tables + s * k;

                // Load 8 chỉ số index từ buffer đã unpack
                // Truy cập: unpack_buffer[v * m + s]
                __m256i idx = _mm256_setr_epi32(
                    unpack_buffer[0 * m + s], unpack_buffer[1 * m + s], 
                    unpack_buffer[2 * m + s], unpack_buffer[3 * m + s],
                    unpack_buffer[4 * m + s], unpack_buffer[5 * m + s], 
                    unpack_buffer[6 * m + s], unpack_buffer[7 * m + s]
                );

                __m256 vals = _mm256_i32gather_ps(table_base, idx, 4);
                acc = _mm256_add_ps(acc, vals);
            }
            _mm256_storeu_ps(out + i, acc);
        }

        // Xử lý các vector lẻ còn lại bằng Scalar
        for (; i < n; ++i) {
            std::vector<uint8_t> tmp(m);
            const uint8_t* src = packed_codes + i * packed_bytes;
            for (size_t j = 0; j < m; ++j) {
                uint8_t byte = src[j / 2];
                tmp[j] = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            }
            out[i] = pq_approx_dist_single(tables, m, k, tmp.data());
        }
    }
#endif

    void pq_approx_dist_batch_packed4(const float *tables, size_t m, size_t k,
                                      const uint8_t *packed_codes, size_t n, float *out)
    {
#if defined(__AVX2__)
        if (cpu_supports_avx2() && n >= 8) {
            pq_approx_dist_batch_packed4_avx2(tables, m, k, packed_codes, n, out);
            return;
        }
#endif
        // Fallback
        size_t packed_bytes = (m + 1) / 2;
        std::vector<uint8_t> tmp(m);
        for (size_t i = 0; i < n; ++i)
        {
            const uint8_t *p = packed_codes + i * packed_bytes;
            for (size_t j = 0; j < m; ++j) {
                uint8_t byte = p[j / 2];
                tmp[j] = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            }
            out[i] = pq_approx_dist_single(tables, m, k, tmp.data());
        }
    }

    void pq_approx_dist_batch(const float *tables, size_t m, size_t k,
                              const uint8_t *codes, size_t n, float *out) {
        pq_approx_dist_batch_scalar(tables, m, k, codes, n, out);
    }
    
    void pq_approx_dist_batch_raw8(const float *tables, size_t m, size_t k,
                                   const uint8_t *raw8_codes, size_t n, float *out) {
        for(size_t i=0; i<n; ++i) {
             float acc = 0.0f;
             const uint8_t* c = raw8_codes + i*m;
             for(size_t j=0; j<m; ++j) acc += tables[j*k + c[j]];
             out[i] = acc;
        }
    }
}