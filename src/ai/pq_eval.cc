/*
 * src/ai/pq_eval.cc
 *
 * PQ Evaluation Kernels.
 * Includes hand-tuned AVX2 intrinsics for 4-bit PQ (k=16).
 */

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
    // --- Scalar Fallback (Reference) ---
    float pq_approx_dist_single(const float *tables, size_t m, size_t k, const uint8_t *code)
    {
        if (!tables || !code)
            return std::numeric_limits<float>::infinity();
        float acc = 0.0f;
        for (size_t i = 0; i < m; ++i)
        {
            acc += tables[i * k + code[i]];
        }
        return acc;
    }

    void pq_approx_dist_batch_scalar(const float *tables, size_t m, size_t k,
                                     const uint8_t *codes, size_t n, float *out)
    {
        for (size_t i = 0; i < n; ++i)
            out[i] = pq_approx_dist_single(tables, m, k, codes + i * m);
    }

    // --- SIMD AVX2 Implementation ---
#if defined(__AVX2__)

    static inline bool cpu_supports_avx2()
    {
        return __builtin_cpu_supports("avx2");
    }

    // Helper: Lookup 8 values from a 16-entry table using indices [0..15]
    // table_lo: Contains values for indices 0..7
    // table_hi: Contains values for indices 8..15
    // indices: 8 integer indices
    // Return: 8 float values corresponding to tables[indices[i]]
    static inline __m256 lookup_16_avx2(__m256 table_lo, __m256 table_hi, __m256i indices)
    {
        // 1. Permute from Low Table.
        // vpermps uses the lowest 3 bits of the index.
        // So index 0 (0000) -> pos 0, index 8 (1000) -> pos 0.
        __m256 v_lo = _mm256_permutevar8x32_ps(table_lo, indices);

        // 2. Permute from High Table.
        __m256 v_hi = _mm256_permutevar8x32_ps(table_hi, indices);

        // 3. Blend results.
        // If index >= 8 (bit 3 set), pick from v_hi, else v_lo.
        // We shift the indices left by 28 so bit 3 becomes the sign bit (bit 31).
        // blendv_ps uses the sign bit to select.
        __m256i mask = _mm256_slli_epi32(indices, 28);
        return _mm256_blendv_ps(v_lo, v_hi, _mm256_castsi256_ps(mask));
    }

    void pq_approx_dist_batch_packed4_avx2(const float *tables, size_t m, size_t k,
                                           const uint8_t *packed_codes, size_t n, float *out)
    {
        // This kernel is specialized for 4-bit PQ (k=16).
        if (k != 16)
        {
            // Fallback to scalar loop if k is not 16
            // In a real optimized system, we might assert or have other kernels.
            // For now, we reuse the scalar logic for the remainder or full array.
            size_t packed_bytes = (m + 1) / 2;
            std::vector<uint8_t> tmp(m);
            for (size_t i = 0; i < n; ++i)
            {
                const uint8_t *p = packed_codes + i * packed_bytes;
                for (size_t j = 0; j < m; ++j)
                {
                    uint8_t byte = p[j / 2];
                    tmp[j] = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                }
                out[i] = pq_approx_dist_single(tables, m, k, tmp.data());
            }
            return;
        }

        size_t packed_stride = (m + 1) / 2;
        size_t i = 0;

        // Process 8 vectors at a time (YMM width)
        for (; i + 8 <= n; i += 8)
        {
            __m256 sum = _mm256_setzero_ps();

            // Iterate over sub-quantizers
            for (size_t j = 0; j < m; ++j)
            {
                // 1. Load Table for sub-quantizer j (16 floats -> 2 YMMs)
                // Cache locality: 'tables' is small (m*16 floats), stays in L1.
                const float *t_ptr = tables + j * 16;
                __m256 t_lo = _mm256_loadu_ps(t_ptr);
                __m256 t_hi = _mm256_loadu_ps(t_ptr + 8);

                // 2. Extract 4-bit codes for the 8 vectors
                // Layout: Packed codes are row-major. We need the j-th nibble from 8 rows.
                size_t byte_off = j / 2;
                bool is_high_nibble = (j % 2 != 0);

                // Scalar gather (faster than VGATHER for bytes/strided small loads)
                // Compiler will likely vectorize these loads or issue efficient scalar loads.
                alignas(32) uint32_t indices_buf[8];
                const uint8_t *base_c = packed_codes + i * packed_stride + byte_off;

                // Unrolling helps CPU pipeline
                uint8_t b0 = base_c[0 * packed_stride];
                uint8_t b1 = base_c[1 * packed_stride];
                uint8_t b2 = base_c[2 * packed_stride];
                uint8_t b3 = base_c[3 * packed_stride];
                uint8_t b4 = base_c[4 * packed_stride];
                uint8_t b5 = base_c[5 * packed_stride];
                uint8_t b6 = base_c[6 * packed_stride];
                uint8_t b7 = base_c[7 * packed_stride];

                if (is_high_nibble)
                {
                    indices_buf[0] = b0 >> 4;
                    indices_buf[1] = b1 >> 4;
                    indices_buf[2] = b2 >> 4;
                    indices_buf[3] = b3 >> 4;
                    indices_buf[4] = b4 >> 4;
                    indices_buf[5] = b5 >> 4;
                    indices_buf[6] = b6 >> 4;
                    indices_buf[7] = b7 >> 4;
                }
                else
                {
                    indices_buf[0] = b0 & 0x0F;
                    indices_buf[1] = b1 & 0x0F;
                    indices_buf[2] = b2 & 0x0F;
                    indices_buf[3] = b3 & 0x0F;
                    indices_buf[4] = b4 & 0x0F;
                    indices_buf[5] = b5 & 0x0F;
                    indices_buf[6] = b6 & 0x0F;
                    indices_buf[7] = b7 & 0x0F;
                }

                __m256i v_idx = _mm256_load_si256((__m256i *)indices_buf);

                // 3. Register Lookup & Accumulate
                __m256 dists = lookup_16_avx2(t_lo, t_hi, v_idx);
                sum = _mm256_add_ps(sum, dists);
            }

            // Store 8 results
            _mm256_storeu_ps(out + i, sum);
        }

        // Handle remaining vectors (Scalar Fallback)
        for (; i < n; ++i)
        {
            float s = 0.0f;
            const uint8_t *row = packed_codes + i * packed_stride;
            for (size_t j = 0; j < m; ++j)
            {
                uint8_t val = row[j / 2];
                uint8_t nibble = (j % 2 != 0) ? (val >> 4) : (val & 0x0F);
                s += tables[j * 16 + nibble];
            }
            out[i] = s;
        }
    }
#endif

    // --- Public Dispatcher ---

    void pq_approx_dist_batch_packed4(const float *tables, size_t m, size_t k,
                                      const uint8_t *packed_codes, size_t n, float *out)
    {
#if defined(__AVX2__)
        // Runtime check + compile time guard
        // Ensure k=16 for the optimized kernel
        if (cpu_supports_avx2() && k == 16)
        {
            pq_approx_dist_batch_packed4_avx2(tables, m, k, packed_codes, n, out);
            return;
        }
#endif
        // Portable Scalar Implementation
        size_t packed_bytes = (m + 1) / 2;
        std::vector<uint8_t> tmp(m);
        for (size_t i = 0; i < n; ++i)
        {
            const uint8_t *p = packed_codes + i * packed_bytes;
            for (size_t j = 0; j < m; ++j)
            {
                uint8_t byte = p[j / 2];
                tmp[j] = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            }
            out[i] = pq_approx_dist_single(tables, m, k, tmp.data());
        }
    }

    void pq_approx_dist_batch(const float *tables, size_t m, size_t k,
                              const uint8_t *codes, size_t n, float *out)
    {
        pq_approx_dist_batch_scalar(tables, m, k, codes, n, out);
    }

} // namespace pomai::ai