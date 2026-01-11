/*
 * src/ai/pq_eval.cc
 *
 * Implementation of PQ approximate distance evaluators.
 *
 * This module provides clear, correct implementations with reasonable
 * micro-optimizations (loop unrolling) so the compiler can auto-vectorize.
 * In addition, when AVX2 is available we provide a SIMD-gather batch path
 * that evaluates multiple candidates in parallel using _mm256_i32gather_ps.
 *
 * Notes on SIMD approach:
 *  - We process candidates in small batches (BATCH = 8) to amortize the
 *    per-subquantizer gather costs. For each subquantizer `s` we gather
 *    table[s*k + idx_i] for i in [0..BATCH) and accumulate the results
 *    into an AVX register. This converts the inner loop over subquantizers
 *    into a sequence of gathers+adds for the batch.
 *
 *  - Gather latency can be significant but for moderate m (e.g. 48..64)
 *    and large candidate counts this often outperforms a purely scalar
 *    per-candidate inner loop because it exploits instruction-level
 *    parallelism and reduces loop overhead.
 *
 *  - This path is guarded by compile-time (__AVX2__) and a runtime
 *    `__builtin_cpu_supports("avx2")` check to ensure safe usage.
 *
 *  - Scalar fallback (pq_approx_dist_batch) remains the default on CPUs
 *    without AVX2 or for small N.
 */

#include "src/ai/pq_eval.h"
#include "src/ai/pq.h" // for packed4 helper
#include <cstring>     // memcpy
#include <limits>
#include <cassert>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pomai::ai
{

    float pq_approx_dist_single(const float *tables, size_t m, size_t k, const uint8_t *code)
    {
        if (!tables || !code || m == 0 || k == 0)
            return std::numeric_limits<float>::infinity();

        // Basic scalar loop. Unroll small constant to help auto-vectorizers.
        double acc = 0.0;
        size_t sub = 0;

        // Unroll 4 at a time
        for (; sub + 4 <= m; sub += 4)
        {
            uint32_t c0 = static_cast<uint32_t>(code[sub + 0]);
            uint32_t c1 = static_cast<uint32_t>(code[sub + 1]);
            uint32_t c2 = static_cast<uint32_t>(code[sub + 2]);
            uint32_t c3 = static_cast<uint32_t>(code[sub + 3]);

            acc += static_cast<double>(tables[(sub + 0) * k + c0]);
            acc += static_cast<double>(tables[(sub + 1) * k + c1]);
            acc += static_cast<double>(tables[(sub + 2) * k + c2]);
            acc += static_cast<double>(tables[(sub + 3) * k + c3]);
        }

        // Remainder
        for (; sub < m; ++sub)
        {
            uint32_t ci = static_cast<uint32_t>(code[sub]);
            acc += static_cast<double>(tables[sub * k + ci]);
        }

        return static_cast<float>(acc);
    }

    void pq_approx_dist_batch_scalar(const float *tables, size_t m, size_t k,
                                     const uint8_t *codes, size_t n, float *out)
    {
        if (!tables || !codes || !out || m == 0 || k == 0)
            return;

        const size_t stride = m;
        for (size_t i = 0; i < n; ++i)
        {
            const uint8_t *code = codes + i * stride;
            out[i] = pq_approx_dist_single(tables, m, k, code);
        }
    }

#if defined(__AVX2__)
    // Runtime check for AVX2 availability
    static inline bool cpu_supports_avx2()
    {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
        return __builtin_cpu_supports("avx2");
#else
        return false;
#endif
#else
        return false;
#endif
    }

    // SIMD gather-based batch evaluator
    // Process candidates in batches of BATCH (8). For each subquantizer `s` we
    // gather table_base[ idx[i] ] for i in [0..BATCH).
    static void pq_approx_dist_batch_avx2_gather(const float *tables, size_t m, size_t k,
                                                 const uint8_t *codes, size_t n, float *out)
    {
        const size_t BATCH = 8; // process 8 candidates in parallel
        size_t i = 0;
        // temporary index array for gather (int32)
        int idx_buf[BATCH];

        // Process full batches
        for (; i + BATCH <= n; i += BATCH)
        {
            // initialize accumulators to zero
            __m256 acc = _mm256_setzero_ps();

            // For each subquantizer
            for (size_t s = 0; s < m; ++s)
            {
                const float *table_base = tables + s * k;
                // gather indices for this sub across the BATCH candidates
                for (size_t b = 0; b < BATCH; ++b)
                {
                    idx_buf[b] = static_cast<int>(codes[(i + b) * m + s]); // code at candidate (i+b), sub s
                }
                // load indices into __m256i (as 32-bit integers)
                __m256i idx_vec = _mm256_setr_epi32(idx_buf[0], idx_buf[1], idx_buf[2], idx_buf[3],
                                                    idx_buf[4], idx_buf[5], idx_buf[6], idx_buf[7]);
                // gather 8 floats from table_base using idx_vec, scale = 4 (sizeof(float))
                __m256 gathered = _mm256_i32gather_ps(table_base, idx_vec, 4);
                // accumulate
                acc = _mm256_add_ps(acc, gathered);
            }

            // store accumulator to out[i..i+BATCH)
            alignas(32) float tmp[BATCH];
            _mm256_storeu_ps(tmp, acc);
            for (size_t b = 0; b < BATCH; ++b)
                out[i + b] = tmp[b];
        }

        // Remainder: scalar
        for (; i < n; ++i)
        {
            out[i] = pq_approx_dist_single(tables, m, k, codes + i * m);
        }
    }
#endif // __AVX2__

    void pq_approx_dist_batch(const float *tables, size_t m, size_t k,
                              const uint8_t *codes, size_t n, float *out)
    {
        if (!tables || !codes || !out || m == 0 || k == 0)
            return;

#if defined(__AVX2__)
        // Heuristic: use SIMD-gather path when there are enough candidates to amortize
        // gather overhead and runtime supports AVX2. Batch path processes 8 candidates at a time.
        if (cpu_supports_avx2() && n >= 16) // threshold: require at least a few batches
        {
            pq_approx_dist_batch_avx2_gather(tables, m, k, codes, n, out);
            return;
        }
#endif

        // fallback scalar
        pq_approx_dist_batch_scalar(tables, m, k, codes, n, out);
    }

    void pq_approx_dist_batch_packed4(const float *tables, size_t m, size_t k,
                                      const uint8_t *packed_codes, size_t n, float *out)
    {
        if (!tables || !packed_codes || !out || m == 0 || k == 0)
            return;

        const size_t packed_bytes = ProductQuantizer::packed4BytesPerVec(m);
        // Temporary per-candidate unpack buffer (stack/automatic). m is typically moderate (e.g. 48..64).
        // Use static allocation on stack when m is small; fallback to vector when large.
        if (m <= 1024)
        {
            uint8_t tmp[1024]; // support up to m=1024 safely
            for (size_t i = 0; i < n; ++i)
            {
                const uint8_t *p = packed_codes + i * packed_bytes;
                ProductQuantizer::unpack4To8(p, tmp, m);
                out[i] = pq_approx_dist_single(tables, m, k, tmp);
            }
        }
        else
        {
            // improbable path for very large m; allocate per-call buffer
            std::vector<uint8_t> tmp(m);
            for (size_t i = 0; i < n; ++i)
            {
                const uint8_t *p = packed_codes + i * packed_bytes;
                ProductQuantizer::unpack4To8(p, tmp.data(), m);
                out[i] = pq_approx_dist_single(tables, m, k, tmp.data());
            }
        }
    }

    // New: raw 8-bit PQ codes batch evaluator.
    // This processes codes stored as contiguous m bytes per vector (no packing).
    // It clamps codes that are out-of-range to (k-1) to avoid UB if data is malformed.
    void pq_approx_dist_batch_raw8(const float *tables, size_t m, size_t k,
                                   const uint8_t *raw8_codes, size_t n, float *out)
    {
        if (!tables || !raw8_codes || !out || m == 0 || k == 0)
            return;

        for (size_t vi = 0; vi < n; ++vi)
        {
            const uint8_t *codes = raw8_codes + vi * m;
            double acc = 0.0;
            for (size_t sub = 0; sub < m; ++sub)
            {
                uint32_t ci = static_cast<uint32_t>(codes[sub]);
                // safety clamp: if a code is >= k (malformed), clamp to last centroid
                if (ci >= static_cast<uint32_t>(k))
                    ci = static_cast<uint32_t>(k - 1);
                acc += static_cast<double>(tables[sub * k + ci]);
            }
            out[vi] = static_cast<float>(acc);
        }
    }

} // namespace pomai::ai