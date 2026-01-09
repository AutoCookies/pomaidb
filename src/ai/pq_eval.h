/*
 * src/ai/pq_eval.h
 *
 * PQ approximate distance evaluator helpers.
 *
 * Overview
 * --------
 * These helpers compute approximate L2 distances between a query and many
 * database vectors represented by PQ codes using precomputed per-query
 * distance tables.
 *
 * Workflow:
 *  1) Per-query: call Codebooks::compute_distance_tables(query, tables)
 *     where `tables` is a preallocated float buffer of length m * k.
 *     Layout: tables[sub * k + centroid] = squared-L2-distance for sub.
 *
 *  2) For each candidate with code[] (m bytes, one centroid index per sub),
 *     the approximate distance is:
 *         sum_{sub=0..m-1} tables[sub * k + code[sub]]
 *
 * This file exposes batch helpers and a packed-4-bit support path. Implemented
 * algorithms favor clarity and allow later vectorized optimizations.
 *
 * Threading: functions are reentrant and thread-safe (no internal shared state).
 *
 * Note: ProductQuantizer::packed4BytesPerVec / unpack4To8 are used to support
 * 4-bit-packed on-disk codes. Include pq.h for access.
 *
 * API:
 *   float pq_approx_dist_single(const float *tables, size_t m, size_t k, const uint8_t *code);
 *   void pq_approx_dist_batch(const float *tables, size_t m, size_t k,
 *                             const uint8_t *codes, size_t n, float *out);
 *   void pq_approx_dist_batch_packed4(const float *tables, size_t m, size_t k,
 *                                     const uint8_t *packed_codes, size_t n, float *out);
 *
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace pomai::ai
{

    // Approximate distance for a single PQ-coded vector.
    // - tables: pointer to m * k floats produced by compute_distance_tables()
    // - m: number of subquantizers
    // - k: codebook size per subquantizer
    // - code: pointer to m bytes (each in [0..k-1])
    float pq_approx_dist_single(const float *tables, size_t m, size_t k, const uint8_t *code);

    // Batch evaluate N candidate codes stored as contiguous N * m bytes.
    // - codes layout: candidate i starts at codes + i*m
    // - out: caller-allocated array of N floats
    void pq_approx_dist_batch(const float *tables, size_t m, size_t k,
                              const uint8_t *codes, size_t n, float *out);

    // Batch evaluate N candidate codes stored packed as 4-bit nibbles.
    // - packed_codes layout: N * packed4BytesPerVec(m) bytes, packed by ProductQuantizer::pack4From8
    // - This function unpacks each candidate into a small stack/local buffer then sums using tables.
    void pq_approx_dist_batch_packed4(const float *tables, size_t m, size_t k,
                                      const uint8_t *packed_codes, size_t n, float *out);

} // namespace pomai::ai