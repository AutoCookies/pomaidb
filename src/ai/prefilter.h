/*
 * src/ai/prefilter.h
 *
 * Binary prefilter utilities: block streaming XOR + popcount (AVX2/POPCNT-friendly).
 *
 * Purpose
 * -------
 * Provide high-performance routines to compute Hamming distances between a
 * bitpacked query fingerprint and many stored bitpacked fingerprints.
 *
 * API (clear, minimal)
 *  - compute_hamming_all(...)       : compute Hamming distance for every vector
 *  - collect_candidates_threshold() : collect indices whose Hamming <= threshold
 *  - topk_by_hamming(...)           : return top-K smallest Hamming distances (candidate indices)
 *
 * Implementation notes
 * --------------------
 * - Fingerprints are stored as contiguous bytes per vector: fp_bytes = bits / 8.
 * - This module implements:
 *    * an AVX2-backed XOR inner loop (when available) to XOR 32-byte chunks quickly
 *      and then uses efficient 64-bit popcount to sum bits per chunk.
 *    * a scalar fallback that processes 64-bit words with builtin popcount.
 * - The functions are reentrant and thread-safe (no internal mutable state).
 *
 * Threading / portability
 * -----------------------
 * - The implementation uses compile-time detection of AVX2. If the build toolchain
 *   defines __AVX2__ and the CPU supports avx2/popcnt at runtime, the AVX2 path is
 *   used. Otherwise the scalar (portable) path is used.
 *
 * Performance tips
 * ----------------
 * - Aligning fingerprints to 32-byte boundaries improves AVX2 load efficiency.
 * - Choose bit-width (SimHash bits) as multiple of 8 for simplicity (we assume byte-packed fingerprints).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

namespace pomai::ai::prefilter
{

    // Compute Hamming distances (number of differing bits) between `query` and each
    // fingerprint in `db`.
    // - query: pointer to bitpacked fingerprint (fp_bytes bytes)
    // - db: pointer to contiguous db_count * fp_bytes bytes storage (row-major: vector i at db + i*fp_bytes)
    // - fp_bytes: bytes per fingerprint (bits / 8)
    // - db_count: number of fingerprints
    // - out: pointer to db_count uint32_t entries (caller-allocated)
    // The function fills out[i] = hamming(query, db[i]).
    void compute_hamming_all(const uint8_t *query, size_t fp_bytes,
                             const uint8_t *db, size_t db_count,
                             uint32_t *out);

    // Collect indices of database vectors whose Hamming distance <= threshold.
    // Returns vector of indices (unsorted), reserve capacity may be preallocated via out_indices.
    void collect_candidates_threshold(const uint8_t *query, size_t fp_bytes,
                                      const uint8_t *db, size_t db_count,
                                      uint32_t threshold, std::vector<size_t> &out_indices);

    // Return top-K smallest Hamming distances as vector of pairs (index, dist).
    // If db_count < K, returns all elements sorted ascending by distance.
    std::vector<std::pair<size_t, uint32_t>> topk_by_hamming(const uint8_t *query, size_t fp_bytes,
                                                             const uint8_t *db, size_t db_count,
                                                             size_t K);

} // namespace pomai::ai::prefilter