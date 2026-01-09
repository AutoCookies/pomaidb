/*
 * src/ai/prefilter.cc
 *
 * Implementation of binary prefilter: block streaming XOR + popcount with
 * multiple optional vectorized paths:
 *
 *  - AVX512 path: loads 64-byte blocks, XORs, stores and performs 64-bit
 *    popcount on eight 64-bit lanes (falls back to builtin popcount).
 *    This reduces loop iterations and memory operations on AVX512-capable CPUs.
 *
 *  - AVX2+PSHUFB path: uses byte-wise lookup (PSHUFB) to compute popcount per
 *    nibble (0..15) and sums the low+high nibble values to obtain per-byte
 *    popcounts. This avoids using integer POPCNT on each 64-bit word and can
 *    be faster on some microarchitectures.
 *
 *  - AVX2 XOR + popcount path: existing implementation that XORs 32 bytes,
 *    stores, and uses __builtin_popcountll on four 64-bit lanes.
 *
 *  - Scalar fallback: portable implementation using 64-bit XOR + __builtin_popcountll.
 *
 * Runtime selection:
 *  - Check for AVX512 runtime support for a fast 64-byte block path.
 *  - Otherwise prefer AVX2+PSHUFB if both AVX2 and SSSE3 are available.
 *  - Otherwise prefer AVX2 XOR+POPCNT if available.
 *  - Otherwise use scalar fallback.
 *
 * The selection logic keeps correctness first; performance-sensitive paths are
 * guarded by both compile-time and runtime checks so the code remains portable.
 */

#include "src/ai/prefilter.h"

#include <cstring>
#include <algorithm>
#include <queue>
#include <functional>
#include <limits>
#include <vector>
#include <cstdint>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace pomai::ai::prefilter
{

    // -------------------- CPU feature helpers --------------------

    static inline bool cpu_has_avx2()
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

    static inline bool cpu_has_ssse3()
    {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
        return __builtin_cpu_supports("ssse3");
#else
        return false;
#endif
#else
        return false;
#endif
    }

    static inline bool cpu_has_avx512f()
    {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
        return __builtin_cpu_supports("avx512f");
#else
        return false;
#endif
#else
        return false;
#endif
    }

    // We'll also check for the AVX512 VPOPCNTDQ extension (if available).
    static inline bool cpu_has_avx512vpopcntdq()
    {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
        return __builtin_cpu_supports("avx512vpopcntdq");
#else
        return false;
#endif
#else
        return false;
#endif
    }

    // -------------------- Scalar fallback --------------------

    static void compute_hamming_all_scalar(const uint8_t *query, size_t fp_bytes,
                                           const uint8_t *db, size_t db_count,
                                           uint32_t *out)
    {
        const size_t words = fp_bytes / 8;
        const size_t tail = fp_bytes % 8;

        for (size_t i = 0; i < db_count; ++i)
        {
            const uint8_t *vec = db + i * fp_bytes;
            uint64_t acc = 0;
            // 64-bit words
            const uint64_t *q64 = reinterpret_cast<const uint64_t *>(query);
            const uint64_t *v64 = reinterpret_cast<const uint64_t *>(vec);
            for (size_t w = 0; w < words; ++w)
            {
                uint64_t x = q64[w] ^ v64[w];
                acc += static_cast<uint64_t>(__builtin_popcountll(x));
            }
            // tail bytes (byte-wise)
            const uint8_t *q8 = query + words * 8;
            const uint8_t *v8 = vec + words * 8;
            for (size_t t = 0; t < tail; ++t)
            {
                uint8_t xb = q8[t] ^ v8[t];
                acc += static_cast<uint64_t>(__builtin_popcount((int)xb));
            }
            out[i] = static_cast<uint32_t>(acc);
        }
    }

    // -------------------- AVX2 XOR + POPCNT path (existing) --------------------

#if defined(__AVX2__)
    static void compute_hamming_all_avx2_xorpop(const uint8_t *query, size_t fp_bytes,
                                                const uint8_t *db, size_t db_count,
                                                uint32_t *out)
    {
        const size_t block = 32; // bytes per AVX2 vector
        size_t full_blocks = fp_bytes / block;
        size_t remaining_after_blocks = fp_bytes % block;

        alignas(32) uint8_t xorbuf[32];

        for (size_t i = 0; i < db_count; ++i)
        {
            const uint8_t *vec = db + i * fp_bytes;
            uint64_t acc = 0;

            // process full 32-byte blocks
            for (size_t b = 0; b < full_blocks; ++b)
            {
                const uint8_t *qptr = query + b * block;
                const uint8_t *vptr = vec + b * block;

                __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(qptr));
                __m256i vv = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(vptr));
                __m256i xr = _mm256_xor_si256(qv, vv);
                _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(xorbuf), xr);

                const uint64_t *u64 = reinterpret_cast<const uint64_t *>(xorbuf);
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[0]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[1]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[2]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[3]));
            }

            // remaining bytes after 32-byte blocks
            size_t rem_off = full_blocks * block;
            size_t words64 = remaining_after_blocks / 8;
            for (size_t w = 0; w < words64; ++w)
            {
                uint64_t qv = 0, vv = 0;
                std::memcpy(&qv, query + rem_off + w * 8, sizeof(uint64_t));
                std::memcpy(&vv, vec + rem_off + w * 8, sizeof(uint64_t));
                uint64_t x = qv ^ vv;
                acc += static_cast<uint64_t>(__builtin_popcountll(x));
            }
            size_t tail_off = rem_off + words64 * 8;
            size_t tail_bytes = remaining_after_blocks % 8;
            for (size_t t = 0; t < tail_bytes; ++t)
            {
                uint8_t xb = query[tail_off + t] ^ vec[tail_off + t];
                acc += static_cast<uint64_t>(__builtin_popcount((int)xb));
            }

            out[i] = static_cast<uint32_t>(acc);
        }
    }
#endif // __AVX2__

// -------------------- AVX2 PSHUFB nibble-lookup popcount path --------------------
//
// This path uses a 16-byte lookup table for nibble popcounts (0..15). For each
// byte we compute popcount(byte) = table[low_nibble] + table[high_nibble].
// We perform the lookups in parallel using _mm256_shuffle_epi8 (PSHUFB) and
// then sum the resulting per-byte popcounts. This approach can be faster than
// doing 64-bit POPCNT on some microarchitectures because it replaces POPCNT
// with cheaper byte-table lookups and byte adds.
//
// Requirements: AVX2 + SSSE3 (PSHUFB), checked at runtime.
#if defined(__AVX2__)
    static void compute_hamming_all_avx2_pshufb(const uint8_t *query, size_t fp_bytes,
                                                const uint8_t *db, size_t db_count,
                                                uint32_t *out)
    {
        const size_t block = 32; // process 32 bytes per iteration (256 bits)
        size_t full_blocks = fp_bytes / block;
        size_t remaining_after_blocks = fp_bytes % block;

        // 16-byte nibble popcount table (0..15)
        const uint8_t nibble_table_arr[16] = {
            0, 1, 1, 2, 1, 2, 2, 3,
            1, 2, 2, 3, 2, 3, 3, 4};

        // create 128-bit vector for lookup and broadcast to 256-bit
        __m128i lut128 = _mm_loadu_si128(reinterpret_cast<const __m128i_u *>(nibble_table_arr));
        __m256i lut256 = _mm256_broadcastsi128_si256(lut128);

        const __m256i mask0F = _mm256_set1_epi8(static_cast<char>(0x0F));

        alignas(32) uint8_t popbuf[32];

        for (size_t i = 0; i < db_count; ++i)
        {
            const uint8_t *vec = db + i * fp_bytes;
            uint64_t acc = 0;

            for (size_t b = 0; b < full_blocks; ++b)
            {
                const uint8_t *qptr = query + b * block;
                const uint8_t *vptr = vec + b * block;

                __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(qptr));
                __m256i vv = _mm256_loadu_si256(reinterpret_cast<const __m256i_u *>(vptr));
                __m256i xr = _mm256_xor_si256(qv, vv);

                // low nibble indices
                __m256i lo_idx = _mm256_and_si256(xr, mask0F);
                // high nibble: shift right by 4 bits within each byte. Use logical shift of 16-bit lanes.
                __m256i hi_shift = _mm256_srli_epi16(xr, 4);
                __m256i hi_idx = _mm256_and_si256(hi_shift, mask0F);

                // lookup low and high nibble popcounts using PSHUFB
                __m256i pop_lo = _mm256_shuffle_epi8(lut256, lo_idx);
                __m256i pop_hi = _mm256_shuffle_epi8(lut256, hi_idx);

                __m256i pop_b = _mm256_add_epi8(pop_lo, pop_hi);

                // store per-byte popcounts
                _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(popbuf), pop_b);

                // sum 32 bytes into acc (sum by 8-byte groups to reduce loop overhead)
                for (size_t chunk = 0; chunk < 4; ++chunk)
                {
                    uint64_t s = 0;
                    const uint8_t *p = popbuf + chunk * 8;
                    s += p[0];
                    s += p[1];
                    s += p[2];
                    s += p[3];
                    s += p[4];
                    s += p[5];
                    s += p[6];
                    s += p[7];
                    acc += s;
                }
            }

            // Remaining bytes after full 32-byte blocks
            size_t rem_off = full_blocks * block;
            // process 8-byte words first (scalar)
            size_t words64 = remaining_after_blocks / 8;
            for (size_t w = 0; w < words64; ++w)
            {
                uint64_t qv = 0, vv = 0;
                std::memcpy(&qv, query + rem_off + w * 8, sizeof(uint64_t));
                std::memcpy(&vv, vec + rem_off + w * 8, sizeof(uint64_t));
                uint64_t x = qv ^ vv;
                acc += static_cast<uint64_t>(__builtin_popcountll(x));
            }
            // tail bytes
            size_t tail_off = rem_off + words64 * 8;
            size_t tail_bytes = remaining_after_blocks % 8;
            for (size_t t = 0; t < tail_bytes; ++t)
            {
                uint8_t xb = query[tail_off + t] ^ vec[tail_off + t];
                acc += static_cast<uint64_t>(__builtin_popcount((int)xb));
            }

            out[i] = static_cast<uint32_t>(acc);
        }
    }
#endif // __AVX2__

// -------------------- AVX512 wide-load + POPCNT on lanes (fallback) --------------------
//
// This path uses 64-byte loads and processes eight 64-bit lanes per iteration.
// It still relies on scalar __builtin_popcountll on each 64-bit lane, but
// reduces the number of load/XOR/store iterations when compared to 32-byte
// paths. This is helpful on CPUs where wide memory ops are beneficial.
#if defined(__AVX512F__)
    static void compute_hamming_all_avx512(const uint8_t *query, size_t fp_bytes,
                                           const uint8_t *db, size_t db_count,
                                           uint32_t *out)
    {
        const size_t block = 64; // bytes
        size_t full_blocks = fp_bytes / block;
        size_t remaining_after_blocks = fp_bytes % block;

        alignas(64) uint8_t xorbuf[64];

        for (size_t i = 0; i < db_count; ++i)
        {
            const uint8_t *vec = db + i * fp_bytes;
            uint64_t acc = 0;

            for (size_t b = 0; b < full_blocks; ++b)
            {
                const uint8_t *qptr = query + b * block;
                const uint8_t *vptr = vec + b * block;

                __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(qptr));
                __m512i vv = _mm512_loadu_si512(reinterpret_cast<const void *>(vptr));
                __m512i xr = _mm512_xor_si512(qv, vv);
                _mm512_storeu_si512(reinterpret_cast<void *>(xorbuf), xr);

                const uint64_t *u64 = reinterpret_cast<const uint64_t *>(xorbuf);
                // 8 lanes
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[0]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[1]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[2]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[3]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[4]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[5]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[6]));
                acc += static_cast<uint64_t>(__builtin_popcountll(u64[7]));
            }

            // remaining bytes after 64-byte blocks
            size_t rem_off = full_blocks * block;
            size_t words64 = remaining_after_blocks / 8;
            for (size_t w = 0; w < words64; ++w)
            {
                uint64_t qv = 0, vv = 0;
                std::memcpy(&qv, query + rem_off + w * 8, sizeof(uint64_t));
                std::memcpy(&vv, vec + rem_off + w * 8, sizeof(uint64_t));
                uint64_t x = qv ^ vv;
                acc += static_cast<uint64_t>(__builtin_popcountll(x));
            }
            size_t tail_off = rem_off + words64 * 8;
            size_t tail_bytes = remaining_after_blocks % 8;
            for (size_t t = 0; t < tail_bytes; ++t)
            {
                uint8_t xb = query[tail_off + t] ^ vec[tail_off + t];
                acc += static_cast<uint64_t>(__builtin_popcount((int)xb));
            }

            out[i] = static_cast<uint32_t>(acc);
        }
    }
#endif // __AVX512F__

    // -------------------- Top-level dispatcher --------------------

    void compute_hamming_all(const uint8_t *query, size_t fp_bytes,
                             const uint8_t *db, size_t db_count,
                             uint32_t *out)
    {
        if (!query || !db || !out || fp_bytes == 0)
            return;

        // Preference order:
        // 1) AVX512 wide-load path (if compiled and runtime supports)
#if defined(__AVX512F__)
        if (cpu_has_avx512f())
        {
            compute_hamming_all_avx512(query, fp_bytes, db, db_count, out);
            return;
        }
#endif

        // 2) AVX2 + PSHUFB path (fast nibble lookup) when SSSE3 + AVX2 present
#if defined(__AVX2__)
        if (cpu_has_avx2() && cpu_has_ssse3())
        {
            compute_hamming_all_avx2_pshufb(query, fp_bytes, db, db_count, out);
            return;
        }
        // 3) AVX2 XOR + POPCNT path
        if (cpu_has_avx2())
        {
            compute_hamming_all_avx2_xorpop(query, fp_bytes, db, db_count, out);
            return;
        }
#endif

        // 4) Fallback scalar
        compute_hamming_all_scalar(query, fp_bytes, db, db_count, out);
    }

    void collect_candidates_threshold(const uint8_t *query, size_t fp_bytes,
                                      const uint8_t *db, size_t db_count,
                                      uint32_t threshold, std::vector<size_t> &out_indices)
    {
        if (!query || !db)
            return;
        out_indices.clear();

        // Compute in chunks and append indices meeting threshold.
        const size_t batch = 1024; // process 1024 vectors per inner batch
        std::vector<uint32_t> tmp;
        tmp.resize(batch);

        size_t processed = 0;
        while (processed < db_count)
        {
            size_t take = std::min(batch, db_count - processed);
            compute_hamming_all(query, fp_bytes, db + processed * fp_bytes, take, tmp.data());
            for (size_t i = 0; i < take; ++i)
            {
                if (tmp[i] <= threshold)
                    out_indices.push_back(processed + i);
            }
            processed += take;
        }
    }

    std::vector<std::pair<size_t, uint32_t>> topk_by_hamming(const uint8_t *query, size_t fp_bytes,
                                                             const uint8_t *db, size_t db_count,
                                                             size_t K)
    {
        std::vector<std::pair<size_t, uint32_t>> empty;
        if (!query || !db || db_count == 0 || K == 0)
            return empty;

        // Use max-heap of size K to track smallest distances.
        using Item = std::pair<uint32_t, size_t>; // (dist, idx)
        struct Cmp
        {
            bool operator()(Item const &a, Item const &b) const { return a.first < b.first; }
        }; // max-heap by distance

        std::priority_queue<Item, std::vector<Item>, Cmp> heap;

        // Process in batches
        const size_t batch = 1024;
        std::vector<uint32_t> tmp;
        tmp.resize(batch);

        size_t processed = 0;
        while (processed < db_count)
        {
            size_t take = std::min(batch, db_count - processed);
            compute_hamming_all(query, fp_bytes, db + processed * fp_bytes, take, tmp.data());
            for (size_t i = 0; i < take; ++i)
            {
                uint32_t dist = tmp[i];
                size_t idx = processed + i;
                if (heap.size() < K)
                {
                    heap.emplace(dist, idx);
                }
                else if (dist < heap.top().first)
                {
                    heap.pop();
                    heap.emplace(dist, idx);
                }
            }
            processed += take;
        }

        // Extract results into vector sorted ascending by distance
        std::vector<std::pair<size_t, uint32_t>> out;
        out.reserve(heap.size());
        while (!heap.empty())
        {
            Item it = heap.top();
            heap.pop();
            out.emplace_back(it.second, it.first);
        }
        std::reverse(out.begin(), out.end()); // ascending order
        return out;
    }

} // namespace pomai::ai::prefilter