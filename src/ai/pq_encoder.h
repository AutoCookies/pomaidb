/*
 * src/ai/pq_encoder.h
 *
 * Batch PQ encoder utilities.
 *
 * This small helper provides a convenient wrapper around ProductQuantizer to:
 *  - Encode a batch of float vectors into contiguous m-byte PQ codes (8-bit per subquant).
 *  - Optionally pack the 8-bit codes into 4-bit packed representation for on-disk storage.
 *
 * API goals (10/10):
 *  - Clear, minimal, well-documented functions suitable for integration into
 *    ingestion pipelines (SoA writer / vector store).
 *  - No hidden allocations in hot path: caller provides output buffers.
 *
 * Example:
 *   ProductQuantizer pq(dim, m, k);
 *   // train/load codebooks...
 *   PQBatchEncoder enc(pq);
 *   // samples: N * dim floats
 *   // codes: buffer size >= N * m bytes
 *   enc.encode_batch(samples, N, codes);
 *
 *   // pack to 4-bit on-disk buffer:
 *   // packed_bytes_per_vec = ProductQuantizer::packed4BytesPerVec(m)
 *   // packed: buffer size >= N * packed_bytes_per_vec
 *   enc.encode_and_pack4(samples, N, codes, packed);
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "src/ai/pq.h"

namespace pomai::ai
{

    class PQBatchEncoder
    {
    public:
        // Construct from a trained ProductQuantizer instance (reference, not owned).
        explicit PQBatchEncoder(const ProductQuantizer &pq) noexcept : pq_(pq) {}

        // Encode N vectors (contiguous) into `out_codes`.
        // - samples: pointer to N * dim floats (row-major)
        // - out_codes: pointer to N * m bytes (caller-allocated)
        // Precondition: out_codes buffer must be at least N * pq_.m() bytes.
        void encode_batch(const float *samples, size_t N, uint8_t *out_codes) const;

        // Convenience: encode then pack to 4-bit representation.
        // - out_packed: buffer of size >= N * ProductQuantizer::packed4BytesPerVec(pq_.m()) bytes.
        // This first writes N*m bytes to scratch_codes (caller may pass null to skip reuse)
        // then packs into out_packed; if crash occurs the partial packs are harmless.
        void encode_and_pack4(const float *samples, size_t N, uint8_t *out_packed, uint8_t *scratch_codes = nullptr) const;

    private:
        const ProductQuantizer &pq_;
    };

} // namespace pomai::ai