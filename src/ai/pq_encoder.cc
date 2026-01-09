/*
 * src/ai/pq_encoder.cc
 *
 * Implementation of PQBatchEncoder.
 *
 * The implementation is straightforward: it calls ProductQuantizer::encode
 * for each vector. This prioritizes clarity and correctness. If profiling
 * shows encode is a hotspot, consider vectorizing or blocking multiple
 * encodes per iteration and using cache-friendly access patterns.
 */

#include "src/ai/pq_encoder.h"
#include <cstring> // memcpy
#include <stdexcept>

namespace pomai::ai
{

    void PQBatchEncoder::encode_batch(const float *samples, size_t N, uint8_t *out_codes) const
    {
        if (!samples || !out_codes)
            throw std::invalid_argument("PQBatchEncoder::encode_batch: null pointer");

        size_t dim = pq_.dim();
        size_t m = pq_.m();

        const float *src = samples;
        uint8_t *dst = out_codes;

        for (size_t i = 0; i < N; ++i)
        {
            pq_.encode(src, dst);
            src += dim;
            dst += m;
        }
    }

    void PQBatchEncoder::encode_and_pack4(const float *samples, size_t N, uint8_t *out_packed, uint8_t *scratch_codes) const
    {
        if (!samples || !out_packed)
            throw std::invalid_argument("PQBatchEncoder::encode_and_pack4: null pointer");

        size_t m = pq_.m();
        size_t packed_bytes = ProductQuantizer::packed4BytesPerVec(m);

        // allocate local scratch if caller didn't provide one
        std::vector<uint8_t> local_codes;
        uint8_t *codes_ptr = scratch_codes;
        if (!codes_ptr)
        {
            try
            {
                local_codes.resize(m);
                codes_ptr = local_codes.data();
            }
            catch (...)
            {
                throw;
            }
        }

        const float *src = samples;
        uint8_t *outp = out_packed;

        for (size_t i = 0; i < N; ++i)
        {
            // encode one vector into codes_ptr (m bytes)
            pq_.encode(src, codes_ptr);

            // pack 8-bit codes into 4-bit nibbles in outp
            // We use ProductQuantizer::pack4From8 for correctness.
            ProductQuantizer::pack4From8(codes_ptr, outp, m);

            src += pq_.dim();
            outp += packed_bytes;
        }
    }

} // namespace pomai::ai