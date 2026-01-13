#pragma once
/*
 * src/ai/eternalecho_quantizer.h
 *
 * Pomai EternalEcho Quantizer (EESQ) - header
 *
 * - Novel, proprietary "EternalEcho Sign-Scale Quantization" (EESQ).
 * - Recursive residual sign-projection with dynamic per-layer scale.
 * - Variable-depth, early-termination; lightweight projection matrices.
 *
 * Public API:
 *   - EternalEchoQuantizer(cfg, dim, seed)
 *   - EchoCode encode(const float* vec) const
 *   - void decode(const EchoCode&, float* out) const
 *   - float approx_dist(const float* query, const EchoCode&) const
 *   - void project_query(const float* query, std::vector<std::vector<float>>& out) const
 *
 * Notes:
 *  - Implementation is standalone (does not rely on SimHash internals).
 *  - Projection matrix is generated deterministically from seed (Gaussian entries).
 */

#include <cstdint>
#include <vector>
#include <memory>
#include <random>
#include <cstddef>

namespace pomai::ai
{

    struct EternalEchoConfig
    {
        // Default layer bits (sum ~256)
        std::vector<uint32_t> bits_per_layer = {96, 64, 48, 32, 16};
        uint32_t max_depth = 5;       // safety (should match bits_per_layer.size())
        float stop_threshold = 1e-2f; // early stop: ||residual||2 < threshold * ||original||
        bool quantize_scales = true;  // if true scale stored as uint8 (0..255)
        float scale_quant_max = 4.0f; // max scale mapped to 255 (heuristic)
    };

    struct EchoCode
    {
        uint8_t depth = 0; // number of layers encoded
        // If quantize_scales==true then scales_q contains uint8_t values (0..255) else scales_f used.
        std::vector<uint8_t> scales_q;        // quantized scales (per layer) if used
        std::vector<float> scales_f;          // full-precision scales (if not quantized)
        std::vector<uint32_t> bits_per_layer; // bits used per layer (for decoding)
        // Packed sign bits per layer. Each entry is a byte-vector containing ceil(b/8) bytes.
        std::vector<std::vector<uint8_t>> sign_bytes;
    };

    class EternalEchoQuantizer
    {
    public:
        EternalEchoQuantizer(size_t dim, const EternalEchoConfig &cfg = EternalEchoConfig(), uint64_t seed = 123456789ULL);
        ~EternalEchoQuantizer() = default;

        // Encode input vector 'vec' (length dim). Returns EchoCode (variable-length).
        EchoCode encode(const float *vec) const;

        // Decode code into out_vec (length dim). out_vec must be allocated by caller.
        void decode(const EchoCode &code, float *out_vec) const;

        // Approximate squared L2 distance between query and reconstructed code.
        // Implementation reconstructs the candidate then computes l2sq(query, recon).
        float approx_dist(const float *query, const EchoCode &code) const;

        // Precompute projections of query onto each layer: out[k] is vector of length bits_per_layer[k],
        // i.e., p_k = R_k^T * query. Useful for optimized distance computations.
        void project_query(const float *query, std::vector<std::vector<float>> &out) const;

        // Accessors
        size_t dim() const noexcept { return dim_; }
        const EternalEchoConfig &config() const noexcept { return cfg_; }

    private:
        size_t dim_;
        EternalEchoConfig cfg_;

        // Projection matrix stored as columns: proj_[col * dim + d]
        // Total cols = sum(bits_per_layer)
        std::vector<float> proj_; // length = total_bits * dim_

        // Prefix sums of bits to locate layer offsets
        std::vector<uint32_t> layer_offsets_; // layer_offsets_[k] = starting column index for layer k

        // RNG seed used to generate proj_
        uint64_t seed_;

        // Helpers
        float l2sq(const float *a, const float *b) const;
        void pack_signs_to_bytes(const std::vector<int8_t> &signs, std::vector<uint8_t> &out_bytes) const;
        void unpack_bytes_to_signs(const std::vector<uint8_t> &in_bytes, uint32_t bits, std::vector<int8_t> &out_signs) const;
        float compute_vector_norm(const float *v) const;
    };

} // namespace pomai::ai