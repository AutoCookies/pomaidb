#pragma once
/*
 * src/ai/eternalecho_quantizer.h
 *
 * Pomai EternalEcho Quantizer (EESQ) - header
 *
 * - Public API remains the same for callers in pomai::ai namespace.
 * - The canonical config struct now lives in pomai::config::EternalEchoConfig
 *   and we bring a local alias (pomai::ai::EternalEchoConfig) for backwards-compatibility.
 */

#include <cstdint>
#include <vector>
#include <memory>
#include <random>
#include <cstddef>

// include central config so we use single canonical config definition
#include "src/core/config.h"

namespace pomai::ai
{

    // Keep API stability: alias the central config into pomai::ai namespace.
    // Other code that refers to pomai::ai::EternalEchoConfig continues to compile.
    using EternalEchoConfig = ::pomai::config::EternalEchoConfig;

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
        // Exact (decode-based) variant: caller provides scratch buffer (dim floats) to avoid heap allocs.
        float approx_dist(const float *query, const EchoCode &code, float *scratch_buf) const;

        // Backward-compatible exact variant: uses thread_local scratch on first call
        float approx_dist(const float *query, const EchoCode &code) const;

        // ADC-style fast approximate distance computed directly on the EchoCode
        // (no full decode). This is intended for ranking / prefilter. It allocates
        // small temporary structures for projections.
        float approx_dist_code_bytes(const std::vector<std::vector<float>> &qproj, float qnorm2, const uint8_t *data, size_t len) const;

        // Precompute projections of query onto each layer: out[k] is vector of length bits_per_layer[k],
        // i.e., p_k = R_k^T * query. Useful for optimized distance computations.
        void project_query(const float *query, std::vector<std::vector<float>> &out) const;

        // Accessors
        size_t dim() const noexcept { return dim_; }
        const EternalEchoConfig &config() const noexcept { return cfg_; }

        // Per-layer column energy sums (sum_j ||col_j||^2), computed at construction.
        const std::vector<float> &layer_col_energy() const noexcept { return layer_col_energy_; }

    private:
        size_t dim_;
        EternalEchoConfig cfg_;

        // Projection matrix stored as columns: proj_[col * dim + d]
        // Total cols = sum(bits_per_layer)
        std::vector<float> proj_; // length = total_bits * dim_

        // Prefix sums of bits to locate layer offsets
        std::vector<uint32_t> layer_offsets_; // layer_offsets_[k] = starting column index for layer k

        // Per-layer precomputed column energy sums
        std::vector<float> layer_col_energy_; // length = layers

        // RNG seed used to generate proj_
        uint64_t seed_;

        // Helpers
        void pack_signs_to_bytes(const std::vector<int8_t> &signs, std::vector<uint8_t> &out_bytes) const;
        void unpack_bytes_to_signs(const std::vector<uint8_t> &in_bytes, uint32_t bits, std::vector<int8_t> &out_signs) const;
        float compute_vector_norm(const float *v) const;
    };

} // namespace pomai::ai