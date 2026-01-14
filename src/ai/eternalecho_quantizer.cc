/*
 * src/ai/eternalecho_quantizer.cc
 *
 * Pomai EternalEcho Quantizer (EESQ) - implementation
 *
 * See header for algorithm description and API.
 *
 * Implementation notes / design choices:
 *  - Projection matrix is generated with Gaussian RV N(0,1) using provided seed.
 *    Storage format: contiguous columns, column-major-like: proj_[col * dim + d]
 *  - For simplicity and clarity this prototype stores quantized scales as uint8 (0..255)
 *    when enabled. The mapping is linear via scale_quant_max.
 *  - encode() reconstructs the layer echo by summing b_k columns scaled by sign and s_k,
 *    subtracts from residual and continues. Early stop uses l2 norm threshold.
 *  - decode() replays echoes to build reconstructed vector.
 *  - approx_dist() reconstructs candidate and computes squared L2 against query.
 *
 * This implementation favors clarity and correctness for a first prototype.
 */

#include "src/ai/eternalecho_quantizer.h"
#include "src/core/cpu_kernels.h"

#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace pomai::ai
{

    // -------------------- Helpers --------------------

    EternalEchoQuantizer::EternalEchoQuantizer(size_t dim, const EternalEchoConfig &cfg, uint64_t seed)
        : dim_(dim), cfg_(cfg), seed_(seed)
    {
        if (dim_ == 0)
            throw std::invalid_argument("EternalEchoQuantizer: dim must be > 0");

        // Respect provided max_depth vs bits_per_layer length
        if (cfg_.bits_per_layer.empty())
            throw std::invalid_argument("EternalEchoQuantizer: bits_per_layer cannot be empty");

        size_t layers = std::min<size_t>(cfg_.bits_per_layer.size(), std::max<size_t>(1, cfg_.max_depth));
        cfg_.bits_per_layer.resize(layers);

        // build layer offsets
        layer_offsets_.resize(layers);
        uint32_t offset = 0;
        for (size_t k = 0; k < layers; ++k)
        {
            layer_offsets_[k] = offset;
            offset += cfg_.bits_per_layer[k];
        }
        uint32_t total_bits = offset;
        if (total_bits == 0)
            throw std::invalid_argument("EternalEchoQuantizer: total bits must be > 0");

        // Initialize proj_ matrix: total_bits columns, each of length dim_
        proj_.resize(static_cast<size_t>(total_bits) * dim_);

        // Deterministic Gaussian initialization
        std::mt19937_64 rng(seed_);
        std::normal_distribution<float> nd(0.0f, 1.0f);

        for (uint32_t col = 0; col < total_bits; ++col)
        {
            size_t base = static_cast<size_t>(col) * dim_;
            for (size_t d = 0; d < dim_; ++d)
            {
                proj_[base + d] = nd(rng);
            }
        }
    }

    float EternalEchoQuantizer::compute_vector_norm(const float *v) const
    {
        double sum = 0.0;
        for (size_t i = 0; i < dim_; ++i)
            sum += static_cast<double>(v[i]) * static_cast<double>(v[i]);
        return static_cast<float>(std::sqrt(sum));
    }

    float EternalEchoQuantizer::l2sq(const float *a, const float *b) const
    {
        double acc = 0.0;
        for (size_t i = 0; i < dim_; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            acc += diff * diff;
        }
        return static_cast<float>(acc);
    }

    void EternalEchoQuantizer::pack_signs_to_bytes(const std::vector<int8_t> &signs, std::vector<uint8_t> &out_bytes) const
    {
        uint32_t bits = static_cast<uint32_t>(signs.size());
        size_t bytes = (bits + 7) / 8;
        out_bytes.assign(bytes, 0);
        for (uint32_t i = 0; i < bits; ++i)
        {
            uint32_t byte_idx = i >> 3;
            uint32_t bit_idx = i & 7;
            // store +1 -> 1, -1 -> 0
            if (signs[i] >= 0)
                out_bytes[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
        }
    }

    void EternalEchoQuantizer::unpack_bytes_to_signs(const std::vector<uint8_t> &in_bytes, uint32_t bits, std::vector<int8_t> &out_signs) const
    {
        out_signs.assign(bits, 1);
        for (uint32_t i = 0; i < bits; ++i)
        {
            uint32_t byte_idx = i >> 3;
            uint32_t bit_idx = i & 7;
            uint8_t b = (byte_idx < in_bytes.size()) ? in_bytes[byte_idx] : 0;
            bool bit = (b >> bit_idx) & 1u;
            out_signs[i] = bit ? int8_t(+1) : int8_t(-1);
        }
    }

    // -------------------- Encoding / Decoding --------------------

    EchoCode EternalEchoQuantizer::encode(const float *vec) const
    {
        if (!vec)
            throw std::invalid_argument("EternalEchoQuantizer::encode: null vec");

        EchoCode code;
        size_t layers = layer_offsets_.size();
        code.bits_per_layer.resize(layers);

        // Prepare residual copy
        std::vector<float> residual(dim_);
        std::memcpy(residual.data(), vec, dim_ * sizeof(float));

        float orig_norm = compute_vector_norm(vec);
        if (orig_norm == 0.0f)
            orig_norm = 1.0f;

        // Temporary reuse buffers
        std::vector<float> proj_vals; // size b_k
        std::vector<int8_t> signs;    // size b_k
        std::vector<float> recon_layer(dim_, 0.0f);

        // cache kernels
        DotFunc dotk = get_pomai_dot_kernel();
        FmaFunc fmak = get_pomai_fma_kernel();

        for (size_t k = 0; k < layers; ++k)
        {
            uint32_t b = cfg_.bits_per_layer[k];
            code.bits_per_layer[k] = b;

            proj_vals.assign(b, 0.0f);
            signs.assign(b, +1);

            uint32_t col0 = layer_offsets_[k];

            // compute p_j = column_j^T * residual for j in 0..b-1
            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                // use optimized dot kernel
                float accf = dotk(col_ptr, residual.data(), dim_);
                proj_vals[j] = accf;
                signs[j] = (proj_vals[j] >= 0.0f) ? int8_t(+1) : int8_t(-1);
            }

            // scale = mean(abs(p_j))
            double sum_abs = 0.0;
            for (uint32_t j = 0; j < b; ++j)
                sum_abs += std::fabs(static_cast<double>(proj_vals[j]));
            float scale = static_cast<float>(sum_abs / static_cast<double>(b));

            // append scale (quantized or full)
            if (cfg_.quantize_scales)
            {
                // clamp scale to [0, scale_quant_max]
                float s = std::min(scale, cfg_.scale_quant_max);
                uint32_t q = static_cast<uint32_t>(std::round((s / cfg_.scale_quant_max) * 255.0f));
                if (q > 255)
                    q = 255;
                code.scales_q.push_back(static_cast<uint8_t>(q));
            }
            else
            {
                code.scales_f.push_back(scale);
            }

            // pack signs into bytes
            std::vector<uint8_t> packed;
            pack_signs_to_bytes(signs, packed);
            code.sign_bytes.push_back(std::move(packed));

            // reconstruct layer echo using fma: recon_layer += (scale * sign_j) * col_j
            std::fill(recon_layer.begin(), recon_layer.end(), 0.0f);
            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float sgn = static_cast<float>(signs[j]);
                float coeff = scale * sgn;
                // use optimized fma kernel to accumulate
                fmak(recon_layer.data(), col_ptr, coeff, dim_);
            }

            // subtract from residual
            for (size_t d = 0; d < dim_; ++d)
                residual[d] -= recon_layer[d];

            // increase depth count
            code.depth = static_cast<uint8_t>(k + 1);

            // check early stop (fixed: pass pointer)
            float res_norm = compute_vector_norm(residual.data());
            if (res_norm <= cfg_.stop_threshold * orig_norm)
            {
                // truncated early
                break;
            }
        }

        // If scales were quantized, ensure scales_f empty; else we may want to ensure scales_q empty.
        if (!cfg_.quantize_scales && !code.scales_q.empty())
            code.scales_q.clear();

        return code;
    }

    void EternalEchoQuantizer::decode(const EchoCode &code, float *out_vec) const
    {
        if (!out_vec)
            throw std::invalid_argument("EternalEchoQuantizer::decode: null out_vec");

        std::fill(out_vec, out_vec + dim_, 0.0f);

        size_t depth = static_cast<size_t>(code.depth);
        if (depth == 0)
            return;

        // Ensure bits_per_layer length consistent
        size_t layers = layer_offsets_.size();
        size_t use_layers = std::min<size_t>(depth, layers);

        FmaFunc fmak = get_pomai_fma_kernel();

        for (size_t k = 0; k < use_layers; ++k)
        {
            uint32_t b = (k < code.bits_per_layer.size()) ? code.bits_per_layer[k] : cfg_.bits_per_layer[k];
            uint32_t col0 = layer_offsets_[k];

            // unpack signs
            std::vector<int8_t> signs;
            unpack_bytes_to_signs(code.sign_bytes[k], b, signs);

            // scale
            float scale = 0.0f;
            if (cfg_.quantize_scales)
            {
                if (k < code.scales_q.size())
                {
                    uint8_t q = code.scales_q[k];
                    scale = (static_cast<float>(q) / 255.0f) * cfg_.scale_quant_max;
                }
                else
                {
                    scale = 0.0f;
                }
            }
            else
            {
                if (k < code.scales_f.size())
                    scale = code.scales_f[k];
                else
                    scale = 0.0f;
            }

            // accumulate echo: out_vec += scale * sum_j sign_j * col_j
            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float sgn = static_cast<float>(signs[j]);
                float coeff = scale * sgn;
                // use optimized fma kernel to accumulate
                fmak(out_vec, col_ptr, coeff, dim_);
            }
        }
    }

    void EternalEchoQuantizer::project_query(const float *query, std::vector<std::vector<float>> &out) const
    {
        if (!query)
            throw std::invalid_argument("EternalEchoQuantizer::project_query: null query");

        size_t layers = layer_offsets_.size();
        out.clear();
        out.resize(layers);

        DotFunc dotk = get_pomai_dot_kernel();

        for (size_t k = 0; k < layers; ++k)
        {
            uint32_t b = cfg_.bits_per_layer[k];
            uint32_t col0 = layer_offsets_[k];
            out[k].assign(b, 0.0f);

            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                // use optimized dot kernel
                out[k][j] = dotk(col_ptr, query, dim_);
            }
        }
    }

    float EternalEchoQuantizer::approx_dist(const float *query, const EchoCode &code) const
    {
        if (!query)
            throw std::invalid_argument("EternalEchoQuantizer::approx_dist: null query");

        // Simple safe implementation: reconstruct candidate then compute l2sq(query, recon).
        std::vector<float> recon(dim_, 0.0f);
        decode(code, recon.data());
        return l2sq(query, recon.data());
    }

} // namespace pomai::ai