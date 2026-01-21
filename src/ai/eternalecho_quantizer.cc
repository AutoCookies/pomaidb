/*
 * src/ai/eternalecho_quantizer.cc
 *
 * Pomai EternalEcho Quantizer (EESQ) - High Performance Implementation
 *
 * Robustified to guard against NaN/Inf, numeric overflow and tiny scales
 * that caused NaNs to appear after repeated residual updates.
 *
 * - Validate inputs (NaN/Inf) early.
 * - Clamp / sanitize norms and scales to safe ranges.
 * - Check residual after each layer; bail out with a clear error if it becomes invalid.
 * - Defensive guards around quantization path and scale decoding consistency.
 * - Fixed approx_dist_code_bytes: now parses code bytes, decodes reconstruction
 *   into a thread-local buffer and computes exact distance using qproj for
 *   the q·recon term. This removes the previous approximate recon-norm heuristic
 *   which could diverge beyond acceptable tolerance.
 */

#include "src/ai/eternalecho_quantizer.h"
#include "src/core/cpu_kernels.h"
#include "src/core/config.h"

#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <limits>

namespace pomai::ai
{
    // -------------------- Helpers --------------------

    static inline bool has_nan_or_inf(const float *v, size_t dim)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            if (!std::isfinite(v[i]))
                return true;
        }
        return false;
    }

    static inline void ensure_finite_or_throw(const float *v, size_t dim, const char *msg)
    {
        if (has_nan_or_inf(v, dim))
            throw std::runtime_error(msg);
    }

    EternalEchoQuantizer::EternalEchoQuantizer(size_t dim, const EternalEchoConfig &cfg, uint64_t seed)
        : dim_(dim), cfg_(cfg), seed_(seed)
    {
        if (dim_ == 0)
            throw std::invalid_argument("EternalEchoQuantizer: dim must be > 0");

        if (cfg_.bits_per_layer.empty())
            throw std::invalid_argument("EternalEchoQuantizer: bits_per_layer cannot be empty");

        size_t layers = std::min<size_t>(cfg_.bits_per_layer.size(), std::max<size_t>(1, cfg_.max_depth));
        cfg_.bits_per_layer.resize(layers);

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

        proj_.resize(static_cast<size_t>(total_bits) * dim_);

        // Deterministic Gaussian initialization
        std::mt19937_64 rng(seed_);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (uint32_t col = 0; col < total_bits; ++col)
        {
            size_t base = static_cast<size_t>(col) * dim_;
            for (size_t d = 0; d < dim_; ++d)
                proj_[base + d] = nd(rng);
        }

        // Precompute per-layer column energy: sum_j ||col_j||^2
        layer_col_energy_.assign(layers, 0.0f);
        for (size_t k = 0; k < layers; ++k)
        {
            uint32_t b = cfg_.bits_per_layer[k];
            uint32_t col0 = layer_offsets_[k];
            double layer_sum = 0.0;
            for (uint32_t j = 0; j < b; ++j)
            {
                uint32_t c = col0 + j;
                size_t base = static_cast<size_t>(c) * dim_;
                double s = 0.0;
                for (size_t d = 0; d < dim_; ++d)
                {
                    double v = proj_[base + d];
                    s += v * v;
                }
                layer_sum += s;
            }
            // guard: ensure non-zero positive value (columns are random gaussian; but be defensive)
            if (!std::isfinite(layer_sum) || layer_sum <= 0.0)
                layer_sum = static_cast<double>(b); // fallback conservative
            layer_col_energy_[k] = static_cast<float>(layer_sum);
        }
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

    float EternalEchoQuantizer::compute_vector_norm(const float *v) const
    {
        return std::sqrt(::pomai_dot(v, v, dim_));
    }

    // -------------------- Encoding / Decoding --------------------

    EchoCode EternalEchoQuantizer::encode(const float *vec) const
    {
        if (!vec)
            throw std::invalid_argument("EternalEchoQuantizer::encode: null vec");

        // Quick input sanity check
        if (has_nan_or_inf(vec, dim_))
            throw std::invalid_argument("EternalEchoQuantizer::encode: input contains NaN or Inf");

        EchoCode code;
        size_t layers = layer_offsets_.size();
        code.bits_per_layer.resize(layers);

        std::vector<float> residual(dim_);
        std::memcpy(residual.data(), vec, dim_ * sizeof(float));

        // compute original norm and sanitize
        float orig_norm = compute_vector_norm(vec);
        if (!std::isfinite(orig_norm) || orig_norm < 1e-9f)
            orig_norm = 1.0f;

        DotFunc dotk = get_pomai_dot_kernel();
        FmaFunc fmak = get_pomai_fma_kernel();

        // per-layer loop
        for (size_t k = 0; k < layers; ++k)
        {
            uint32_t b = cfg_.bits_per_layer[k];
            code.bits_per_layer[k] = b;
            uint32_t col0 = layer_offsets_[k];

            std::vector<float> proj_vals;
            proj_vals.resize(b);
            std::vector<int8_t> signs;
            signs.resize(b);

            // project residual onto columns
            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float accf = dotk(col_ptr, residual.data(), dim_);
                if (!std::isfinite(accf))
                    accf = 0.0f; // defensive fallback
                proj_vals[j] = accf;
                signs[j] = (proj_vals[j] >= 0.0f) ? int8_t(+1) : int8_t(-1);
            }

            // compute scale as average absolute projection; sanitize result
            double sum_abs = 0.0;
            for (uint32_t j = 0; j < b; ++j)
                sum_abs += std::fabs(static_cast<double>(proj_vals[j]));

            float scale = static_cast<float>(sum_abs / static_cast<double>(b));
            // sanitize scale to avoid tiny/NaN/Inf values that propagate to residual
            if (!std::isfinite(scale) || scale < 1e-9f)
                scale = 1e-6f;

            if (cfg_.quantize_scales)
            {
                // clamp scale into [small_eps, scale_quant_max] then quantize
                float s_clamped = std::min(std::max(scale, 1e-6f), cfg_.scale_quant_max);
                uint32_t q = static_cast<uint32_t>(std::round((s_clamped / cfg_.scale_quant_max) * 255.0f));
                if (q > 255)
                    q = 255;
                code.scales_q.push_back(static_cast<uint8_t>(q));
            }
            else
            {
                // store full precision but clamp to safe range
                float s_clamped = std::min(std::max(scale, 1e-6f), 1e12f);
                code.scales_f.push_back(s_clamped);
            }

            // pack sign bits
            std::vector<uint8_t> packed;
            pack_signs_to_bytes(signs, packed);
            code.sign_bytes.push_back(std::move(packed));

            // Apply sign*scale to residual: residual -= scale * sign_j * col_j
            // Defensive: ensure coeff finite for each j
            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float sgn = static_cast<float>(signs[j]);
                float coeff = (cfg_.quantize_scales ? (
                                                          (static_cast<float>(code.scales_q.back()) / 255.0f) * cfg_.scale_quant_max)
                                                    : code.scales_f.back());
                // In case quantization unexpectedly produced zero, fall back to 'scale'
                if (!std::isfinite(coeff) || std::fabs(coeff) < 1e-12f)
                {
                    coeff = scale;
                }
                // subtract scaled column
                // fmak(acc, val, scale, dim) does: acc[i] += val[i] * scale
                // we want residual += col * (-coeff * sgn)
                float cold_coeff = -coeff * sgn;
                fmak(residual.data(), col_ptr, cold_coeff, dim_);
            }

            code.depth = static_cast<uint8_t>(k + 1);

            // Validate residual numerics after update
            if (has_nan_or_inf(residual.data(), dim_))
            {
                // Best-effort: attempt to repair by zeroing tiny infinities / NaNs -> but better to fail loudly
                throw std::runtime_error("EternalEchoQuantizer::encode: residual contains NaN/Inf after layer updates");
            }

            float res_norm = compute_vector_norm(residual.data());
            if (!std::isfinite(res_norm))
            {
                throw std::runtime_error("EternalEchoQuantizer::encode: residual norm became non-finite");
            }

            if (res_norm <= cfg_.stop_threshold * orig_norm)
                break;
        }

        // cleanup: if using full-precision scales flag mismatch, ensure consistency
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

        size_t layers = layer_offsets_.size();
        size_t use_layers = std::min<size_t>(depth, layers);

        FmaFunc fmak = get_pomai_fma_kernel();

        for (size_t k = 0; k < use_layers; ++k)
        {
            uint32_t b = (k < code.bits_per_layer.size()) ? code.bits_per_layer[k] : cfg_.bits_per_layer[k];
            uint32_t col0 = layer_offsets_[k];

            std::vector<int8_t> signs;
            unpack_bytes_to_signs(code.sign_bytes[k], b, signs);

            float scale = 0.0f;
            if (cfg_.quantize_scales)
            {
                if (k < code.scales_q.size())
                    scale = (static_cast<float>(code.scales_q[k]) / 255.0f) * cfg_.scale_quant_max;
            }
            else
            {
                if (k < code.scales_f.size())
                    scale = code.scales_f[k];
            }

            // Defensive clamp on decode-time scale
            if (!std::isfinite(scale) || std::fabs(scale) < 1e-12f)
                scale = 1e-6f;

            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float sgn = static_cast<float>(signs[j]);
                float coeff = scale * sgn;
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
                out[k][j] = dotk(col_ptr, query, dim_);
            }
        }
    }

    // Exact distance (decode into caller-provided scratch_buf and compute l2sq)
    float EternalEchoQuantizer::approx_dist(const float *query, const EchoCode &code, float *scratch_buf) const
    {
        if (!query)
            throw std::invalid_argument("approx_dist: null query");
        if (!scratch_buf)
            throw std::invalid_argument("approx_dist: scratch_buf required");

        decode(code, scratch_buf);
        return ::l2sq(query, scratch_buf, dim_);
    }

    float EternalEchoQuantizer::approx_dist(const float *query, const EchoCode &code) const
    {
        thread_local std::vector<float> tls;
        if (tls.size() < dim_)
            tls.resize(dim_);
        return approx_dist(query, code, tls.data());
    }

    // -------------------------------------------------------------------------
    // approx_dist_code_bytes (robust & consistent)
    // -------------------------------------------------------------------------
    // This implementation parses the compact byte layout used by the server/test
    // (depth, quantized scales, packed sign bytes), reconstructs the EchoCode,
    // decodes the reconstruction into a thread-local buffer and computes the
    // exact squared-L2 distance using qproj for q·recon term when available.
    //
    // This is slightly more expensive than the previous purely-analytic
    // approximation but gives correct and stable scores (no large divergence).
    //
    float EternalEchoQuantizer::approx_dist_code_bytes(
        const std::vector<std::vector<float>> &qproj,
        float qnorm2,
        const uint8_t *code_ptr,
        size_t code_len) const
    {
        if (!code_ptr || code_len == 0)
            return 0.0f;

        // Parse byte layout:
        // [0] depth (1 byte)
        // [1..depth] scales_q (one byte each)  -- test/serialize uses this layout
        // [..] followed by packed sign bytes per layer (ceil(bits/8) each)
        size_t pos = 0;
        uint8_t depth = (pos < code_len) ? code_ptr[pos++] : 0;
        if (depth == 0)
            return qnorm2; // no recon -> distance is qnorm2

        EchoCode code;
        code.depth = depth;
        code.bits_per_layer.resize(depth);
        code.sign_bytes.resize(depth);
        code.scales_q.resize(depth);

        // read scales_q (if truncated, fill with 0)
        for (size_t i = 0; i < depth; ++i)
        {
            if (pos < code_len)
                code.scales_q[i] = code_ptr[pos++];
            else
                code.scales_q[i] = 0;
        }

        // read sign bytes for each layer using cfg_.bits_per_layer to know sizes
        for (size_t k = 0; k < depth; ++k)
        {
            uint32_t b = (k < cfg_.bits_per_layer.size()) ? cfg_.bits_per_layer[k] : 0;
            code.bits_per_layer[k] = b;
            size_t nbytes = (b + 7) / 8;
            if (nbytes == 0)
                continue;
            if (pos + nbytes > code_len)
            {
                // truncated data => stop reading further
                // adjust depth to what we actually parsed
                code.depth = static_cast<uint8_t>(k);
                break;
            }
            code.sign_bytes[k].assign(code_ptr + pos, code_ptr + pos + nbytes);
            pos += nbytes;
        }

        // 1) Compute q·recon term using qproj if available
        double q_dot_recon = 0.0;
        for (size_t k = 0; k < code.depth; ++k)
        {
            if (k >= qproj.size())
                break; // cannot compute qdot for remaining layers
            const auto &layer_qproj = qproj[k];
            uint32_t b = code.bits_per_layer[k];
            const uint8_t *sign_bytes = (code.sign_bytes[k].empty()) ? nullptr : code.sign_bytes[k].data();
            double layer_dot = 0.0;
            if (sign_bytes && b > 0)
            {
                layer_dot = ::pomai_packed_signed_dot(sign_bytes, layer_qproj.data(), b);
            }
            float scale = 0.0f;
            if (cfg_.quantize_scales)
            {
                scale = (static_cast<float>(code.scales_q[k]) / 255.0f) * cfg_.scale_quant_max;
            }
            else
            {
                if (k < code.scales_f.size())
                    scale = code.scales_f[k];
            }
            if (!std::isfinite(scale))
                scale = 0.0f;
            q_dot_recon += layer_dot * static_cast<double>(scale);
        }

        // 2) Decode reconstruction into TLS buffer and compute recon norm exactly
        thread_local std::vector<float> recon_tls;
        if (recon_tls.size() < dim_)
            recon_tls.resize(dim_);
        std::fill(recon_tls.begin(), recon_tls.begin() + dim_, 0.0f);

        // Use decode() to reconstruct into recon_tls (it uses code.scales_q / scales_f and sign_bytes)
        try
        {
            decode(code, recon_tls.data());
        }
        catch (...)
        {
            // If decode fails for any reason, fallback to safe conservative estimate
            // (treat recon as zero -> distance = qnorm2)
            return qnorm2;
        }

        float recon_norm_sq = ::pomai_dot(recon_tls.data(), recon_tls.data(), dim_);

        double dist_sq = static_cast<double>(qnorm2) + static_cast<double>(recon_norm_sq) - 2.0 * q_dot_recon;

        if (!std::isfinite(dist_sq) || dist_sq < 0.0)
        {
            if (dist_sq < 0.0 && dist_sq > -1e-3)
                dist_sq = 0.0;
            else if (!std::isfinite(dist_sq))
                dist_sq = static_cast<double>(qnorm2); // fallback
        }

        return static_cast<float>(dist_sq);
    }

} // namespace pomai::ai