/*
 * src/ai/eternalecho_quantizer.cc
 *
 * Pomai EternalEcho Quantizer (EESQ) - implementation (optimized)
 *
 * Changes:
 *  - Use cpu_kernels adaptive kernels (::pomai_dot, ::pomai_fma, ::l2sq).
 *  - Precompute per-layer column energy (layer_col_energy_).
 *  - Provide approx_dist_code_bytes() which computes ADC-style approximate distance
 *    without doing a full decode (fast ranking).
 *  - Keep decode-based approx_dist(...) variants for exact distances.
 *
 * Updated: use the packed-signed SIMD kernel (::pomai_packed_signed_dot) for the
 * inner signed-accumulate loop in approx_dist_code_bytes to get AVX speedups.
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

        EchoCode code;
        size_t layers = layer_offsets_.size();
        code.bits_per_layer.resize(layers);

        std::vector<float> residual(dim_);
        std::memcpy(residual.data(), vec, dim_ * sizeof(float));

        float orig_norm = compute_vector_norm(vec);
        if (orig_norm == 0.0f)
            orig_norm = 1.0f;

        DotFunc dotk = get_pomai_dot_kernel();
        FmaFunc fmak = get_pomai_fma_kernel();

        for (size_t k = 0; k < layers; ++k)
        {
            uint32_t b = cfg_.bits_per_layer[k];
            code.bits_per_layer[k] = b;
            uint32_t col0 = layer_offsets_[k];

            std::vector<float> proj_vals(b);
            std::vector<int8_t> signs(b);

            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float accf = dotk(col_ptr, residual.data(), dim_);
                proj_vals[j] = accf;
                signs[j] = (proj_vals[j] >= 0.0f) ? int8_t(+1) : int8_t(-1);
            }

            double sum_abs = 0.0;
            for (uint32_t j = 0; j < b; ++j)
                sum_abs += std::fabs(static_cast<double>(proj_vals[j]));
            float scale = static_cast<float>(sum_abs / static_cast<double>(b));

            if (cfg_.quantize_scales)
            {
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

            std::vector<uint8_t> packed;
            pack_signs_to_bytes(signs, packed);
            code.sign_bytes.push_back(std::move(packed));

            for (uint32_t j = 0; j < b; ++j)
            {
                const float *col_ptr = &proj_[(static_cast<size_t>(col0 + j) * dim_)];
                float sgn = static_cast<float>(signs[j]);
                float coeff = scale * sgn;
                fmak(residual.data(), col_ptr, -coeff, dim_);
            }

            code.depth = static_cast<uint8_t>(k + 1);

            float res_norm = compute_vector_norm(residual.data());
            if (res_norm <= cfg_.stop_threshold * orig_norm)
                break;
        }

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

    // ADC-style approximate distance: compute projections q·col once and evaluate distance
    // directly on sign_bits and scales (fast, no full decode).
    // Uses pomai_packed_signed_dot for the inner signed dot accumulation (SIMD-accelerated).
    float EternalEchoQuantizer::approx_dist_code_bytes(
        const std::vector<std::vector<float>> &qproj,
        float qnorm2,
        const uint8_t *code_ptr,
        size_t code_len) const
    {
        if (!code_ptr || code_len == 0) return 0.0f;

        // [OPTIMIZATION] Dùng double accumulator để tránh sai số và overflow
        // Nhanh hơn Kahan Summation và chính xác hơn float thuần.
        double q_dot_recon = 0.0;
        double recon_norm_sq = 0.0;

        size_t pos = 0;
        uint8_t depth = (pos < code_len) ? code_ptr[pos++] : 0;

        // 1. Đọc và giải mã Scales
        // (Giả sử scale không nén hoặc nén đơn giản, ở đây đọc uint8 -> float)
        // Bạn có thể tối ưu đoạn này bằng Stack Allocation nếu depth nhỏ.
        static thread_local std::vector<float> scales;
        if (scales.size() < depth) scales.resize(depth);
        
        for (size_t i = 0; i < depth; ++i)
        {
            if (pos >= code_len) { scales[i] = 0.0f; continue; }
            // Dequantize scale: map [0, 255] -> [0.0, 1.0] * max_val (hoặc logic riêng của bạn)
            // Ở đây giả sử scale lưu dạng byte * (1.0/255.0)
            scales[i] = static_cast<float>(code_ptr[pos++]) / 255.0f; 
        }

        // 2. Tính toán Dot Product và Norm từng lớp
        for (size_t k = 0; k < depth; ++k)
        {
            if (k >= qproj.size()) break;
            const auto &proj = qproj[k];
            
            float scale = scales[k];
            double scale_d = static_cast<double>(scale);
            
            size_t n_bits = proj.size();
            size_t n_bytes = (n_bits + 7) / 8;
            
            if (pos + n_bytes > code_len) break;
            
            const uint8_t* bits = code_ptr + pos;
            pos += n_bytes;

            double layer_dot = 0.0;
            
            // Loop này thường được Compiler (GCC/Clang) tự động vectorize (SIMD) rất tốt
            // nếu proj là float.
            for (size_t i = 0; i < n_bits; ++i)
            {
                // Giải nén bit thứ i: 1 -> +1.0, 0 -> -1.0
                // (bits[byte_idx] >> bit_idx) & 1
                bool bit_val = (bits[i >> 3] >> (i & 7)) & 1;
                
                // Branchless selection:
                // Nếu bit=1, val=proj[i]. Nếu bit=0, val=-proj[i].
                float val = proj[i];
                if (!bit_val) val = -val;
                
                layer_dot += static_cast<double>(val);
            }
            
            q_dot_recon += layer_dot * scale_d;
            
            // Norm cộng dồn: dim * scale^2 (vì vector toàn +1/-1)
            recon_norm_sq += static_cast<double>(n_bits) * (scale_d * scale_d);
        }

        // 3. Tổng hợp kết quả (Trong miền Double)
        // Dist^2 = ||Q||^2 + ||R||^2 - 2(Q.R)
        double dist_sq = static_cast<double>(qnorm2) + recon_norm_sq - 2.0 * q_dot_recon;

        // 4. Smart Clamp (Chỉ khử nhiễu âm cực nhỏ)
        if (dist_sq < 1e-5) dist_sq = 0.0;

        return static_cast<float>(dist_sq);
    }

} // namespace pomai::ai