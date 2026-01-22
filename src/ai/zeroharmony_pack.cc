#include "src/ai/zeroharmony_pack.h"
#include "src/core/cpu_kernels.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace pomai::ai
{
    static inline uint16_t f32_to_f16(float x)
    {
        uint32_t f;
        std::memcpy(&f, &x, 4);
        uint32_t s = (f >> 31) & 0x1;
        uint32_t e = (f >> 23) & 0xFF;
        uint32_t m = f & 0x7FFFFF;
        if (e == 0)
            return (uint16_t)(s << 15);
        if (e == 255)
            return (uint16_t)((s << 15) | 0x7C00 | (m ? 1 : 0));
        int ne = (int)e - 127 + 15;
        if (ne >= 31)
            return (uint16_t)((s << 15) | 0x7C00);
        if (ne <= 0)
            return (uint16_t)(s << 15);
        return (uint16_t)((s << 15) | (ne << 10) | (m >> 13));
    }

    static inline float f16_to_f32(uint16_t h)
    {
        uint32_t s = (h >> 15) & 0x1;
        uint32_t e = (h >> 10) & 0x1F;
        uint32_t m = h & 0x3FF;
        uint32_t f;
        if (e == 0)
            f = (s << 31) | ((m != 0) ? ((127 - 15) << 23) | (m << 13) : 0);
        else if (e == 31)
            f = (s << 31) | 0x7F800000 | (m << 13);
        else
            f = (s << 31) | ((e + 127 - 15) << 23) | (m << 13);
        float out;
        std::memcpy(&out, &f, 4);
        return out;
    }

    ZeroHarmonyPacker::ZeroHarmonyPacker(const pomai::config::ZeroHarmonyConfig &cfg, size_t dim)
        : cfg_(cfg), dim_(dim) {}

    void ZeroHarmonyPacker::compute_mean(const float *vecs, size_t n, std::vector<float> &mean) const
    {
        mean.assign(dim_, 0.0f);
        if (n == 0)
            return;
        for (size_t i = 0; i < n; ++i)
        {
            const float *v = vecs + i * dim_;
            for (size_t d = 0; d < dim_; ++d)
                mean[d] += v[d];
        }
        float inv = 1.0f / static_cast<float>(n);
        for (size_t d = 0; d < dim_; ++d)
            mean[d] *= inv;
    }

    std::vector<uint8_t> ZeroHarmonyPacker::pack_with_mean(const float *vec, const std::vector<float> &mean) const
    {
        std::vector<uint8_t> out;
        out.reserve(dim_ * 5);
        uint32_t zrun = 0;

        for (size_t d = 0; d < dim_; ++d)
        {
            float delta = vec[d] - mean[d];
            if (std::abs(delta) <= cfg_.zero_threshold)
            {
                zrun++;
            }
            else
            {
                while (zrun > 0)
                {
                    uint8_t r = (zrun > 255) ? 255 : (uint8_t)zrun;
                    out.push_back(0);
                    out.push_back(r);
                    zrun -= r;
                }
                out.push_back(1);
                if (cfg_.use_half_nonzero)
                {
                    uint16_t h = f32_to_f16(delta);
                    out.push_back(h & 0xFF);
                    out.push_back(h >> 8);
                }
                else
                {
                    uint32_t f;
                    std::memcpy(&f, &delta, 4);
                    out.push_back(f & 0xFF);
                    out.push_back((f >> 8) & 0xFF);
                    out.push_back((f >> 16) & 0xFF);
                    out.push_back(f >> 24);
                }
            }
        }
        while (zrun > 0)
        {
            uint8_t r = (zrun > 255) ? 255 : (uint8_t)zrun;
            out.push_back(0);
            out.push_back(r);
            zrun -= r;
        }
        return out;
    }

    bool ZeroHarmonyPacker::unpack_to(const uint8_t *p, size_t len, const std::vector<float> &mean, float *out) const
    {
        if (mean.size() != dim_)
            return false;
        for (size_t i = 0; i < dim_; ++i)
            out[i] = mean[i];

        size_t pos = 0, d = 0;
        while (pos < len && d < dim_)
        {
            uint8_t tag = p[pos++];
            if (tag == 0)
            {
                if (pos >= len)
                    return false;
                uint8_t run = p[pos++];
                d += run;
            }
            else
            {
                float delta;
                if (cfg_.use_half_nonzero)
                {
                    if (pos + 2 > len)
                        return false;
                    uint16_t h = (uint16_t)p[pos] | ((uint16_t)p[pos + 1] << 8);
                    delta = f16_to_f32(h);
                    pos += 2;
                }
                else
                {
                    if (pos + 4 > len)
                        return false;
                    uint32_t f = (uint32_t)p[pos] | ((uint32_t)p[pos + 1] << 8) | ((uint32_t)p[pos + 2] << 16) | ((uint32_t)p[pos + 3] << 24);
                    std::memcpy(&delta, &f, 4);
                    pos += 4;
                }
                if (d < dim_)
                    out[d] = mean[d] + delta;
                d++;
            }
        }
        return (d == dim_) || (pos == len);
    }

    float ZeroHarmonyPacker::approx_dist(const float *q, const uint8_t *p, size_t len, const std::vector<float> &mean) const
    {
        thread_local std::vector<float> tls;
        if (tls.size() < dim_)
            tls.resize(dim_);
        if (!unpack_to(p, len, mean, tls.data()))
            return std::numeric_limits<float>::infinity();
        return ::pomai::core::kernels_internal::impl_l2sq(q, tls.data(), dim_);
    }

    float ZeroHarmonyPacker::approx_dist_with_cutoff(const float *q, const uint8_t *p, size_t len, const std::vector<float> &mean, float cutoff) const
    {
        // compute squared L2 incrementally; early exit if acc > cutoff
        float acc = 0.0f;
        size_t pos = 0;
        size_t d = 0;
        while (pos < len && d < dim_)
        {
            uint8_t tag = p[pos++];
            if (tag == 0)
            {
                if (pos >= len)
                    return std::numeric_limits<float>::infinity();
                uint8_t run = p[pos++];
                // zeros: each increases acc by (q[d]-mean[d])^2 but delta=0 -> contribution = (q[d]-mean[d])^2
                for (uint8_t r = 0; r < run && d < dim_; ++r, ++d)
                {
                    float diff = q[d] - mean[d];
                    acc += diff * diff;
                    if (acc > cutoff)
                        return std::numeric_limits<float>::infinity();
                }
            }
            else
            {
                float delta = 0.0f;
                if (cfg_.use_half_nonzero)
                {
                    if (pos + 2 > len)
                        return std::numeric_limits<float>::infinity();
                    uint16_t h = (uint16_t)p[pos] | ((uint16_t)p[pos + 1] << 8);
                    delta = f16_to_f32(h);
                    pos += 2;
                }
                else
                {
                    if (pos + 4 > len)
                        return std::numeric_limits<float>::infinity();
                    uint32_t f = (uint32_t)p[pos] | ((uint32_t)p[pos + 1] << 8) | ((uint32_t)p[pos + 2] << 16) | ((uint32_t)p[pos + 3] << 24);
                    std::memcpy(&delta, &f, 4);
                    pos += 4;
                }
                if (d < dim_)
                {
                    float v = mean[d] + delta;
                    float diff = q[d] - v;
                    acc += diff * diff;
                    if (acc > cutoff)
                        return std::numeric_limits<float>::infinity();
                }
                ++d;
            }
        }
        // finish remaining dims if any (shouldn't be many)
        while (d < dim_)
        {
            float diff = q[d] - mean[d];
            acc += diff * diff;
            if (acc > cutoff)
                return std::numeric_limits<float>::infinity();
            ++d;
        }
        return acc;
    }
}