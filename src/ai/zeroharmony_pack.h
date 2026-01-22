#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "src/core/config.h"

namespace pomai::ai
{
    class ZeroHarmonyPacker
    {
    public:
        explicit ZeroHarmonyPacker(const pomai::config::ZeroHarmonyConfig &cfg, size_t dim);
        ~ZeroHarmonyPacker() = default;

        void compute_mean(const float *vectors, size_t n_vecs, std::vector<float> &out_mean) const;
        std::vector<uint8_t> pack_with_mean(const float *vec, const std::vector<float> &mean) const;
        bool unpack_to(const uint8_t *packed, size_t len, const std::vector<float> &mean, float *out) const;
        float approx_dist(const float *query, const uint8_t *packed, size_t len, const std::vector<float> &mean) const;
        float approx_dist_with_cutoff(const float *q, const uint8_t *p, size_t len, const std::vector<float> &mean, float cutoff) const;
        
        size_t dim() const noexcept { return dim_; }

    private:
        pomai::config::ZeroHarmonyConfig cfg_;
        size_t dim_;
    };
}