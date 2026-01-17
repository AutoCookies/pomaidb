#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include "src/core/config.h"

namespace pomai::ai
{

    class ProductQuantizer
    {
    public:
        ProductQuantizer(size_t dim, const pomai::config::PQConfig &cfg);
        ~ProductQuantizer() = default;

        size_t dim() const noexcept { return dim_; }
        size_t m() const noexcept { return m_; }
        size_t k() const noexcept { return k_; }
        size_t subdim() const noexcept { return subdim_; }
        const pomai::config::PQConfig &config() const noexcept { return cfg_; }

        void train(const float *samples, size_t n_samples);
        void encode(const float *vec, uint8_t *out_codes) const;
        std::vector<uint8_t> encode_vec(const float *vec) const;

        static void pack4From8(const uint8_t *src8, uint8_t *dst_nibbles, size_t m);
        static void unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8, size_t m);
        static inline size_t packed4BytesPerVec(size_t m) noexcept { return (m + 1) / 2; }

        bool save_codebooks(const std::string &path) const;
        bool load_codebooks(const std::string &path);

        void compute_distance_tables(const float *query, float *out_tables) const;

        const float *codebooks_data() const noexcept { return codebooks_.empty() ? nullptr : codebooks_.data(); }
        size_t codebooks_float_count() const noexcept { return codebooks_.size(); }
        bool load_codebooks_from_buffer(const float *src, size_t float_count);

    private:
        size_t dim_;
        size_t m_;
        size_t k_;
        size_t subdim_;
        pomai::config::PQConfig cfg_;
        std::vector<float> codebooks_;
    };

} // namespace pomai::ai