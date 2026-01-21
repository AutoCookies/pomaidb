#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "src/core/config.h"

namespace pomai::ai
{
    class SimHash;

    class FingerprintEncoder
    {
    public:
        virtual ~FingerprintEncoder() = default;
        virtual size_t bytes() const noexcept = 0;
        virtual void compute(const float *vec, uint8_t *out_bytes) const = 0;
        virtual void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const = 0;
        static std::unique_ptr<FingerprintEncoder> createSimHash(
            size_t dim,
            const pomai::config::FingerprintConfig &cfg,
            uint64_t seed = 123456789ULL);
        static std::unique_ptr<FingerprintEncoder> createOPQSign(
            size_t dim,
            const pomai::config::FingerprintConfig &cfg,
            const std::string &rotation_path = std::string(),
            uint64_t seed = 123456789ULL);
    };

} // namespace pomai::ai