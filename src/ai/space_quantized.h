#pragma once
// ai/space_quantized.h
//
// Simple quantized L2 space that expects payloads composed of quantized bytes
// (uint8 per-dimension or packed 4-bit values).
//
// Refactored to use centralized pomai::config::QuantizedSpaceConfig.

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/quantize.h"
#include "src/core/config.h" // [ADDED]

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cstring>
#include <limits>

namespace pomai::ai
{

    class QuantizedL2SpaceBase
    {
    public:
        virtual ~QuantizedL2SpaceBase() = default;
        virtual size_t dim() const = 0;
        virtual int precision_bits() const = 0;
    };

    template <typename dist_t = float>
    class QuantizedL2Space : public hnswlib::SpaceInterface<dist_t>, public QuantizedL2SpaceBase
    {
        size_t dim_;
        int bits_;                                // 8 or 4
        pomai::config::QuantizedSpaceConfig cfg_; // Store config

    public:
        // [CHANGED] Constructor takes Config
        QuantizedL2Space(size_t dim, const pomai::config::QuantizedSpaceConfig &cfg)
            : dim_(dim), cfg_(cfg)
        {
            if (dim_ == 0)
                throw std::invalid_argument("QuantizedL2Space: dim must be > 0");

            bits_ = static_cast<int>(cfg_.precision_bits);

            if (bits_ != 8 && bits_ != 4)
                throw std::invalid_argument("QuantizedL2Space: only 8 or 4 bits supported");
        }

        size_t dim() const override { return dim_; }
        int precision_bits() const override { return bits_; }
        const pomai::config::QuantizedSpaceConfig &config() const noexcept { return cfg_; }

        // size of payload (without PPEHeader)
        size_t get_data_size() override
        {
            if (bits_ == 8)
                return dim_;
            // bits == 4: packed nibbles
            return (dim_ + 1) / 2;
        }

        // distance function expects pointers to quantized payload (no header)
        // This placeholder is never intended to be called directly; PomaiSpace
        // performs proper resolution/dequantization and calls the underlying L2 kernel.
        static dist_t QuantizedDist(const void * /*p1*/, const void * /*p2*/, const void * /*param*/) noexcept
        {
            // Return maximum distance to indicate "not supported here".
            return std::numeric_limits<dist_t>::max();
        }

        // The DISTFUNC we return will be a wrapper that dequantizes inside PomaiSpace.
        // However HNSW expects a pointer to a function of signature DISTFUNC<dist_t>.
        // We return QuantizedDist as placeholder — real logic handled via PomaiSpace.
        hnswlib::DISTFUNC<dist_t> get_dist_func() override { return &QuantizedDist; }

        void *get_dist_func_param() override { return this; }
    };
}